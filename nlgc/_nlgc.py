# author: proloy Das <proloy@umd.edu>
# author: Behrad Soleimani <behrad@umd.edu>

import itertools
from multiprocessing import cpu_count, current_process

import scipy
import copy
import logging
import numpy as np
import pickle
import warnings
from functools import reduce
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import mne
from mne import (Forward, Label)
from mne.forward import is_fixed_orient
from mne.inverse_sparse.mxne_inverse import _prepare_gain
from mne.minimum_norm.inverse import _check_reference
from mne.source_estimate import (_prepare_label_extraction, _BaseVolSourceEstimate, _BaseVectorSourceEstimate,
                                 SourceEstimate,
                                 MixedSourceEstimate, VolSourceEstimate)
from mne.source_space import SourceSpaces
from mne.utils import (logger, _check_option, _validate_type)
from scipy import linalg, sparse

from .opt import *
from ._stat import fdr_control
from ._bias_utils import debias_deviances
from ._gen_utils import LazyProperty


_default_lambda_range = np.asanyarray([5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, ])


class NLGC:
    """NLGC object

    Provides a an object including captured connectivity map via NLGC and its related parameters.

    Parameters
    ----------
    subject: str
        subject_id
    nx: int
        n_sources
    ny: int
        n_sensors
    t: int
        n_samples
    p: int
        VAR model order
    n_eigenmodes: int
        number of eigenmodes
    n_segments: int
        number of chunks used for non-centrality parameter estimation
    d_raw: numpy array (n_sources * n_sources)
        *biased* deviance matrix
    bias_f: float
        full model bias (scalar)
    biar_r: numpy array (n_sources * n_sources)
        reduced model bias matrix, [.]_{i,j} corresponds to link j->i
    """
    def __init__(self, subject, nx, ny, t, p, n_eigenmodes, n_segments, d_raw, bias_f, bias_r,
                 model_f, conv_flag, label_names, label_vertidx, forward_orig, cov_orig, whitener, eig_src_weights, debug=None):

        self.subject = subject
        self.nx = nx
        self.ny = ny
        self.t = t
        self.p = p
        self.n_eigenmodes = n_eigenmodes
        self.n_segments = n_segments
        self.d_raw = d_raw
        self.bias_f = bias_f
        self.bias_r = bias_r
        self._model_f = model_f
        self._conv_flag = conv_flag
        self._labels = label_names
        self._label_vertidx = label_vertidx
        self.forward_orig = forward_orig
        self.cov_orig = cov_orig
        self.whitener = whitener
        self.eig_src_weights = eig_src_weights
        self._debug = debug

    def _plot_reduced_models_convergence(self, max_itr=1):
        fig, ax = plt.subplots(self.n_segments)
        for conv_, ax_ in zip(self._conv_flag, ax):
            ax_.hist(np.reshape(conv_ / max_itr, (1, self.nx ** 2)).T)

        return fig, ax

    @LazyProperty
    def avg_debiased_dev(self):
        """averaging the calculted deviances across chunks (n_segments)

            """
        debiased_deviances = [debias_deviances(*args) for args in zip(self.d_raw, self.bias_f, self.bias_r)]
        if self.n_segments > 1:
            return reduce(lambda x, y: x + y, debiased_deviances) / self.n_segments
        else:
            return debiased_deviances[0]

    def get_J_statistics(self, alpha=0.1):
        """calculating J-stat (connectivity map) from deviance matrix

        Parameters
        ----------
        alpha : float
            individual-level confidence interval
            """
        return fdr_control(self.avg_debiased_dev, self.p * (self.n_eigenmodes**2), alpha)

    def pickle_as(self, filename):
        """saving the object as a pickle

        Parameters
        ----------
        filename : str
            file name (including directory address)
            """
        if filename.endswith('.pkl') or filename.endswith('.pickled') or filename.endswith('.pickle'):
            pass
        else:
            filename += '.pkl'

        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def plot(self):
        pass


def nlgc_map(name, evoked, forward, noise_cov, labels, order, self_history=None, n_eigenmodes=2, alpha=0.0, beta=0.0,
        patch_idx=[], n_segments=1, loose=0.0, depth=0.0, pca=True, rank=None, lambda_range=None, lambda1=None, lambda2=None,
        max_iter=500, max_cyclic_iter=3, tol=1e-5, sparsity_factor=0.0, cv=5, use_lapack=True, use_es=True, var_thr=1.0, verbose=False):
    """NLGC connectivity map estimation

    This function estimates the causal connectivity map across sources given the MEG measurements, forward model,
    measurement noise covariance matrix, and a few model-related parameters.

    Parameters
    ----------
    name: str
        subject's name
    evoked: mne.Evoked
        MEG evoked response in MNE-python standard format
    forward: mne.Forward
        forward solution in MNE-python standard format
    noise_cov: mne.Covariance
        measurement noise covariance matrix (could be obtained from empty room or base-line recordings)
    labels: mne.SourceSpaces | mne.Forward | mne.Labels
        source space, forward solution, or list of labels, all in MNE-python standard format
    order: int
        VAR model order
    self_history: int | None
        number of removed self-history lags in VAR model to mitigate possible overfitting (a[:self_history,i,i]=0)
        (default = None)
    n_eigenmodes: int
        number of eignemodes
    alpha: int | float
        Inv-Gamma(alpha*t/2 - 1, beta*t) prior on the state noise covariance matrix
    beta: int | float
        Inv-Gamma(alpha*t/2 - 1, beta*t) prior on the state noise covariance matrix
    patch_idx: list | None
        subset of patch indices to find the connectivity within them (None = whole source space)
    n_segments: int
        number of segments which divides the MEG recording into equal parts for non-centrality parameter estimation
    {loose, depth, pca, rank}: float/boolean
        forward model computation parameters, check mne.inverse_sparse.mxne_inverse for more info
    lambda_range: numpy 1d array
        an array of the regularization coefficients for cross-validation
    max_iter: int
        maximum number of iterations for EM-based parameter estimation
    max_cyclic_iter: int
        maximum number of cyclic iterations to update VAR coefficients (A's) and covariance (q's)
    tol: float
        tolerance for EM convergence (in terms of relative jump of log-likelihood function)
    sparsity_factor: float
        the threshold to remove reduced models with sufficiently small VAR coefficients in their corresponding
        full models for speeding up the calculations (None = all possible reduced models)
    cv: int
        number of folds used for cross-validation
    use_es: boolean
        if True, uses estimation stability for CV metric, otherwise it uses log-likelihood value; check this for
            more info: https://doi.org/10.1080/10618600.2015.1020159 (ESVC)
    var_thr: float
        the threshold to limit the number of reduced models by considering only the possible links between the active
        sources which explain 'var_thr' of the total power
        (default = 1, i.e., all sources)
    

    Returns
    -------
    nlgc_obj : NLGC object
        contains the connectivity map and the some related parameters (see NLGC class for more info)
    """

    _check_reference(evoked)

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    weights, G, label_vertidx, label_names, gain_info, whitener = \
        _prepare_eigenmodes(evoked, forward, noise_cov, labels, n_eigenmodes, loose, depth, pca, rank)

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    if verbose:
        print('Whitening data matrix.')

    M = np.dot(whitener, M)

    # Normalization
    M_normalizing_factor = linalg.norm(np.dot(M, M.T) / M.shape[1], ord='fro')
    G_normalizing_factor = np.sqrt(np.sum(G ** 2, axis=0))
    G /= G_normalizing_factor
    # G *= np.sqrt(M_normalizing_factor)
    M /= np.sqrt(M_normalizing_factor)
    r = 1 / M_normalizing_factor

    if len(patch_idx) == 0:
        raise ValueError("Length of patch_idx should not be zero")

    n, _ = G.shape
    n, nnx = G.shape
    nx = nnx // n_eigenmodes
    _, t = M.shape
    tt = t // n_segments

    d_raw = np.zeros((n_segments, nx, nx))
    bias_r = np.zeros((n_segments, nx, nx))
    bias_f = np.zeros((n_segments, 1))
    conv_flag = np.zeros((n_segments, nx, nx))

    models = []

    for this_segment in range(0, n_segments):
        if verbose:
            print('Segment: ', this_segment + 1)
            print(f"nlgc_map max iter = {max_iter}")
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            _gc_extraction(M[:, this_segment * tt: (this_segment + 1) * tt], G, r, p=order, p1=self_history,
                           n_eigenmodes=n_eigenmodes,
                           ROIs=patch_idx,
                           alpha=alpha, beta=beta, cv=cv, lambda_range=lambda_range, lambda1=lambda1, 
                           lambda2=lambda2, max_iter=max_iter,
                           max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor,
                           use_lapack=use_lapack, use_es=use_es, var_thr=var_thr, verbose=verbose)
        d_raw[this_segment] = d_raw_
        bias_r[this_segment] = bias_r_
        bias_f[this_segment] = bias_f_
        models.append(model_f)
        conv_flag[this_segment] = conv_flag_

    nlgc_obj = NLGC(name, nx, n, t, order, n_eigenmodes, n_segments, d_raw, bias_f, bias_r, models,
                    conv_flag, label_names, label_vertidx, forward, noise_cov, whitener, weights)

    return nlgc_obj


def _gc_extraction(y, f, r, p, p1, n_eigenmodes=2, var_thr=1.0, ROIs=[], alpha=0, beta=0,
        lambda_range=None, lambda1=None, lambda2=None, max_iter=500, max_cyclic_iter=3,
        tol=1e-5, sparsity_factor=0.0, cv=5, use_lapack=True, use_es=True, verbose=False):
    n, m = f.shape
    nx = m // n_eigenmodes

    if lambda1 is not None:
        assert(lambda2 is not None)
    if lambda2 is not None:
        assert(lambda1 is not None)
        if verbose:
            print("individual lambdas specified for a and b coeffcients, ignoring lambda range")

    kwargs = {
        'use_es': use_es,
        'alpha': alpha,
        'beta': beta,
        'max_iter': max_iter,
        'max_cyclic_iter': max_cyclic_iter,
        'rel_tol': tol,
        'verbose':verbose
    }

    # learn the full model
    n_jobs = cv if isinstance(cv, int) else cv.get_n_splits()
    n_jobs = min(n_jobs, cpu_count())

    if lambda_range is None:
        lambda_range = _default_lambda_range

    e, u = linalg.eigh(f.dot(f.T))
    temp = u.T.dot(y)
    c = (temp ** 2).sum(axis=1)
    from scipy import optimize
    fun = lambda x: (c / (1 + x * e) ** 2).sum() - 1.2 * n * y.shape[1]
    fprime = lambda x: - 2 * ((c * e) / (1 + x * e) ** 3).sum()

    if fun(0) > 0:
        q_val = optimize.newton(fun, 1)
    else:
        q_val = 0.0001
    q_init = q_val * np.eye(m)
    a_init = None

    if len(lambda_range) > 1 and lambda1 is None:
        model_f = NeuraLVARCV(p, p1, n_eigenmodes, 10, cv, n_jobs, use_lapack=use_lapack)
        model_f.fit(y, f, r * np.eye(n), lambda_range, a_init=a_init, q_init=q_init.copy(), **kwargs)
    else:
        model_f = NeuraLVAR(p, p1, n_eigenmodes, use_lapack=use_lapack)
        lambda_range = lambda_range[0]
        model_f.fit(y, f, r * np.eye(n), lambda_range, lb=lambda1, la=lambda2, a_init=a_init, q_init=q_init.copy(), **kwargs)

    bias_f = model_f.compute_bias(y)

    warnings.filterwarnings('ignore')

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    conv_flag = np.zeros((nx, nx), dtype=np.bool_)

    # learn reduced models
    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    lambda_f = model_f.lambda_

    sparsity = np.linalg.norm(model_f._parameters[0], axis=0, ord=1) * np.diag(model_f._parameters[2])[None, :]

    if var_thr < 1:
        x_ = np.sum(model_f._parameters[4][:, :m] ** 2, axis=0)
        total_power = np.zeros(m // n_eigenmodes)
        for n in range(n_eigenmodes):
            total_power += x_[n::n_eigenmodes]
        sorted_idx = np.argsort(total_power)[::-1]
        sorted_pow_ratio = np.cumsum(total_power[sorted_idx])
        sorted_pow_ratio /= sorted_pow_ratio[-1]
        idx = ((sorted_pow_ratio > var_thr) != 0).argmax()
        ROIs = sorted_idx[:idx + 1]

    links_to_check = []
    for i, j in itertools.product(ROIs, repeat=2):
        # Exclude i == j cases
        if i == j:
            continue
        # Exclude small cross-regression cases
        target = _expand_roi_indices_as_tup(j, n_eigenmodes)
        source = _expand_roi_indices_as_tup(i, n_eigenmodes)
        if sparsity[target, source].sum() <= sparsity_factor * sparsity[target, target].sum():
            continue
        # Append rest of the links to check
        links_to_check.append((j, i))

    if verbose:
        print(f"Checking {len(links_to_check)} links...")

    # Memory management for Parallel implementation
    shared_y, info_y, shm_y = create_shared_mem(y)
    shared_f, info_f, shm_f = create_shared_mem(f)
    shared_bias_r, info_bias_r, shm_bias_r = create_shared_mem(bias_r)
    shared_ll_r, info_ll_r, shm_ll_r = create_shared_mem(dev_raw)
    shared_conv_flag, info_conv_flag, shm_conv_flag = create_shared_mem(dev_raw)
    shared_args = (info_y, info_f, info_bias_r, info_ll_r, info_conv_flag)  # shared memory
    args = (r, lambda_f, a_f, q_f, p, p1, n_eigenmodes, use_lapack)  # can be passed directly

    # Parallel
    if len(links_to_check) == 0:
        bias_f = 0
    else:
        n_jobs = min(cpu_count(), len(links_to_check))
        Parallel(n_jobs=n_jobs, verbose=10)\
            (delayed(_learn_reduced_model_parallel)(link, *(shared_args + args), **kwargs) for link in links_to_check)
        # # serial
        # [_learn_reduced_model_parallel(link, *(shared_args + args), ** kwargs) for link in links_to_check]

        ll_r = np.reshape(shared_ll_r, dev_raw.shape).copy()
        bias_r = np.reshape(shared_bias_r, dev_raw.shape).copy()
        conv_flag = np.reshape(shared_conv_flag, dev_raw.shape).copy()
        for shm in (shm_conv_flag, shm_bias_r, shm_f, shm_ll_r, shm_y):
            shm.close()
            try:
                shm.unlink()
            except:
                print(f"\nUnlink shared-memory issue!")

        indices = tuple(z for z in zip(*links_to_check))
        dev_raw[indices] = 2 * model_f.ll
        dev_raw[indices] -= 2 * ll_r[indices]

    # # Old log ratio implementation
    # dev_raw_[j, i] = sum(map(lambda x: np.log(model_r._parameters[2][x, x]) - np.log(model_f._parameters[2][x, x]),
    #                          target))
    # # if dev_raw_[j, i] < 0:
    # #     warnings.filterwarnings('ignore')
    # bias_r_[j, i] = model_r.compute_bias_idx(y, target)
    # bias_f_[j, i] = model_f.compute_bias_idx(y, target)
    return dev_raw, bias_r, bias_f, model_f, conv_flag


def _learn_reduced_model(i, j, y, f, r, lambda_f, a, q, n, p, p1, n_eigenmodes, use_lapack, alpha, beta,
        **kwargs):
    target = _expand_roi_indices_as_tup(j, n_eigenmodes)
    source = _expand_roi_indices_as_tup(i, n_eigenmodes)
    link = '->'.join(map(lambda x: ','.join(map(str, x)), (source, target)))
    a_init = a.copy()
    a_init[:, target, source] = 0.0
    model_r = NeuraLVAR(p, p1, n_eigenmodes, use_lapack=use_lapack)
    if isinstance(r, list):
        assert(len(r) == 2)
        cov = scipy.linalg.block_diag(r[0] * np.eye(n//2), r[1] * np.eye(n//2))
    else:
        cov = r * np.eye(n)
    model_r.fit(y, f, cov, lambda_f, a_init=a_init, q_init=q.copy(), restriction=link, alpha=alpha,
                beta=beta, **kwargs)
    bias = model_r.compute_bias(y)
    ll = model_r.ll
    conv_flag = len(model_r._lls[0]) == kwargs['max_iter']
    return ll, bias, conv_flag


def _learn_reduced_model_parallel(link_index, info_y, info_f, info_bias_r, info_ll_r, info_conv_flag, r, lambda_f, a, q,
        p, p1, n_eigenmodes, use_lapack, alpha, beta, **kwargs):
    try:
        y, shm_y = link_share_memory(info_y)
        f, shm_f = link_share_memory(info_f)
        bias_r, shm_bias_r = link_share_memory(info_bias_r)
        ll_r, shm_ll_r = link_share_memory(info_ll_r)
        conv_flag, shm_conv_flag = link_share_memory(info_conv_flag)
    except BaseException as e:
        logger.error("Could not link to memory")
        raise e

    n = f.shape[0]
    j, i = link_index
    logger.debug(f"{current_process().name} working on {i, j}th link")
    ll, bias, flag = _learn_reduced_model(i, j, y, f, r, lambda_f, a, q, n, p, p1, n_eigenmodes, use_lapack,
                                          alpha, beta, **kwargs)
    ll_r[j, i] = ll
    bias_r[j, i] = bias
    conv_flag[j, i] = flag
    for shm in (shm_y, shm_f, shm_bias_r, shm_ll_r, shm_conv_flag):
        shm.close()


def _prepare_eigenmodes(evoked, forward, noise_cov, labels, n_eigenmodes=2, loose=0.0, depth=0.0, pca=True, rank=None,
        mode='svd_flip'):
    depth_dict = {'exp': depth, 'limit_depth_chs': 'whiten', 'combine_xyz': 'fro', 'limit': None}

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(forward, evoked.info, noise_cov, pca,
                                                                               depth_dict, loose, rank)

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    # whiten the data
    logger.info('Whitening data matrix.')
    if isinstance(labels, Forward):
        weights, G, label_vertidx, src_flip = _reduce_lead_field(forward, labels, n_eigenmodes, data=gain.T)
        label_names = []
        for i, label in enumerate(labels['src']):
            label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
    elif isinstance(labels, SourceSpaces):
        weights, G, label_vertidx, src_flip = _reduce_lead_field(forward, labels, n_eigenmodes, data=gain.T)
        label_names = []
        for i, label in enumerate(labels):
            label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
    elif isinstance(labels, list):
        if isinstance(labels[0], Label):
            weights = None # not implemented
            G, label_vertidx, src_flip = _extract_label_eigenmodes(forward, labels, gain.T, mode, n_eigenmodes,
                                                                   allow_empty=True)
            label_names = [label.name for label in labels]
        else:
            raise ValueError('Not supported {:s}: elements of labels are expected to be mne.Labels, '
                             'if a list is provided.'.format(type(labels[0])))
    else:
        raise ValueError('Not supported {:s}: labels are expected to be either an mne.SourceSpace or'
                         'mne.Forward object or list of mne.Labels.'.format(labels))

    # test if there are empty columns
    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
    label_vertidx = [i for select, i in zip(sel, label_vertidx) if select]
    src_flip = [i for select, i in zip(sel, src_flip) if select]
    discarded_labels = []
    j = 0
    for i, sel_ in enumerate(sel[::n_eigenmodes]):
        if not sel_:
            discarded_labels.append(labels.pop(i - j))
            label_vertidx.pop(i - j)
            j += 1
    assert j == len(discarded_labels)
    if j > 0:
        logger.info('No sources were found in following {:d} ROIs:\n'.format(len(discarded_labels)) +
                    '\n'.join(map(lambda x: str(x.name), discarded_labels)))

    return weights, G, label_vertidx, label_names, gain_info, whitener


def _reduce_lead_field(forward, src, n_eigenmodes, data=None):
    import mne
    if data is None:
        logger.info('Using the raw forward solution')
        data = np.swapaxes(forward['sol']['data'], 0, 1)  # (n_sources, n_channels)
    data = data.copy()

    if isinstance(src, mne.Forward):
        src = src['src']

    grouped_vertidx_no_offset, grouped_vertidx, n_groups, n_verts = _prepare_leadfield_reduction(src, forward['src'])
    group_eigenmodes = np.zeros((sum(n_groups) * n_eigenmodes,) + data.shape[1:], dtype=data.dtype)
    
    lhweights = []
    rhweights = []
    
    for i, (this_grouped_vertidx, this_grouped_vertidx_no_offset) in \
                enumerate(zip(grouped_vertidx, grouped_vertidx_no_offset)):
        eig_src_weights, this_group_eigenmodes, percentage_explained = _truncatedsvd(data[this_grouped_vertidx], n_eigenmodes, return_pecentage_exaplained=True)
        print(f"patch {i}: vertices {data[this_grouped_vertidx].shape[0]} -> {n_eigenmodes} leadfield reduction explained {percentage_explained*100:.3f}% variance")
        group_eigenmodes[i * n_eigenmodes:(i + 1) * n_eigenmodes] = this_group_eigenmodes
        if i < n_groups[0]:
            lhweights.append([eig_src_weights, this_grouped_vertidx_no_offset])
        else:
            rhweights.append([eig_src_weights, this_grouped_vertidx_no_offset])

    weights = [lhweights, rhweights]
    src_flips = [None] * sum(n_groups)
    return weights, group_eigenmodes.T, grouped_vertidx, src_flips


def _prepare_label_extraction(labels, src):
    vertno = [s['vertno'] for s in src]
    label_vertidx = []
    for label in labels:
        if label.hemi == 'lh':
            this_vertices = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertices)
        elif label.hemi == 'rh':
            this_vertices = np.intersect1d(vertno[1], label.vertices)
            vertidx = len(vertno[0]) + np.searchsorted(vertno[1], this_vertices)
        if len(vertidx) == 0:
            vertidx = None
        label_vertidx.append(vertidx)
    return label_vertidx


def assign_labels(labels, src_target, src_origin, thresh=0):
    """Assign the patch indices of the corresponding labels from origin into target source space

    This function returns the patch indices of the (ROI) labels in the target source space (e.g. 'ico-1') from the
    origin source space (e.g. 'ico-4')

    Parameters
    ----------
    labels:  mne.Labels | mne.Label
        labels in standard MNE-python format
    src_target: mne.SourceSpaces
        target source space, e.g. ico-4
    src_origin: mne.SourceSpaces
        origin source space, e.g. ico-4

    Returns
    -------
    label_vertidx: list
        vertex(patch) index
    """
    label_vertidx_origin = _prepare_label_extraction(labels, src_origin)
    _, group_vertidx, _, _ = _prepare_leadfield_reduction(src_target, src_origin)
    label_vertidx = []
    for this_label_vertidx_origin in label_vertidx_origin:
        this_label_vertidx = []
        for i, this_group_vertidx in enumerate(group_vertidx):
            this_vertices = np.intersect1d(this_group_vertidx, this_label_vertidx_origin)
            if len(this_vertices) > thresh:
                this_label_vertidx.append(i)
        this_label_vertidx = np.asanyarray(this_label_vertidx)
        label_vertidx.append(this_label_vertidx)
    return label_vertidx


def _prepare_leadfield_reduction(src_target, src_origin):
    vertno_origin = [s['vertno'] for s in src_origin]
    vertno_target = [s['vertno'] for s in src_target]
    pinfo_target = [s['pinfo'] for s in src_target]
    n_verts = [s['nuse'] for s in src_origin]
    n_groups = [s['nuse'] for s in src_target]
    grouped_vertidx = []
    grouped_vertidx_no_offset = []
    
    for k, (this_vertno_target, this_pinfo_target, this_vertno_origin) in enumerate(zip(vertno_target, pinfo_target,
                                                                                        vertno_origin)):
        offset = 0 if k == 0 else n_verts[k - 1]
        for this_vert, this_pinfo in zip(this_vertno_target, this_pinfo_target):
            this_vertices = np.intersect1d(this_vertno_origin, this_pinfo)
            vertidx = offset + np.searchsorted(this_vertno_origin, this_vertices)
            
            # offset ensures that rh indices are sequential with the lh indices, but for indexing
            # into the rh sources spaces object, we don't want this offset since the indices 
            # overlap with the lh source spaces indices. just create another list for this
            vertidx_no_offset = np.searchsorted(this_vertno_origin, this_vertices)
            if len(vertidx) == 0:
                vertidx = None
                vertidx_no_offset = None
            grouped_vertidx.append(vertidx)
            grouped_vertidx_no_offset.append(vertidx_no_offset)
            
    return grouped_vertidx_no_offset, grouped_vertidx, n_groups, n_verts


def _extract_label_eigenmodes(fwd, labels, data=None, mode='mean', n_eigenmodes=2, allow_empty=False,
        trans=None, mri_resolution=True, ):
    "Zero columns corresponds to empty labels"
    src = fwd['src']
    _validate_type(src, SourceSpaces)
    _check_option('mode', mode, ['svd', 'svd_flip'] + ['auto'])
    func = _svd_funcs[mode]

    if len(src) > 2:
        if src[0]['type'] != 'surf' or src[1]['type'] != 'surf':
            raise ValueError('The first 2 source spaces have to be surf type')
        if any(np.any(s['type'] != 'vol') for s in src[2:]):
            raise ValueError('source spaces have to be of vol type')

        n_aparc = len(labels)
        n_aseg = len(src[2:])
        n_labels = n_aparc + n_aseg
    else:
        n_labels = len(labels)

    # create a dummy stc
    kind = src.kind
    vertno = [s['vertno'] for s in src]
    nvert = np.array([len(v) for v in vertno])
    if kind == 'surface':
        stc = SourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    elif kind == 'mixed':
        stc = MixedSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    else:
        stc = VolSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    stcs = [stc]

    vertno = None
    for si, stc in enumerate(stcs):
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = \
                _prepare_label_extraction(stc, labels, src, mode.replace('svd', 'mean'),
                                          allow_empty)
        if isinstance(stc, (_BaseVolSourceEstimate,
                            _BaseVectorSourceEstimate)):
            _check_option(
                'mode', mode, ('svd',),
                'when using a volume or mixed source space')
            mode = 'svd' if mode == 'auto' else mode
        else:
            mode = 'svd_flip' if mode == 'auto' else mode

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        if data is None:
            logger.info('Using the raw forward solution')
            data = np.swapaxes(fwd['sol']['data'], 0, 1)  # (n_sources, n_channels)
        data = data.copy()

        # do the extraction
        label_eigenmodes = np.zeros((n_labels * n_eigenmodes,) + data.shape[1:], dtype=data.dtype)
        for i, (vertidx, flip, label) in enumerate(zip(label_vertidx, src_flip, labels)):
            if vertidx is not None:
                if isinstance(vertidx, sparse.csr_matrix):
                    assert mri_resolution
                    assert vertidx.shape[1] == data.shape[0]
                    this_data = np.reshape(data, (data.shape[0], -1))
                    this_data = vertidx * this_data
                    this_data.shape = \
                        (this_data.shape[0],) + stc.data.shape[1:]
                else:
                    this_data = data[vertidx]
                label_eigenmodes[i * n_eigenmodes:(i + 1) * n_eigenmodes] = \
                    func(flip, this_data, n_eigenmodes)

        return label_eigenmodes.T, label_vertidx, src_flip


def _expand_roi_indices_as_tup(reg_idx, emod):
    return tuple(range(reg_idx * emod, reg_idx * emod + emod))


def _truncatedsvd(a, n_components=2, return_pecentage_exaplained=False):
    if n_components > min(*a.shape):
        raise ValueError('n_components={:d} should be smaller than '
                         'min({:d}, {:d})'.format(n_components, *a.shape))
    u, s, vh = linalg.svd(a, full_matrices=False, compute_uv=True,
                          overwrite_a=True, check_finite=True,
                          lapack_driver='gesdd')
    if return_pecentage_exaplained:
        return u, vh[:n_components] * s[:n_components][:, None], s[:n_components].sum() / s.sum()
    return u, vh[:n_components] * s[:n_components][:, None]


_svd_funcs = {
    'svd_flip': lambda flip, data, n_components: _truncatedsvd(flip * data, n_components),
    'svd': lambda flip, data, n_components: _truncatedsvd(data, n_components)
}


# Note for covariance, source_weighting needs to be applied twice!
def _reapply_source_weighting(X, source_weighting):
    X *= source_weighting[:, None]
    return X