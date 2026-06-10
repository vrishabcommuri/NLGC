
from multiprocessing import cpu_count, current_process

import numpy as np
import pickle

from functools import reduce
from matplotlib import pyplot as plt
from mne.forward import is_fixed_orient
from mne.minimum_norm.inverse import _check_reference
from mne.minimum_norm import apply_inverse, make_inverse_operator, InverseOperator
import mne.minimum_norm
from scipy import linalg
from .opt import *
from ._stat import fdr_control
from ._bias_utils import debias_deviances
from ._gen_utils import LazyProperty
from ._nlgc_utils import _gc_extraction, _prepare_eigenmodes, NLGC, surface_ico4_to_surface_eigs
from ._nlgc_test_utils import run_GT_sim



def nlgc_map(name, evoked, forward, noise_cov, labels, order, self_history=None, n_eigenmodes=2, alpha=0.0, beta=0.0,
        patch_idx=[], n_segments=1, loose=0.0, depth=0.0, pca=True, rank=None, lambda_range=None, lambda1=None, lambda2=None,
        max_iter=500, max_cyclic_iter=3, tol=1e-5, sparsity_factor=0.0, cv=5, use_lapack=True, use_es=True, var_thr=1.0, cv_type = 'seeded', verbose=False, warm_start = False, n_orients = 1):
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
    cv_type: str
        the type of cv either using seeded lambda or 
    

    Returns
    -------
    nlgc_obj : NLGC object
        contains the connectivity map and the some related parameters (see NLGC class for more info)
    """

    _check_reference(evoked)

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

        
    weights, G, label_vertidx, label_names, gain_info, whitener = \
        _prepare_eigenmodes(evoked.info, forward, noise_cov, labels, n_eigenmodes, n_orients, loose, depth, pca, rank)


    stc_init = None
    if warm_start:
        inv = make_inverse_operator(evoked.info, forward, noise_cov, loose=loose, depth=depth, rank=rank, fixed=True, verbose=verbose)
        inv_stc = apply_inverse(evoked, inv)

        # sources are stacked in the VAR(1) representation, so has dim (n_times, n_sources * n_lags)
        _x = np.zeros((inv_stc.data.shape[1], n_eigenmodes * 84 * order))
        stc_init = surface_ico4_to_surface_eigs(inv_stc, weights, n_eigenmodes) # (n_samples, n_sources)

        # roll initialization data to represent each lag
        for _p in range(order):
            _x[:, _p * n_eigenmodes * 84: (_p+1) * n_eigenmodes * 84] = np.ascontiguousarray(np.roll(stc_init, -_p, axis=0))
        
        # TODO: stc initialization this way includes autocorrelation structure in x
        # which may influence the connections derived in A under sparsity constraints.
        # one solution to this is to fit a VAR(1) or VAR(p) model using least-squares 
        # to estimate the first- or pth-order autocorrelation in the data and then 
        # use the residual from the model as the initialization.
        stc_init = (_x, np.ascontiguousarray(np.roll(_x, -1, axis=0)))  # _x, x_

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


    # run_GT_sim(lead_field_gen = False, lf = G, seed = 0, band = "wide", fs = 50, natures = 'all', 
    #     root = None, subject_id = None, session_name = None, trans = None, order = order, t = 500, n_eigenmodes = n_eigenmodes,
    #     n_segments = 1, loose = loose, depth = depth, pca = pca, rank = rank, lambda_range = lambda_range,
    #     max_iter = max_iter, max_cyclic_iter = max_cyclic_iter, tol = tol, sparsity_factor = sparsity_factor, cv = cv ,var_thr = var_thr, alpha = alpha)
    

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
                           n_orients = n_orients,
                           ROIs=patch_idx,
                           alpha=alpha, beta=beta, cv=cv, lambda_range=lambda_range, lambda1=lambda1, 
                           lambda2=lambda2, max_iter=max_iter,
                           max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor,
                           use_lapack=use_lapack, use_es=use_es, var_thr=var_thr, xs_init=stc_init, verbose=verbose)
        d_raw[this_segment] = d_raw_
        bias_r[this_segment] = bias_r_
        bias_f[this_segment] = bias_f_
        models.append(model_f)
        conv_flag[this_segment] = conv_flag_

    nlgc_obj = NLGC(name, nx, n, t, order, n_eigenmodes, n_orients, n_segments, d_raw, bias_f, bias_r, models,
                    conv_flag, label_names, label_vertidx, forward, whitener, weights)

    return nlgc_obj