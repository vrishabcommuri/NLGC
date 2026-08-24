import numpy as np
from mne.forward import is_fixed_orient
from mne.io.constants import FIFF
from mne.minimum_norm.inverse import _check_reference
from scipy import linalg
from nlgc.utils.leadfield import prepare_eigenmodes
from nlgc.nlgc_utils import gc_extraction, NLGC
from nlgc.config import ModelConfig
from nlgc.utils.initialize import initialize_em_state
import time
from nlgc.utils.param_vis import generate_report

def nlgc_map(name, evoked, forward, noise_cov, src_target, patch_idx, 
             config=None, save_dir = None, **kwargs):
    """NLGC connectivity map estimation

    This function estimates the causal connectivity map across sources given the
    MEG measurements, forward model, measurement noise covariance matrix, and a
    few model-related parameters.

    Parameters
    ----------
    name: str
        subject's name
    evoked: mne.Evoked
        MEG evoked response in MNE-python standard format
    forward: mne.Forward
        forward solution in MNE-python standard format
    noise_cov: mne.Covariance
        measurement noise covariance matrix (obtained from empty room
        or base-line recordings)
    src_target: mne.SourceSpaces used for leadfield summarization (e.g., ico1)
    order: int
        VAR model order
    self_history: int | None
        number of removed self-history lags in VAR model to mitigate possible
        overfitting (a[:self_history,i,i]=0) (default = None)
    n_eigenmodes: int
        number of eignemodes
    alpha: int | float
        Inv-Gamma(alpha*t/2 - 1, beta*t) prior on the state noise covariance
        matrix
    beta: int | float
        Inv-Gamma(alpha*t/2 - 1, beta*t) prior on the state noise covariance
        matrix
    patch_idx: list | None
        subset of patch indices to find the connectivity within them (None =
        whole source space)
    n_segments: int
        number of segments which divides the MEG recording into equal parts for
        non-centrality parameter estimation
    {loose, depth, pca, rank}: float/boolean
        forward model computation parameters, check
        mne.inverse_sparse.mxne_inverse for more info
    lambda_range: numpy 1d array
        an array of the regularization coefficients for cross-validation
    max_iter: int
        maximum number of iterations for EM-based parameter estimation
    max_cyclic_iter: int
        maximum number of cyclic iterations to update VAR coefficients (A's) and
        covariance (q's)
    tol: float
        tolerance for EM convergence (in terms of relative jump of
        log-likelihood function)
    sparsity_factor: float
        the threshold to remove reduced models with sufficiently small VAR
        coefficients in their corresponding full models for speeding up the
        calculations (None = all possible reduced models)
    cv: int
        number of folds used for cross-validation
    use_es: boolean
        if True, uses estimation stability for CV metric, otherwise it uses
        log-likelihood value; check this for
            more info: https://doi.org/10.1080/10618600.2015.1020159 (ESVC)
    var_thr: float
        the threshold to limit the number of reduced models by considering only
        the possible links between the active sources which explain 'var_thr' of
        the total power (default = 1, i.e., all sources)
    cv_type: str
        the type of cv either using seeded lambda or 
    

    Returns
    -------
    nlgc_obj : NLGC object
        contains the connectivity map and the some related parameters (see NLGC
        class for more info)
    """


    start_time = time.time()

    if config is None:
        config = ModelConfig.from_legacy_kwargs(kwargs)


    if any(l <= 0 for l in config.sparsity.lambda_range):
        raise ValueError('only positive lambdas are allowed, got '
                         f'{config.sparsity.lambda_range}')

    _check_reference(evoked)

    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    n_segments = config.latent.n_segments

    # free-orientation forwards are the volume/mixed source space case, which
    # prepare_eigenmodes supports via _reduce_lead_field_vol -- but only when
    # the config actually asks for multiple orientations
    # if not is_fixed_orient(forward) and n_orients <= 1:
    #     raise ValueError(f"Can't work with free orientation forward: {forward} "
    #                      f"unless config.latent.n_orients > 1")
                         
    if src_target[0]["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError("Can't work non-head source space orientation. "
                         "Try applying trans to source space to put it in "
                         "head coordinates")
        

    # keyword args: the positional form put the whole ModelConfig into
    # prepare_eigenmodes' `n_eigenmodes: int` slot
    weights, G, label_vertidx, label_names, gain_info, whitener = \
            prepare_eigenmodes(evoked.info, forward, noise_cov, src_target,
                               n_eigenmodes=n_eigenmodes,
                               n_orients=n_orients,
                               loose=config.forward.loose,
                               depth=config.forward.depth,
                               pca=config.forward.pca,
                               rank=config.forward.rank)

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    M = np.dot(whitener, M)
    r = 1.0

    if len(patch_idx) == 0:
        raise ValueError("Length of patch_idx should not be zero")

    n, nnx = G.shape
    # each ROI contributes n_eigenmodes * n_orients columns
    nx = nnx // (n_eigenmodes * n_orients)
    _, t = M.shape
    tt = t // n_segments

    d_raw = np.zeros((n_segments, nx, nx))
    bias_r = np.zeros((n_segments, nx, nx))
    bias_f = np.zeros((n_segments, 1))
    conv_flag = np.zeros((n_segments, nx, nx))

    models = []
    for this_segment in range(0, n_segments):
        if config.numerical.verbose:
            print('Segment: ', this_segment + 1)
            print(f"nlgc_map max iter = {config.optimizer.max_iter}")

        y = M[:, this_segment * tt: (this_segment + 1) * tt]
        F = G

        F_companion, R_companion, em_state = initialize_em_state(
            y=y, F=F, r=r, config=config, evoked=evoked, forward=forward,
            noise_cov=noise_cov, weights=weights)

        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            gc_extraction(y.T, F_companion, R_companion, ROIs=patch_idx,
                          em_state=em_state, config=config)
        
        d_raw[this_segment] = d_raw_
        bias_r[this_segment] = bias_r_
        bias_f[this_segment] = bias_f_
        models.append(model_f)
        conv_flag[this_segment] = conv_flag_

    # TODO !!!SHOULD NOT return nlgc object because a common workflow is to
    # pickle the result and pickling a class is sensitive to internal class
    # changes and will only be unpickleable with the exact version of nlgc used
    # to produce it installed
    nlgc_obj = NLGC(name, nx, n, t, 
                    config.latent.order, 
                    config.latent.n_eigenmodes, 
                    config.latent.n_orients, 
                    config.latent.n_segments, 
                    d_raw, bias_f, bias_r, models,
                    conv_flag, label_names, label_vertidx, 
                    forward, whitener, weights)
    
    stop_time = time.time()

    total_time = stop_time - start_time

    nlgc_param_dict = {
        name: name,
        'n_eigenmodes': config.latent.n_eigenmodes,
        'n_orients': config.latent.n_orients,
        'order': config.latent.order,
        'lambda_range': config.sparsity.lambda_range,
        'best_lambda': nlgc_obj._model_f[0].lambda_,
        'use_es': config.validation.use_es,
        'nlgc_map_time': total_time
    }
    if save_dir is not None:
        generate_report(save_dir = save_dir, model = nlgc_obj, 
                        param_dict = nlgc_param_dict)

    return nlgc_obj