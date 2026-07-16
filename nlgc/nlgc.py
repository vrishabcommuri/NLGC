import numpy as np
from mne.forward import is_fixed_orient
from mne.minimum_norm.inverse import _check_reference
from scipy import linalg
from nlgc.utils.leadfield import prepare_eigenmodes
from nlgc.utils.warm_start import warm_start_sources
from nlgc.nlgc_utils import gc_extraction, NLGC
from nlgc.config import ModelConfig
from nlgc.opt.em import EMState


def nlgc_map(name, evoked, forward, noise_cov, labels, patch_idx, 
             config=None, **kwargs):
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
        measurement noise covariance matrix (could be obtained from empty room
        or base-line recordings)
    labels: mne.SourceSpaces | mne.Forward | mne.Labels
        source space, forward solution, or list of labels, all in MNE-python
        standard format
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

    if config is None:
        config = ModelConfig.from_legacy_kwargs(kwargs)
        
    _check_reference(evoked)

    if not is_fixed_orient(forward):
        raise ValueError(f"Can't work with free orientation forward: {forward}")

        
    weights, G, label_vertidx, label_names, gain_info, whitener = \
            prepare_eigenmodes(evoked.info, forward, noise_cov, labels, config)

    
    em_state = EMState()
    if config.optimizer.warm_start:
        em_state.smoothed_state = warm_start_sources(evoked, forward, noise_cov, 
                                                     weights, config)
        
        # !!! TODO the smoothed state will just be overwritten after the first
        # iter since the kf marginal likelihood p(y|theta) doesn't depend on x
        # (x is not a parameter!). we need to treat the "smoothed state" above
        # as oracle and then fit a simple VAR model to it to obtain warm-start
        # parameter estimates for A and Q; those can then be loaded into
        # em_state above. this can be done with pymc VAR and find_map.
    
    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    M = np.dot(whitener, M)

    # Normalization
    M_normalizing_factor = linalg.norm(np.dot(M, M.T) / M.shape[1], ord='fro')
    G_normalizing_factor = np.sqrt(np.sum(G ** 2, axis=0))
    G /= G_normalizing_factor
    M /= np.sqrt(M_normalizing_factor)
    r = 1 / M_normalizing_factor

    if len(patch_idx) == 0:
        raise ValueError("Length of patch_idx should not be zero")

    n_eigenmodes = config.latent.n_eigenmodes
    n_segments = config.latent.n_segments
    
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
        if config.numerical.verbose:
            print('Segment: ', this_segment + 1)
            print(f"nlgc_map max iter = {config.optimizer.max_iter}")
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            gc_extraction(M[:, this_segment * tt: (this_segment + 1) * tt], G, 
                          r, ROIs=patch_idx, em_state=em_state, config=config)
        
        d_raw[this_segment] = d_raw_
        bias_r[this_segment] = bias_r_
        bias_f[this_segment] = bias_f_
        models.append(model_f)
        conv_flag[this_segment] = conv_flag_

    nlgc_obj = NLGC(name, nx, n, t, 
                    config.latent.order, 
                    config.latent.n_eigenmodes, 
                    config.latent.n_orients, 
                    config.latent.n_segments, 
                    d_raw, bias_f, bias_r, models,
                    conv_flag, label_names, label_vertidx, 
                    forward, whitener, weights)

    return nlgc_obj