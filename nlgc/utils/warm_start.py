from mne.minimum_norm import apply_inverse, make_inverse_operator
from nlgc.utils.transforms import surface_ico4_to_surface_eigs
import numpy as np


def _triage_warm_start(evoked, forward, noise_cov, weights):
    assert evoked is not None
    assert forward is not None
    assert noise_cov is not None
    assert weights is not None


def warm_start_sources(evoked, forward, noise_cov, weights, config):
    _triage_warm_start(evoked, forward, noise_cov, weights)

    inv = make_inverse_operator(evoked.info, forward, noise_cov, 
                                loose=config.forward.loose, 
                                depth=config.forward.depth, 
                                rank=config.forward.rank, 
                                fixed=True, 
                                verbose=config.numerical.verbose)
    
    inv_stc = apply_inverse(evoked, inv)

    neig = config.latent.n_eigenmodes
    order = config.latent.order

    # sources are stacked in the VAR(1) representation, so has dim (n_times,
    # n_sources * n_lags)
    latent_state = np.zeros((inv_stc.data.shape[1], neig * 84 * order))
    
    # (n_time, n_sources)
    stc_init = surface_ico4_to_surface_eigs(inv_stc, weights, neig) 

    # roll initialization data to represent each lag
    for lagidx in range(order):
        latent_state[:, lagidx * neig * 84: (lagidx+1) * neig * 84] = \
                        np.ascontiguousarray(np.roll(stc_init, -lagidx, axis=0))
    
    # TODO: stc initialization this way includes autocorrelation structure in x
    # which may influence the connections derived in A under sparsity
    # constraints. one solution to this is to fit a VAR(1) or VAR(p) model using
    # least-squares to estimate the first- or pth-order autocorrelation in the
    # data and then use the residual from the model as the initialization.
    return latent_state