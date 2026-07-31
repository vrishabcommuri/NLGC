from nlgc.utils.warm_start import warm_start_sources
from nlgc.opt.em import EMState
from scipy import linalg, optimize
import numpy as np


def initialize_em_state(y, F, r, config, evoked=None, forward=None, noise_cov=None, 
                        weights=None):
    F_companion, R_companion, em_state = companion_init(y, F, r, config)
    # em_iter spans BOTH phases -- solve_params runs em_blas for n_warmup_iter
    # then em_jax for max_iter -- so the trajectory needs room for their sum.
    # solve_params re-allocates this to the same size on every fit; keeping it
    # here means the state is valid (_triage_em_state asserts non-None) for
    # callers that use it before fitting.
    em_state.log_likelihood = np.zeros(config.optimizer.n_warmup_iter +
                                       config.optimizer.max_iter + 1)

    if config.optimizer.warm_start:
        em_state.smoothed_state = warm_start_sources(evoked, forward, noise_cov, 
                                                     weights, config)
        
        # !!! TODO the smoothed state will just be overwritten after the first
        # iter since the kf marginal likelihood p(y|theta) doesn't depend on x
        # (x is not a parameter!). we need to treat the "smoothed state" above
        # as oracle and then fit a simple VAR model to it to obtain warm-start
        # parameter estimates for A and Q; those can then be loaded into
        # em_state above. this can be done with pymc VAR and find_map.

    return F_companion, R_companion, em_state


def companion_init(y, F, r, config):
    total_sensor_dim, total_latent_dim = F.shape
    zero_companion = np.zeros((total_latent_dim * config.latent.order,
                               total_latent_dim * config.latent.order))

    m = total_latent_dim
    p = config.latent.order

    A = np.block([[np.zeros_like(zero_companion[:m])],
                  [np.eye(N = m*(p-1), M = m*p)]])

    Q = np.zeros_like(zero_companion)
    Q[:m,:m] = data_driven_Q_init(y, F)

    F = np.hstack([F, np.zeros((total_sensor_dim, m*(p-1)))])
    R = r * np.eye(total_sensor_dim)

    em_state = EMState(
        A = A,
        A_mask = np.ones_like(zero_companion),
        Q = Q, 
        P0 = np.zeros_like(zero_companion),
        N0 = np.zeros_like(zero_companion),
        N_sources_upper = total_latent_dim,
    )

    return F, R, em_state


def data_driven_Q_init(y, F):
    n, m = F.shape
    e, U = linalg.eigh(F @ F.T)
    source_proj = U.T @ y
    est_source_pow = (source_proj ** 2).sum(axis=1)

    def fun(x):
        return (est_source_pow / (1 + x * e) ** 2).sum() - 1.2 * n * y.shape[1]

    if fun(0) > 0:
        q_val = optimize.newton(fun, 1)
    else:
        q_val = 0.0001

    return q_val * np.eye(m)