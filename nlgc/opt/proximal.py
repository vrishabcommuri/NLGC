from jaxopt import ProximalGradient
import jax
import jax.numpy as jnp         
from functools import partial   
import dataclasses          
jax.config.update("jax_enable_x64", True)    

solver = None
solve_for_Q = None
# what the current `solver`/`solve_for_Q` were built for, so repeated calls with
# identical settings (e.g. every task in a reused multiprocessing worker) can
# no-op instead of rebuilding the jitted closures and busting their cache
_solver_signature = None


def instantiate_proximal_solvers(config, N_sources, force=False):
    global solver, solve_for_Q, _solver_signature
    if hasattr(config, 'latent'):
        p = config.latent.order
        m = N_sources
        n_orients = config.latent.n_orients
        alpha = config.sparsity.alpha
        beta = config.sparsity.beta
    else:
        # testing config; don't need to instantiate entire config class 
        assert isinstance(config, dict) 
        p = config['order']
        m = N_sources
        n_orients = config['n_orients']
        alpha = config['alpha']
        beta = config['beta']

    signature = (p, m, n_orients, alpha, beta)
    if not force and solver is not None and _solver_signature == signature:
        return solver

    prox = partial(proxg_vec, p = p, m = m, n_orients = n_orients)

    solve_for_Q = jax.jit(
        partial(_solve_for_Q,
            alpha=alpha,
            beta=beta,
            n_orients=n_orients,
        )
    )

    solver = ProximalGradient(fun = f, prox = prox)
    _solver_signature = signature
    return solver


def f(x, s1, s2, Qinv):
    U = Qinv @ x
    return (-2 * jnp.sum(U * s1) + jnp.sum(U * (x @ s2)))


def proxg_vec(x, lam, t, p, m, n_orients = 3):
    N = m // n_orients
    thresh = t * lam

    B = x.reshape(N, n_orients, p, N, n_orients)

    norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4), keepdims=True))

    scale = 1.0 - thresh / jnp.maximum(norms, 1e-12)
    scale = jnp.maximum(scale, 0.0)

    B_out = B * scale

    return B_out.reshape(x.shape)
    

@jax.jit
def proximal_param_update(em_state, smoother_result, lambda_):
    s1, s2, s3, n = calculate_ss_jax(em_state, smoother_result)
    
    m = em_state.N_sources_upper

    Qinv = jax.scipy.linalg.solve(
        em_state.Q[:m, :m],
        jnp.eye(m)
    )

    A_shrunk = solver.run(em_state.A[:m], 
                          hyperparams_prox=lambda_, s1=s1, 
                          s2=s2, Qinv=Qinv).params

    em_state = dataclasses.replace(em_state,
                                   A = em_state.A.at[:m].set(A_shrunk))
    Q_new = solve_for_Q(em_state.A[:m], s1, s2, s3)

    em_state = dataclasses.replace(em_state,
                                   Q = em_state.Q.at[:m, :m].set(Q_new))


    return em_state


def _solve_for_Q(A, s1, s2, s3, alpha, beta, n_orients):
    sigma = s3 - A @ s1.T - s1 @ A.T + A @ s2 @ A.T
    sigma = 0.5 * (sigma + sigma.T)

    m = sigma.shape[0]
    n_blocks = m // n_orients

    idx = jnp.arange(n_blocks)

    # (n_blocks, n_orients, n_blocks, n_orients)
    S = sigma.reshape(n_blocks, n_orients,
                      n_blocks, n_orients)

    # Extract block diagonal
    blocks = S[idx, :, idx, :]
    blocks = blocks + beta * jnp.eye(n_orients, dtype=sigma.dtype)

    # Construct block-diagonal Q
    Q_new = jnp.zeros_like(sigma)
    Q_new = Q_new.reshape(n_blocks, n_orients,
                      n_blocks, n_orients)
    Q_new = Q_new.at[idx, :, idx, :].set(blocks)
    Q_new = Q_new.reshape(m, m)

    Q_new = Q_new / (1.0 + alpha)

    return Q_new


def calculate_ss_jax(em_state, smoother_result):
    """Calculates the required second order expectations"""

    m = em_state.N_sources_upper
    x_bar = smoother_result.smoothed_state
    s_bar = smoother_result.smoothed_cov
    b = smoother_result.smoother_gain

    p = b.shape[1] // m
    n = x_bar.shape[0] - p

    s_cross = b @ s_bar[:, :m]

    x_ = x_bar[:, :m]

    s1 = (x_[p:].T @ x_bar[p-1:-1]) / n + s_cross.T
    s2 = (x_bar[p-1:-1].T @ x_bar[p-1:-1]) / n + s_bar
    
    s3 = (x_[p:].T @ x_[p:]) / n + s_bar[:m, :m]

    return s1, s2, s3, n


def calculate_ss(x_bar, s_bar, b, m, p):
    """Calculates the required second order expectations

    Parameters
    ----------
    x_bar : ndarray of shape (n_samples, n_sources*order)
        smoothed means
    s_bar : ndarray of shape (n_sources*order, n_sources*order)
        smoothed covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoother gain
    m : int
        n_sources
    p : int
        order
    Returns
    -------
    s1 : ndarray of shape (n_sources, n_sources*order)
        n, n-1
    s2 : ndarray of shape (n_sources*order, n_sources*order)
        n-1, n-1 (augmented)
    s3 : ndarray of shape (n_sources, n_sources)
        n, n

    Notes
    -----
    the scaling by 1/n normalizes the Q function by time samples.
    """

    s_cross = b.dot(s_bar[:, :m])
    x_ = x_bar[:, :m]
    n = (x_bar.shape[0] - p)

    # compute the following quantities carefully
    # s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s1 = x_[p:].T.dot(x_bar[p - 1:-1]) / n + s_cross.T

    # s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s2 = x_bar[p - 1:-1].T.dot(x_bar[p - 1:-1]) / n + s_bar
    if (jnp.diag(s2) <= 0).any():
        raise ValueError('diag(s2) values are not non-negative!')
    # s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)
    s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[(p - 1) * m:, (p - 1) * m:]
    # s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[:m, :m]

    return s1, s2, s3, n


