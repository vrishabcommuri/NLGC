import jax
import jax.numpy as jnp
import dataclasses
from nlgc.opt.fastac import Fasta
from functools import partial
jax.config.update("jax_enable_x64", True)    


@partial(jax.jit, static_argnames=("config",))
def proximal_param_update(em_state, smoother_result, config, lambda_):
    s1, s2, s3, n = calculate_ss_jax(em_state, smoother_result)
    
    n_orients = config.latent.n_orients
    m = em_state.N_sources_upper
    max_fasta_iter = config.optimizer.max_fasta_iter
    p = config.latent.order

    A_prev = em_state.A[:m]
    Q_upper = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]

    
    dtype = jnp.result_type(A_prev)
    out_type = jax.ShapeDtypeStruct((m, m*p), dtype)

    A_shrunk = jax.pure_callback(solve_for_a, 
                    out_type,
                    Q_upper,
                    s1,
                    s2,
                    A_prev,
                    A_mask,
                    lambda_,
                    n_orients=n_orients,
                    max_iter=max_fasta_iter,
                    tol=1e-4,
                    verbose=config.numerical.verbose,
                    vmap_method='sequential')

    em_state = dataclasses.replace(em_state,
                                   A = em_state.A.at[:m].set(A_shrunk))
    
    Q_new = solve_for_Q(em_state.A[:m], s1, s2, s3,
                        config.sparsity.alpha,
                        config.sparsity.beta,
                        n_orients)

    em_state = dataclasses.replace(em_state,
                                   Q = em_state.Q.at[:m, :m].set(Q_new))
    
    rel_A_change = relative_A_change_jax(A_shrunk, A_prev)

    return em_state, rel_A_change


def solve_for_a(Q, s1, s2, A, A_mask, lambda2, n_orients=3, max_iter=5000, 
                tol=1e-5, verbose=0):
    """
    solve for A using group sparse proximal gradient.
    """
    if lambda2 == 0:
        return jnp.linalg.solve(s2, s1.T).T

    m = A.shape[0]
    p = A.shape[1] // m

    # ------------------------------------------------------------
    # feature standardization and target whitening
    # ------------------------------------------------------------

    # precondition the problem to transform it into a standard least-squares
    # space. this speeds up convergence and ensures the group lasso penalty
    # treats all variances equally.

    # standardize s2 by its diagonal
    d = jnp.sqrt(jnp.diag(s2))
    d_safe = jnp.maximum(d, 1e-12)
    s2_tilde = s2 / jnp.outer(d_safe, d_safe)
    s1_tilde = s1 / d_safe[None, :]

    # whiten targets by Q^{-1/2} 
    # (using eigh since Q is symmetric positive definite)
    evals, evecs = jnp.linalg.eigh(Q)
    evals_safe = jnp.maximum(evals, 1e-12)
    
    q_inv_sqrt = evecs @ jnp.diag(1.0 / jnp.sqrt(evals_safe)) @ evecs.T
    q_sqrt = evecs @ jnp.diag(jnp.sqrt(evals_safe)) @ evecs.T

    # apply to s1 and initial A
    s1_tilde = q_inv_sqrt @ s1_tilde
    A_tilde = (q_inv_sqrt @ A) * d_safe[None, :]

    # ------------------------------------------------------------
    # objective and related funcs
    # ------------------------------------------------------------

    def f_fun(x):
        xs2 = x @ s2_tilde
        return (jnp.trace(xs2 @ x.T) - 2.0 * jnp.trace(s1_tilde @ x.T))

    def grad_fun(x):
        return 2.0 * (x @ s2_tilde - s1_tilde)

    def g_fun(x):
        n_sources = m // n_orients

        B = x.reshape(n_sources, n_orients, p, n_sources, n_orients)

        norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4), keepdims=True))

        return lambda2 * jnp.sum(norms)

    def prox_fun(x, t):
        n_sources = m // n_orients

        B = x.reshape(n_sources, n_orients, p, n_sources, n_orients)

        norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4), keepdims=True))

        scale = jnp.maximum(1.0 - lambda2 * t / jnp.maximum(norms, 1e-12), 0.0)

        B = B * scale
        x_new = B.reshape(x.shape)

        # enforce link testing constraints
        x_new = x_new * A_mask

        return x_new

    # ------------------------------------------------------------
    # FASTA
    # ------------------------------------------------------------

    fasta = Fasta(
        f_fun,
        g_fun,
        grad_fun,
        prox_fun,
        beta=0.5,
        n_iter=max_iter,
        verbose=verbose,
    )
    
    # we train on the preconditioned matrix
    fasta.learn(A_tilde, tol)

    A_tilde_new = fasta.coefs_

    # ------------------------------------------------------------
    # reverse preconditioning
    # ------------------------------------------------------------
    
    A_new = (q_sqrt @ A_tilde_new) / d_safe[None, :]

    return A_new


def solve_for_Q(A, s1, s2, s3, alpha, beta, n_orients):
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

    s1 = x_[p:].T.dot(x_bar[p - 1:-1]) / n + s_cross.T

    s2 = x_bar[p - 1:-1].T.dot(x_bar[p - 1:-1]) / n + s_bar
    if (jnp.diag(s2) <= 0).any():
        raise ValueError('diag(s2) values are not non-negative!')

    s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[:m, :m]

    return s1, s2, s3, n


def relative_A_change_jax(curr_A, prev_A, eps=1e-12):
    delta = curr_A - prev_A

    return (
        jnp.linalg.norm(delta)
        / jnp.maximum(jnp.linalg.norm(prev_A), eps)
    )
