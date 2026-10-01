import jax
import jax.numpy as jnp
import dataclasses
from functools import partial
import time
from nlgc.opt.fastac import Fasta
jax.config.update("jax_enable_x64", True)    


@partial(jax.jit, static_argnames=("config",))
def proximal_param_update(em_state, smoother_result, config, lambda_):
    s1, s2, s3, n = calculate_ss_jax(em_state, smoother_result)
    
    n_orients = config.latent.n_orients
    m = em_state.N_sources_upper
    max_fasta_iter = config.optimizer.max_fasta_iter
    p = config.latent.order
    lagsparsity = config.sparsity.lagsparsity
    fasta_tol = config.optimizer.fasta_tol

    A_prev = em_state.A[:m]
    Q_upper = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    q_val = em_state.q_val
    
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
                    lagsparsity=lagsparsity,
                    tol=fasta_tol,
                    verbose=config.numerical.verbose,
                    vmap_method='sequential')

    em_state = dataclasses.replace(em_state,
                                   A = em_state.A.at[:m].set(A_shrunk))
    
    Q_new = solve_for_Q(em_state.A[:m], s1, s2, s3, n, n_orients, 
                        nu0 = config.qprior.nu0,                 # set auto
                        q_base = q_val,                          # set auto
                        singular_values = em_state.Q_prior_scales, 
                        source_mass = None,                      # set auto
                        sigma_gamma = config.qprior.sigma_gamma, 
                        sigma_min = config.qprior.sigma_min, 
                        sigma_max = config.qprior.sigma_max, 
                        eig_floor = config.qprior.eig_floor)      

    em_state = dataclasses.replace(em_state,
                                   Q = em_state.Q.at[:m, :m].set(Q_new))
    

    obj, _, _, _ = penalized_q_objective(A_shrunk, Q_new, s1, s2, s3, lambda_, 
                                         n_orients, lagsparsity)
    
    rel_A_change = relative_A_change_jax(A_shrunk, A_prev)

    return em_state, rel_A_change, obj


def solve_for_a(Q, s1, s2, A, A_mask, lambda2, n_orients=3, max_iter=5000, 
                lagsparsity=True, tol=1e-5, verbose=0):
    """
    solve for A using group sparse proximal gradient.
    """
    if lambda2 == 0:
        return jnp.linalg.solve(s2, s1.T).T

    m = A.shape[0]
    p = A.shape[1] // m

    assert jnp.count_nonzero(A != (A*A_mask)) == 0

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

        if lagsparsity:
            norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4), keepdims=True))
        else:
            norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 2, 4), keepdims=True))

        return lambda2 * jnp.sum(norms)

    def prox_fun(x, t):
        n_sources = m // n_orients

        B = x.reshape(n_sources, n_orients, p, n_sources, n_orients)

        if lagsparsity:
            norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4), keepdims=True))
        else:
            norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 2, 4), keepdims=True))

        scale = jnp.maximum(1.0 - lambda2 * t / jnp.maximum(norms, 1e-12), 0.0)

        B = B * scale
        x_new = B.reshape(x.shape)

        # !! unwhiten the A matrix to apply the mask correctly !!

        # enforce link testing constraints in unrotated space
        A_current = (q_sqrt @ x_new) / d_safe[None, :]
        A_masked = A_current * A_mask
        
        # re-whiten to rotated space for next proxgrad step
        x_new = (q_inv_sqrt @ A_masked) * d_safe[None, :]

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
    
    assert (A_new * (~A_mask.astype(bool)).astype(float)).sum() == 0
    
    return A_new


def make_block_spd_and_diagonal_shrink(
    q_blocks,
    kappa=0.25,
    eig_floor=1e-10,
    reference_scale=1e-8,
):
    q_blocks = 0.5 * (
        q_blocks + jnp.swapaxes(q_blocks, -1, -2)
    )

    eigval, eigvec = jnp.linalg.eigh(q_blocks)
    block_scale = jnp.maximum(
        jnp.max(jnp.abs(eigval), axis=-1, keepdims=True),
        reference_scale,
    )
    eigval = jnp.maximum(eigval, eig_floor * block_scale)

    q_spd = jnp.einsum(
        "...ij,...j,...kj->...ik",
        eigvec,
        eigval,
        eigvec,
    )
    q_spd = 0.5 * (
        q_spd + jnp.swapaxes(q_spd, -1, -2)
    )

    p = q_blocks.shape[-1]
    eye = jnp.eye(p, dtype=q_blocks.dtype)
    diag = jnp.diagonal(q_spd, axis1=-2, axis2=-1)
    q_diag = eye * diag[..., None, :]

    q_new = q_diag + kappa * (q_spd - q_diag)
    return 0.5 * (
        q_new + jnp.swapaxes(q_new, -1, -2)
    )


def solve_for_Q(A, s1, s2, s3, n_transitions, block_size, nu0=None, q_base=1e-4,
    singular_values=None, source_mass=None, sigma_gamma=0.0, sigma_min=0.25,
    sigma_max=4.0, eig_floor=1e-10):
    """
    Inverse-Wishart MAP update for a block-diagonal innovation covariance Q.

    Qhat := scatter matrix Q update (from smoother statistics)

                 [(nu0 + block_size + 1) * Q0_r] + [n_transitions * Qhat]
    Q_wish_map = --------------------------------------------------------
                           nu0 + n_transitions + block_size + 1    

    Where Q0_r = Q_base * sigma^(2 * sigma_gamma)
    
    the sigma exponent scales the weighting of sigma, which are the normalized
    singular values from the leadfield compression

    Parameters
    ----------
    A, s1, s2, s3
        A is the transition matrix. s1, s2, s3 are *average* RTS/EM sufficient
        statistics, so their residual expression is an average expected
        innovation covariance.

    n_transitions : int
        Number of time transitions used by the smoother statistics.

    block_size : int
        Number of retained modes per catchment, e.g. 3.

    nu0 : float or None
        Inverse-Wishart prior degrees of freedom. Must exceed block_size - 1. If
        None, uses block_size + 2, a weak proper prior.

    q_base : float
        Global prior mode of Q when singular_values is None or sigma_gamma=0.

    singular_values : array or None
        Shape (n_blocks, block_size), one retained singular-value vector per
        catchment. These should be from the local SVD before they would have
        been placed in the leadfield.

    source_mass : array or None
        Shape (n_blocks,). Use fine-source count N_r for a basic correction only
        when fine source columns represent equal mass. Prefer total quadrature
        mass: sum(vertex areas) or sum(voxel volumes). Singular values are
        divided by sqrt(source_mass).

    sigma_gamma : float
        Strength of normalized singular-value modulation:
          0.0 -> Q prior ignores singular values 0.25 -> weak modulation 1.0 ->
          full normalized sigma^2 scaling

    eig_floor : float
        Final strict-SPD eigenvalue floor.
    """
    q_hat = s3 - A @ s1.T - s1 @ A.T + A @ s2 @ A.T
    q_hat = 0.5 * (q_hat + q_hat.T)

    m = q_hat.shape[0]
    if m % block_size != 0:
        raise ValueError(
            f"State dimension {m} is not divisible by block_size={block_size}"
        )

    n_blocks = m // block_size
    if nu0 is None:
        nu0 = float(block_size + 2)

    if nu0 <= block_size - 1:
        raise ValueError(
            "nu0 must be greater than block_size - 1 for a proper "
            "inverse-Wishart prior."
        )

    idx = jnp.arange(n_blocks)

    q4 = q_hat.reshape(
        n_blocks, block_size, n_blocks, block_size
    )
    q_blocks = q4[idx, :, idx, :]
    q_blocks = 0.5 * (
        q_blocks + jnp.swapaxes(q_blocks, -1, -2)
    )

    # define the prior mode Q0_r = q_base * I
    if singular_values is None or sigma_gamma == 0.0:
        # in this case we don't normalize using leadfield metrics
        q0_blocks = q_base * jnp.broadcast_to(
            jnp.eye(block_size, dtype=q_hat.dtype),
            (n_blocks, block_size, block_size),
        )
    else:
        # normalize using leadfield metrics (e.g., catchment/voronoi region
        # singular values)
        sigma = jnp.asarray(singular_values, dtype=q_hat.dtype)

        if sigma.shape != (n_blocks, block_size):
            raise ValueError(
                "singular_values must have shape "
                f"({n_blocks}, {block_size}), got {sigma.shape}"
            )

        # remove first-order catchment-size/source-mass dependence. this is
        # necessary because different catchment (voronoi) regions have different
        # numbers of fine source vectors and represent differently-sized
        # volumes. the condensed leadfield singular values scale with N, the
        # number of fine vectors in the catchment, so we may optionally
        # normalize them 
        if source_mass is not None:
            source_mass = jnp.asarray(
                source_mass,
                dtype=q_hat.dtype,
            )

            if source_mass.shape != (n_blocks,):
                raise ValueError(
                    "source_mass must have shape "
                    f"({n_blocks},), got {source_mass.shape}"
                )

            sigma = sigma / jnp.sqrt(
                jnp.maximum(source_mass[:, None], 1e-12)
            )

        # define a centering value and rescale all singular values to be wrt the
        # reference
        positive_sigma = jnp.where(sigma > 0.0, sigma, jnp.nan)
        sigma_ref = jnp.nanmedian(positive_sigma)

        sigma_norm = sigma / jnp.maximum(sigma_ref, 1e-12)
        sigma_norm = jnp.clip(sigma_norm, sigma_min, sigma_max)

        # q0_j = q_base * sigma_norm_j^(2 * gamma)
        # gamma=0 returns isotropic blocks.
        prior_var = q_base * sigma_norm ** (2.0 * sigma_gamma)
        q0_blocks = jax.vmap(jnp.diag)(prior_var)

    # if Q ~ IW(Psi0, nu0), select Psi0 so mode(Q) = Q0:
    # mode(IW(Psi0, nu0)) = Psi0 / (nu0 + block_size + 1).
    psi0_blocks = (nu0 + block_size + 1.0) * q0_blocks

    # posterior: IW(Psi0 + n Qhat, nu0 + n).
    # its mode is:
    # (Psi0 + n Qhat) / (nu0 + n + block_size + 1).
    q_blocks = (psi0_blocks + n_transitions * q_blocks) /\
               (nu0 + n_transitions + block_size + 1.0)

    # symmetrize for safety
    q_blocks = 0.5 * (q_blocks + jnp.swapaxes(q_blocks, -1, -2))

    # numerical SPD enforcement; should rarely change blocks if smoother
    # statistics and model updates are internally consistent.
    eigval, eigvec = jnp.linalg.eigh(q_blocks)

    block_scale = jnp.maximum(jnp.max(jnp.abs(eigval), axis=-1, keepdims=True),
                              q_base)

    eigval = jnp.maximum(eigval, eig_floor * block_scale)

    q_blocks = jnp.einsum("...ij,...j,...kj->...ik", eigvec, eigval, eigvec)

    # symmetrize for safety
    q_blocks = 0.5 * (q_blocks + jnp.swapaxes(q_blocks, -1, -2))

    Q_new = jnp.zeros_like(q_hat).reshape(n_blocks, block_size, 
                                          n_blocks, block_size)
    
    Q_new = Q_new.at[idx, :, idx, :].set(q_blocks)

    return Q_new.reshape(m, m)

def _unpack_chol(theta, d):
    """lower-triangular cholesky factor from unconstrained parameters."""
    L = jnp.zeros((d, d), dtype=theta.dtype).at[jnp.tril_indices(d)].set(theta)
    return L.at[jnp.diag_indices(d)].set(jnp.exp(jnp.diag(L)))


def _pack_chol(L, d):
    """inverse of _unpack_chol."""
    L = L.at[jnp.diag_indices(d)].set(jnp.log(jnp.diag(L)))
    return L[jnp.tril_indices(d)]

def q_block_objective(theta, S, a_s, a_c, d):
    """
    penalized objective for a single Q block, in cholesky coordinates.

    a_s * sum(log var) + a_c * logdet C + tr(Q^-1 S), which expands back to
    the IW log-posterior plus -c * logdet C
    """
    L = _unpack_chol(theta, d)
    Q = L @ L.T

    log_var = jnp.log(jnp.diag(Q))
    logdet_Q = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    logdet_C = logdet_Q - jnp.sum(log_var)

    return a_s * jnp.sum(log_var) + a_c * logdet_C + jnp.trace(jnp.linalg.solve(Q, S))


def penalized_q_objective(A, Q, s1, s2, s3, lambda_, n_orients, lagsparsity):
    """
    compute the penalized Q function objective (up to additive constants).
    """

    m = A.shape[0]
    p = A.shape[1] // m

    # -------- quadratic A term --------
    quad = jnp.trace((A @ s2) @ A.T) - 2.0 * jnp.trace(s1 @ A.T)

    # -------- covariance term --------
    Sigma = s3 - A @ s1.T - s1 @ A.T + A @ s2 @ A.T

    Sigma = 0.5 * (Sigma + Sigma.T)

    sign, logdet = jnp.linalg.slogdet(Q)

    q_term = logdet + jnp.trace(jnp.linalg.solve(Q, Sigma))

    # -------- sparsity penalty --------
    n_sources = m // n_orients

    B = A.reshape(n_sources, n_orients, p, n_sources, n_orients)

    if lagsparsity:
        norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 4)))
    else:
        norms = jnp.sqrt(jnp.sum(B * B, axis=(1, 2, 4)))

    penalty = lambda_ * jnp.sum(norms)

    total = quad + q_term + penalty

    return total, quad, q_term, penalty


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
