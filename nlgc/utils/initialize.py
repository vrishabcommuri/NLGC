from nlgc.utils.warm_start import warm_start_sources
from nlgc.opt.em import EMState
from scipy import linalg, optimize
import numpy as np


def initialize_em_state(y, F, r, singular_values, config, evoked=None, 
                        forward=None, noise_cov=None, weights=None):
    F_companion, R_companion, em_state = companion_init(y, F, r, config)

    em_state.log_likelihood = np.zeros(config.optimizer.max_iter + 1)
    em_state.Q_prior_scales = singular_values

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


def data_driven_Q_init(y, F, target_factor=1.2, q_floor=1e-8, q_ceiling=1e8,
    svd_rtol=None, verbose=False):
    """
    Return isotropic initial Q = q I using only F's observable sensor subspace.

    Parameters
    ----------
    y : array, shape (n_sensors, n_times)
        Whitened sensor data.
    F : array, shape (n_sensors, n_states)
        Whitened condensed leadfield.
    target_factor : float
        Target scale in the original root heuristic.
    q_floor, q_ceiling : float
        Bounds for a strictly positive covariance scale.
    svd_rtol : float | None
        Relative singular-value cutoff. Default is numerical precision based.
    """
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)

    if y.ndim != 2 or F.ndim != 2:
        raise ValueError(f"Expected 2D arrays; y={y.shape}, F={F.shape}")

    n_obs, n_state = F.shape

    if y.shape[0] != n_obs:
        raise ValueError(
            f"F is ({n_obs}, {n_state}) but y is {y.shape}; expected "
            "y.shape[0] == F.shape[0]."
        )

    if not np.isfinite(y).all() or not np.isfinite(F).all():
        raise ValueError("y or F contains NaN/Inf.")

    U, s, _ = linalg.svd(F, full_matrices=False, check_finite=True)

    if s.size == 0 or s[0] == 0:
        raise ValueError("F has zero numerical rank.")

    if svd_rtol is None:
        svd_rtol = np.finfo(float).eps * max(F.shape)

    keep = s > (svd_rtol * s[0])

    if not np.any(keep):
        raise ValueError("No nonzero singular values retained for F.")

    U_obs = U[:, keep]
    eigvals = s[keep] ** 2

    # energy of y in the observation model's identifiable sensor subspace.
    projected = U_obs.T @ y
    est_source_pow = np.sum(projected**2, axis=1)

    target = target_factor * n_obs * y.shape[1]

    def fun(q):
        return np.sum(est_source_pow / (1.0 + q * eigvals)**2) - target

    f0 = fun(0.0)

    # under the observable-subspace version, f(q) -> -target < 0.
    if f0 <= 0:
        q_val = q_floor
        status = "root_not_needed_f0_nonpositive"
    else:
        lo = 0.0
        hi = max(1.0, q_floor)

        while fun(hi) > 0 and hi < q_ceiling:
            hi *= 10.0

        if fun(hi) > 0:
            q_val = q_ceiling
            status = "root_not_bracketed_used_ceiling"
        else:
            sol = optimize.root_scalar(
                fun,
                bracket=(lo, hi),
                method="brentq",
                xtol=max(q_floor * 0.1, 1e-14),
                rtol=1e-8,
            )
            q_val = max(float(sol.root), q_floor)
            status = "brentq" if sol.converged else "brentq_not_converged"

    if verbose:
        residual = y - U_obs @ (U_obs.T @ y)
        print(
            f"F={F.shape}; effective_rank={keep.sum()}; "
            f"sigma=[{s[keep].min():.3e}, {s[keep].max():.3e}]; "
            f"observable_y_fraction="
            f"{np.sum(projected**2) / np.sum(y**2):.4f}; "
            f"f(0)={f0:.3e}; q={q_val:.3e}; status={status}; "
            f"residual_norm_sq={np.sum(residual**2):.3e}"
        )

    return q_val * np.eye(n_state)