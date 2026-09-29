from nlgc.utils.warm_start import warm_start_sources
from nlgc.opt.em import EMState
from scipy import linalg, optimize
import numpy as np


def initialize_em_state(y, F, r, singular_values, config, evoked=None, 
                        forward=None, noise_cov=None, weights=None):
    F_companion, R_companion, em_state = companion_init(y, F, r, config)

    em_state.log_likelihood = np.zeros(config.optimizer.max_iter + 1)
    em_state.Q_prior_scales = singular_values
    
    if config.latent.n_eigenmodes > 1:
        assert config.latent.n_orients == 1, \
            "mixed multiple eigenmodes and orientations not supported"
    
    if config.latent.n_orients == 1:
        em_state.Q_prior_scales = singular_values.flatten()[:, np.newaxis]


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
    Q_upper, q_val = data_driven_Q_init(y, F, verbose=config.numerical.verbose)
    Q[:m,:m] = Q_upper

    F = np.hstack([F, np.zeros((total_sensor_dim, m*(p-1)))])
    R = r * np.eye(total_sensor_dim)

    em_state = EMState(
        A = A,
        A_mask = np.ones_like(zero_companion),
        Q = Q, 
        P0 = np.zeros_like(zero_companion),
        N0 = np.zeros_like(zero_companion),
        N_sources_upper = total_latent_dim,
        q_val = q_val,
    )

    return F, R, em_state


def data_driven_Q_init(y, F, target_factor=1.2, capture_fraction=0.20,
    q_floor=1e-8, q_ceiling=1e8, svd_rtol=None, verbose=False):
    """
    Initialize isotropic Q = q I for whitened data where R = I.

    The initialization combines: a scalar root heuristic, when a positive root
    exists; and a gain-engagement lower bound that prevents a nearly zero prior
         covariance from making the Kalman update inert.

    Parameters
    ----------
    y : array, shape (n_obs, n_times)
        Whitened sensor data.
    F : array, shape (n_obs, n_state)
        Whitened forward/gain matrix.
    target_factor : float
        Retained from the original scalar root criterion.
    capture_fraction : float
        Desired observation-mode innovation capture at a robust leadfield
        singular-value scale. Typical range: 0.10 to 0.30.
    q_floor, q_ceiling : float
        Hard numerical/model bounds on q.
    """
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)

    if y.ndim != 2 or F.ndim != 2:
        raise ValueError(f"Expected 2D arrays; y={y.shape}, F={F.shape}")

    n_obs, n_state = F.shape

    if y.shape[0] != n_obs:
        raise ValueError(
            f"F is {F.shape} but y is {y.shape}; "
            "expected y.shape[0] == F.shape[0]."
        )

    if not np.isfinite(y).all() or not np.isfinite(F).all():
        raise ValueError("y or F contains NaN/Inf.")

    if not (0.0 < capture_fraction < 1.0):
        raise ValueError("capture_fraction must lie strictly between 0 and 1.")

    U, s, _ = linalg.svd(F, full_matrices=False, check_finite=True)

    if s.size == 0 or s[0] <= 0.0:
        raise ValueError("F has zero numerical rank.")

    if svd_rtol is None:
        svd_rtol = np.finfo(float).eps * max(F.shape)

    keep = s > (svd_rtol * s[0])

    if not np.any(keep):
        raise ValueError("No singular values retained for F.")

    U_obs = U[:, keep]
    sigma2 = s[keep] ** 2
    projected = U_obs.T @ y
    projected_energy = np.sum(projected**2, axis=1)

    # ------------------------------------------------------------
    # gain-engagement lower bound
    # ------------------------------------------------------------
    sigma2_ref = np.mean(sigma2) # rms gain scale
    
    sigma2_ref = max(float(sigma2_ref), np.finfo(float).tiny)

    q_gain = capture_fraction / ((1.0 - capture_fraction) * sigma2_ref)


    # ------------------------------------------------------------
    # root criterion
    # ------------------------------------------------------------
    target = target_factor * n_obs * y.shape[1]

    def fun(q):
        return (
            np.sum(projected_energy / (1.0 + q * sigma2) ** 2)
            - target
        )

    f0 = fun(0.0)
    q_root = None
    root_status = "not_attempted"

    if f0 > 0.0:
        lo = 0.0
        hi = max(1.0, q_gain, q_floor)

        while fun(hi) > 0.0 and hi < q_ceiling:
            hi *= 10.0

        if fun(hi) <= 0.0:
            sol = optimize.root_scalar(
                fun,
                bracket=(lo, hi),
                method="brentq",
                xtol=max(q_floor * 0.1, 1e-14),
                rtol=1e-8,
            )

            if sol.converged:
                q_root = float(sol.root)
                root_status = "brentq"
            else:
                root_status = "brentq_not_converged"
        else:
            root_status = "root_not_bracketed"
    else:
        root_status = "f0_nonpositive"

    # never initialized below engagement scale
    if q_root is None:
        q_val = q_gain
        status = f"gain_floor_{root_status}"
    else:
        q_val = q_root
        status = root_status

    q_val = float(np.clip(q_val, q_floor, q_ceiling))

    if verbose:
        observable_fraction = np.sum(projected**2) /\
                              max(np.sum(y**2), np.finfo(float).tiny)
    

        h_ref = q_val * sigma2_ref / (1.0 + q_val * sigma2_ref)

        print(
            f"F={F.shape}; rank={int(keep.sum())}; "
            f"sigma=[{s[keep].min():.3e}, {s[keep].max():.3e}]; "
            f"sigma2_ref={sigma2_ref:.3e}; "
            f"observable_y_fraction={observable_fraction:.4f}; "
            f"f0={f0:.3e}; q_root={q_root}; "
            f"q_gain={q_gain:.3e}; q={q_val:.3e}; "
            f"h_ref={h_ref:.3f}; status={status}"
        )

    return q_val * np.eye(n_state), q_val
