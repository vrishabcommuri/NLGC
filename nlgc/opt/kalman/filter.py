import numpy as np
from scipy import linalg
from .steady_state import solve_ss_covariance
from numpy.typing import NDArray
import dataclasses
from dataclasses import dataclass
from typing import Union
import jax
import jax.numpy as jnp  
jax.config.update("jax_enable_x64", True)


Array = NDArray[np.float64] | jnp.ndarray

@jax.tree_util.register_dataclass
@dataclass
class FilterResult:
    filtered_state: Array
    predicted_state: Array
    filtered_cov: Array
    predicted_cov: Array
    negative_log_likelihood: float
    innovation_precision: Array
    kalman_gain: Array
    predicted_cov_directional_derivative: Union[Array, None] = None

@jax.tree_util.register_dataclass
@dataclass
class RTSSmootherResult:
    smoothed_state: Union[Array, None] = None
    smoothed_cov: Union[Array, None] = None
    smoother_gain: Union[Array, None] = None
    negative_log_likelihood: float = 0.0 # copied from filter for convenience

@dataclass
class DisturbanceSmootherResult:
    disturbance: Array
    disturbance_information: Array
    model_fit: float
    negative_log_likelihood: float # copied from filter for convenience


def _copycast_rtssmoother_result_numpy(smoother_result):
    return dataclasses.replace(
            smoother_result,
            smoothed_state = np.array(smoother_result.smoothed_state),
            smoothed_cov = np.array(smoother_result.smoothed_cov),
            smoother_gain = np.array(smoother_result.smoother_gain),
            negative_log_likelihood = \
                np.float64(smoother_result.negative_log_likelihood),
    )


def forward_filter_preamble(A, F, Q, R, P0):
    N_sensors, _ = F.shape
    (P_pred, N_pred) = solve_ss_covariance(A, F, Q, R, P0)

    FP_pred = F @ P_pred 
    innovation_cov = FP_pred @ F.T + R
    (chol, lflag) = linalg.cho_factor(innovation_cov)

    # S := F P F^T + R
    # solve S kalman_gain = F P
    # -> kalman_gain^T =  P F^T S^{-1}
    kalman_gain = linalg.cho_solve((chol, lflag), FP_pred).T

    innovation_precision = linalg.cho_solve((chol, lflag), np.eye(N_sensors))

    # already multiplied by 1/2
    logdet_innovation_cov = np.log(np.diag(chol)).sum() 
    
    # smoother filtering from prediction P_filt = (1 - K F) P_pred
    P_filt = P_pred - kalman_gain @ FP_pred
    
    return P_pred, N_pred, kalman_gain, innovation_precision, \
           logdet_innovation_cov, P_filt


@jax.jit
def forward_filter_jax(y, F, R, em_state):
    """
    lightweight kalman filter that uses jax-safe routines and updates state
    covariances using fixed iterations of a newton-raphson update instead of qz
    decomposition. 
    
    this is intended for parameters identified after burn-in using the more
    expensive but more robust blas routines
    """
    #---------------------------------------------------------------------------
    # setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A
    Q = em_state.Q
    P0 = em_state.P0

    # companion jitter for inversion stability
    jitter = 1e-8
    Q += jnp.eye(Q.shape[0]) * jitter
    Q = 0.5 * (Q + Q.T)

    #---------------------------------------------------------------------------
    # steady-state kalman filter
    #---------------------------------------------------------------------------

    # compute these quantities outside of the jax filter for numerical stability
    dtype = A.dtype
    out_types = (
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # P_pred
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # N_pred
        jax.ShapeDtypeStruct((N_sources, N_sensors), dtype), # kalman_gain
        jax.ShapeDtypeStruct((N_sensors, N_sensors), dtype), # innov_precision
        jax.ShapeDtypeStruct((), dtype),                     # logdet_innov_cov
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # P_filt
    )

    (P_pred, N_pred, kalman_gain, innovation_precision, 
     logdet_innovation_cov, P_filt) = jax.pure_callback(
        forward_filter_preamble, 
        out_types, A, F, Q, R, P0, vmap_method='sequential'
    )

    #---------------------------------------------------------------------------
    # filtering setup
    #---------------------------------------------------------------------------
    
    negative_log_likelihood = 0.0

    #---------------------------------------------------------------------------
    # forward filtering
    #---------------------------------------------------------------------------

    def filter_step(carry, y_t):
        filtered_prev, negative_log_likelihood = carry

        # Predict
        predicted = A @ filtered_prev

        # Update
        innovation = y_t - F @ predicted
        filtered = predicted + kalman_gain @ innovation

        negative_log_likelihood += 0.5 * \
                          innovation.T @ innovation_precision @ innovation + \
                          logdet_innovation_cov

        carry = (filtered, negative_log_likelihood)
        outputs = (predicted, filtered)

        return carry, outputs

    filtered_state_time0 = jnp.zeros(N_sources)

    (_, negative_log_likelihood), (predicted_state, filtered_state) = (
        jax.lax.scan(
            filter_step,
            init=(filtered_state_time0, negative_log_likelihood),
            xs=y,
        )
    )
        
    filter_result = FilterResult(
        filtered_state = filtered_state,
        predicted_state = predicted_state,
        filtered_cov = P_filt,
        predicted_cov = P_pred,
        predicted_cov_directional_derivative = N_pred,
        innovation_precision = innovation_precision,
        kalman_gain = kalman_gain,
        negative_log_likelihood = negative_log_likelihood
    )

    em_state = dataclasses.replace(em_state, P0=P_pred, N0=N_pred)
        
    return em_state, filter_result
    

def forward_filter_blas(y, F, R, em_state, use_lapack=True):
    #---------------------------------------------------------------------------
    # setup
    #---------------------------------------------------------------------------
    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape

    A = em_state.A
    Q = em_state.Q
    P0 = em_state.P0
    predicted_state = np.empty((N_times, N_sources), dtype=np.float64)
    filtered_state = np.empty_like(predicted_state)


    assert predicted_state.shape[0] == y.shape[0]
    assert predicted_state.flags['C_CONTIGUOUS']
    assert filtered_state.flags['C_CONTIGUOUS']

    
    #---------------------------------------------------------------------------
    # steady-state kalman filter
    #---------------------------------------------------------------------------

    # companion jitter for inversion stability
    jitter = 1e-8
    Q += np.eye(Q.shape[0]) * jitter

    P_pred, _ = solve_ss_covariance(A, F, Q, R, P0)
    
    FP_pred = F @ P_pred 
    innovation_cov = FP_pred @ F.T + R
    (chol, lflag) = linalg.cho_factor(innovation_cov)

    # S := F P F^T + R
    # solve S kalman_gain = F P
    # -> kalman_gain^T =  P F^T S^{-1}
    kalman_gain = linalg.cho_solve((chol, lflag), FP_pred).T

    innovation_precision = linalg.cho_solve((chol, lflag), np.eye(N_sensors))

    # already multiplied by 1/2
    logdet_innovation_cov = np.log(np.diag(chol)).sum() 
    
    # smoother filtering from prediction P_filt = (1 - K F) P_pred
    P_filt = P_pred - kalman_gain @ FP_pred
    
    #---------------------------------------------------------------------------
    # filtering setup
    #---------------------------------------------------------------------------
    
    # cast to F_contiguous arrays for BLAS
    F, A, kalman_gain = align_cast((F, A, kalman_gain), use_lapack)
    
    innovation = np.empty(N_sensors)

    if use_lapack:
        gemv = linalg.get_blas_funcs(["gemv"], (A, filtered_state[0]))[0]

    negative_log_likelihood = 0.0

    #---------------------------------------------------------------------------
    # forward filtering
    #---------------------------------------------------------------------------

    for t in range(N_times):
        # predict
        if t == 0:
            predicted_state[t] = 0
        else:
            if use_lapack:
                predicted_state[t] = gemv(1, A, filtered_state[t - 1], beta=0,
                                          y=predicted_state[t], 
                                          overwrite_y=True)
            else:
                predicted_state[t] = A @ filtered_state[t - 1]

        filtered_state[t] = predicted_state[t]

        # update
        if use_lapack:
            innovation[:] = y[t]
            innovation = gemv(-1, F, predicted_state[t], beta=1, y=innovation,
                              overwrite_y=True)

            gemv(1, kalman_gain, innovation, beta=1, y=filtered_state[t],
                 overwrite_y=True)
        else:
            innovation[:] = y[t] - F @ predicted_state[t]
            filtered_state[t] += kalman_gain @ innovation


        negative_log_likelihood += 0.5 * \
                innovation.T @ innovation_precision @ innovation + \
                logdet_innovation_cov
        
        
    filter_result = FilterResult(
        filtered_state = filtered_state,
        predicted_state = predicted_state,
        filtered_cov = P_filt,
        predicted_cov = P_pred,
        innovation_precision = innovation_precision,
        kalman_gain = kalman_gain,
        negative_log_likelihood = negative_log_likelihood
    )

    em_state = dataclasses.replace(em_state, P0=P_pred)
        
    return em_state, filter_result


def rts_smoother_preamble(A, P_pred, P_filt):
    # solve P_pred smoother_gain = A P_filt
    # -> smoother_gain^T = P_filt A^T P_pred^{-1}  
    try:
        cho_factor = linalg.cho_factor(P_pred, lower=True, check_finite=False)
        smoother_gain = linalg.cho_solve(cho_factor, A @ P_filt, check_finite=False).T
    except np.linalg.LinAlgError:
        smoother_gain, *_ = linalg.lstsq(P_pred, A @ P_filt, check_finite=False)
        smoother_gain = smoother_gain.T

    # J := smoother_gain
    # P_smoothed = P_filt + J (P_smoothed − P_pred) J^T.
    # P_smoothed = P_filt + J P_smoothed J^T - J P_pred J^T
    # P_hat := P_filt - J P_pred J^T
    # substitute for lyapunov form J P_smoothed J^T - P_smoothed + P_hat = 0
    P_hat = P_filt - smoother_gain @ P_pred @ smoother_gain.T
    P_smoothed = linalg.solve_discrete_lyapunov(smoother_gain, P_hat)
    P_smoothed = 0.5 * (P_smoothed + P_smoothed.T)

    return smoother_gain, P_smoothed       


def rts_smoother_jax(y, F, R, em_state):
    #---------------------------------------------------------------------------
    # rts smoother setup
    #---------------------------------------------------------------------------
   
    assert y.shape[1] == F.shape[0]
    N_sources = F.shape[1]
    A = em_state.A

    # smoother operates on filtered states
    em_state, filter_result = forward_filter_jax(y, F, R, em_state)

    filtered_state = filter_result.filtered_state
    predicted_state = filter_result.predicted_state
    P_pred = filter_result.predicted_cov
    P_filt = filter_result.filtered_cov
    
    # --------------------------------------------------------------------------
    # RTS smoother
    # --------------------------------------------------------------------------

    dtype = A.dtype
    out_types = (
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # smoother_gain
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # P_smoothed
    )

    smoother_gain, P_smoothed = jax.pure_callback(
        rts_smoother_preamble,
        out_types, A, P_pred, P_filt, vmap_method='broadcast_all'
    )

    #---------------------------------------------------------------------------
    # backward smoothing
    #--------------------------------------------------------------------------

    def smoother_step(smoothed_t_plus_1, inputs):
        filtered_t, predicted_t_plus_1 = inputs
        
        update_residual = smoothed_t_plus_1 - predicted_t_plus_1

        smoothed_t = filtered_t + smoother_gain @ update_residual
    
        return smoothed_t, smoothed_t

    smoothed_last = filtered_state[-1]

    _, smoothed_seq = jax.lax.scan(
        smoother_step,
        init=smoothed_last,
        xs=(filtered_state[:-1], predicted_state[1:]),
        reverse=True,
    )
    
    # scan outputs are [t=0, t=1, ..., t=N-2]. Append the last state t=N-1
    smoothed_state = jnp.vstack([smoothed_seq, smoothed_last[None, :]])
    
    smoother_result = RTSSmootherResult(
        smoothed_state = smoothed_state,   
        smoothed_cov = P_smoothed,
        smoother_gain = smoother_gain,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return em_state, smoother_result


def rts_smoother_blas(y, F, R, em_state, use_lapack=True):
    #---------------------------------------------------------------------------
    # rts smoother setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A
    
    # smoother operates on filtered states
    em_state, filter_result = forward_filter_blas(y, F, R, em_state, use_lapack)

    filtered_state = filter_result.filtered_state.copy()
    predicted_state = filter_result.predicted_state
    P_pred = filter_result.predicted_cov
    P_filt = filter_result.filtered_cov
    
    # --------------------------------------------------------------------------
    # RTS smoother
    # --------------------------------------------------------------------------

    # solve P_pred smoother_gain = A P_filt
    # -> smoother_gain^T = P_filt A^T P_pred^{-1}
    # if nonconvergence do psuedoinverse
    try:    
        cho_factor = linalg.cho_factor(P_pred, lower=True, check_finite=False)
        smoother_gain = linalg.cho_solve(cho_factor, A @ P_filt, 
                                         check_finite=False).T

    except np.linalg.LinAlgError:
        smoother_gain, *rest = linalg.lstsq(P_pred, A @ P_filt, 
                                            check_finite=False)
        smoother_gain = smoother_gain.T

    # J := smoother_gain
    # P_smoothed = P_filt + J (P_smoothed − P_pred) J^T.
    # P_smoothed = P_filt + J P_smoothed J^T - J P_pred J^T
    # P_hat := P_filt - J P_pred J^T
    # substitute for lyapunov form J P_smoothed J^T - P_smoothed + P_hat = 0
    P_hat = P_filt - smoother_gain @ P_pred @ smoother_gain.T
    P_smoothed = linalg.solve_discrete_lyapunov(smoother_gain, P_hat)

    # enforce numerical symmetry
    P_smoothed = 0.5 * (P_smoothed + P_smoothed.T)

    #---------------------------------------------------------------------------
    # smoothing setup
    #---------------------------------------------------------------------------

    smoother_gain ,= align_cast((smoother_gain,), use_lapack)

    if use_lapack:
        gemv = linalg.get_blas_funcs(["gemv"], (A, filtered_state[0]))[0]

    update_residual = np.empty(N_sources)

    #---------------------------------------------------------------------------
    # backward smoothing
    #---------------------------------------------------------------------------

    for t in reversed(range(N_times - 1)):
        update_residual[:] = filtered_state[t+1] - predicted_state[t+1]

        if use_lapack:
            gemv(1, smoother_gain, update_residual, beta=1, y=filtered_state[t],
                overwrite_y=True)
        else:
            filtered_state[t] += smoother_gain @ update_residual

    
    smoother_result = RTSSmootherResult(
        smoothed_state = filtered_state,   # overwritten to save a gemv alloc
        smoothed_cov = P_smoothed,
        smoother_gain = smoother_gain,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return em_state, smoother_result


def disturbance_smoother_preamble(A, kalman_gain, F, innovation_precision):
    # L is the closed-loop state transition matrix 
    L = A - A @ kalman_gain @ F
    
    # M is the projected innovation precision 
    proj_innov_precision = F.T @ innovation_precision @ F

    # solve the discrete Lyapunov equation for the state disturbance information
    # N. we want N = L^T N L + M. scipy solves a x a^H - x + q = 0, therefore
    # we must pass L.T to compute the backward variance.
    information_mat = linalg.solve_discrete_lyapunov(L.T, proj_innov_precision)
    information_mat = 0.5 * (information_mat + information_mat.T)

    return L, information_mat


def disturbance_smoother_jax(y, F, R, em_state):
    #---------------------------------------------------------------------------
    # disturbance smoother setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_sources = F.shape[1]
    A = em_state.A

    # smoother operates on filtered states
    em_state, filter_result = forward_filter_jax(y, F, R, em_state)

    innovation_precision = filter_result.innovation_precision
    kalman_gain = filter_result.kalman_gain
    predicted_state = filter_result.predicted_state

    # --------------------------------------------------------------------------
    # disturbance smoother preamble
    # --------------------------------------------------------------------------

    dtype = A.dtype
    out_types = (
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # L
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype), # information_mat
    )

    L, information_mat = jax.pure_callback(
        disturbance_smoother_preamble,
        out_types, A, kalman_gain, F, innovation_precision,
        vmap_method='broadcast_all'
    )

    # for latent state process noise, the disturbance information is exactly N
    disturbance_information = information_mat

    # --------------------------------------------------------------------------
    # backward smoothing
    # --------------------------------------------------------------------------

    innovation = y - predicted_state @ F.T

    # project innovations into information space
    # (T, N_sensors) @ (N_sensors, N_sensors) = (T, N_sensors)
    innovation_information = innovation @ innovation_precision
    
    # (T, N_sensors) @ (N_sensors, N_sources) = (T, N_sources)
    proj_innovation_information = innovation_information @ F

    def smoother_step(r_t_plus_1, proj_innov_t):
        # backward information recursion for the latent state disturbance (r_t)
        # r_t = F^T \Omega v_t + L^T r_{t+1}
        r_t = proj_innov_t + L.T @ r_t_plus_1
        return r_t, r_t

    # init
    r_last = jnp.zeros(N_sources, dtype=dtype)

    _, smoothed_disturbance = jax.lax.scan(
        smoother_step,
        init=r_last,
        xs=proj_innovation_information,
        reverse=True,
    )

    # sum_t (r_t^T N r_t) where r_t is the state disturbance vector 
    # and N is its variance/information matrix
    negative_avg_scaled_disturbance = \
        -jnp.sum((smoothed_disturbance @ disturbance_information) * \
        smoothed_disturbance)

    smoother_result = DisturbanceSmootherResult(
        disturbance = smoothed_disturbance,
        disturbance_information = disturbance_information,
        model_fit = negative_avg_scaled_disturbance,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return em_state, smoother_result



def measurement_smoother_preamble(A, kalman_gain, F, innovation_precision):
    L = A - A @ kalman_gain @ F
    proj_innov_precision = F.T @ innovation_precision @ F

    # solve N = L^T N L + F^T \Omega F for the backward information variance
    information_mat = linalg.solve_discrete_lyapunov(L.T, proj_innov_precision)
    information_mat = 0.5 * (information_mat + information_mat.T)

    AK = A @ kalman_gain
    
    # we use innovation_precision (N_sensors, N_sensors) rather than
    # proj_innov_precision (N_sources, N_sources) so it aligns with (AK)^T N
    # (AK).
    disturbance_information = innovation_precision + AK.T @ information_mat @ AK

    return L, AK, disturbance_information


def measurement_smoother_jax(y, F, R, em_state):
    #---------------------------------------------------------------------------
    # disturbance smoother setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A

    # smoother operates on filtered states
    em_state, filter_result = forward_filter_jax(y, F, R, em_state)

    innovation_precision = filter_result.innovation_precision
    kalman_gain = filter_result.kalman_gain
    predicted_state = filter_result.predicted_state

    # --------------------------------------------------------------------------
    # disturbance smoother preamble
    # --------------------------------------------------------------------------

    dtype = A.dtype
    out_types = (
        jax.ShapeDtypeStruct((N_sources, N_sources), dtype),  # L
        jax.ShapeDtypeStruct((N_sources, N_sensors), dtype),  # AK
        jax.ShapeDtypeStruct((N_sensors, N_sensors), dtype),  # dist info
    )

    L, AK, disturbance_information = jax.pure_callback(
        measurement_smoother_preamble,
        out_types, A, kalman_gain, F, innovation_precision,
        vmap_method='broadcast_all'
    )

    # --------------------------------------------------------------------------
    # backward smoothing
    # --------------------------------------------------------------------------

    innovation = y - predicted_state @ F.T

    # (T, N_sensors) @ (N_sensors, N_sensors) = (T, N_sensors)
    innovation_information = innovation @ innovation_precision
    
    # (T, N_sensors) @ (N_sensors, N_sources) = (T, N_sources)
    proj_innovation_information = innovation_information @ F

    def smoother_step(r_t, inputs):
        innov_info_t, proj_innov_info_t = inputs

        smoothed_dist_t = innov_info_t - AK.T @ r_t

        r_t_minus_1 = proj_innov_info_t + L.T @ r_t

        return r_t_minus_1, smoothed_dist_t

    # init
    r_last = jnp.zeros(N_sources, dtype=dtype)

    _, smoothed_disturbance = jax.lax.scan(
        smoother_step,
        init=r_last,
        xs=(innovation_information, proj_innovation_information),
        reverse=True,
    )

    # sum_t (e_t^T C e_t) where e is measurement disturbance vec 
    # and C is disturbance information
    negative_avg_scaled_disturbance = \
        -jnp.sum((smoothed_disturbance @ disturbance_information) * \
        smoothed_disturbance)

    smoother_result = DisturbanceSmootherResult(
        disturbance = smoothed_disturbance,
        disturbance_information = disturbance_information,
        model_fit = negative_avg_scaled_disturbance,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return em_state, smoother_result


def align_cast(args, use_lapack):
    """internal function to typecast (to np.float64) and/or memory-align
       ndarrays

    Parameters
    ----------
    args: tuple of ndarrays of arbitrary shape use_lapack: bool
        whether to make F_contiguous or not.
    Returns
    -------
    args: tuple
        after alignment and typecasting
    """
    args = tuple([arg if arg.dtype == np.float64 else arg.astype(np.float64) 
                  for arg in args])
    if use_lapack:
        args = tuple([arg if arg.flags['F_CONTIGUOUS'] else arg.copy(order='F') 
                      for arg in args])
    return args
