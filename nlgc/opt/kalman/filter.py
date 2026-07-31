import numpy as np
from scipy import linalg
from .steady_state import (solve_ss_covariance_qz, 
                           solve_ss_covariance_newton_raphson)
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
    N0 = em_state.N0

    #---------------------------------------------------------------------------
    # steady-state kalman filter
    #---------------------------------------------------------------------------

    # companion jitter for inversion stability
    jitter = 1e-8
    Q += jnp.eye(Q.shape[0]) * jitter

    (P_pred, N_pred) = solve_ss_covariance_newton_raphson(A.T, F.T, Q, R, 
                                                          P0, N0)

    FP_pred = F @ P_pred 
    innovation_cov = FP_pred @ F.T + R
    (chol, lflag) = jax.scipy.linalg.cho_factor(innovation_cov)

    # S := F P F^T + R
    # solve S kalman_gain = F P
    # -> kalman_gain^T =  P F^T S^{-1}
    kalman_gain = jax.scipy.linalg.cho_solve((chol, lflag), FP_pred).T

    innovation_precision = jax.scipy.linalg.cho_solve((chol, lflag), 
                                                      jnp.eye(N_sensors))

    # already multiplied by 1/2
    logdet_innovation_cov = jnp.log(jnp.diag(chol)).sum() 
    
    # smoother filtering from prediction P_filt = (1 - K F) P_pred
    P_filt = P_pred - kalman_gain @ FP_pred

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
        
    return filter_result
    

def forward_filter_blas(y, F, R, em_state, use_lapack=True):
    #---------------------------------------------------------------------------
    # setup
    #---------------------------------------------------------------------------
    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape

    A = em_state.A
    Q = em_state.Q
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

    P_pred = solve_ss_covariance_qz(A, F, Q, R)
    
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
        
    return filter_result
        

def rts_smoother_jax(y, F, R, em_state):
    #---------------------------------------------------------------------------
    # rts smoother setup
    #---------------------------------------------------------------------------
   
    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A

    # smoother operates on filtered states
    filter_result = forward_filter_jax(y, F, R, em_state)

    filtered_state = filter_result.filtered_state
    predicted_state = filter_result.predicted_state
    P_pred = filter_result.predicted_cov
    P_filt = filter_result.filtered_cov
    
    # --------------------------------------------------------------------------
    # RTS smoother
    # --------------------------------------------------------------------------

    # solve P_pred smoother_gain = A P_filt
    # -> smoother_gain^T = P_filt A^T P_pred^{-1}    
    cho_factor = jax.scipy.linalg.cho_factor(P_pred, lower=True, 
                                             check_finite=False)
    smoother_gain = jax.scipy.linalg.cho_solve(cho_factor, A @ P_filt, 
                                               check_finite=False).T

    
    # J := smoother_gain
    # P_smoothed = P_filt + J (P_smoothed − P_pred) J^T.
    # P_smoothed = P_filt + J P_smoothed J^T - J P_pred J^T
    # P_hat := P_filt - J P_pred J^T
    # substitute for lyapunov form J P_smoothed J^T - P_smoothed + P_hat = 0
    P_hat = P_filt - smoother_gain @ P_pred @ smoother_gain.T

    # solve_discrete lyapunov doesn't exist in jax ecosystem, so we use
    # conjugate gradients:
    # J := smoother_gain
    # J P_smoothed J^T - P_smoothed + P_hat = 0
    # L(P_smoothed) = P_smoothed - J P_smoothed J^T
    # then L(P_smoothed) = P_hat can be solved by CG
    def lyap_operator(x):
        X = x.reshape(N_sources, N_sources) # CG takes vector input
        Y = X - smoother_gain @ X @ smoother_gain.T
        return Y.ravel()

    b = P_hat.ravel() # GC rhs must be vector

    x, info = jax.scipy.sparse.linalg.bicgstab(
        lyap_operator,
        b,
        x0=P_filt.ravel(),      
        tol=1e-8,
        maxiter=10,
    )

    P_smoothed = x.reshape(N_sources, N_sources)

    #---------------------------------------------------------------------------
    # backward smoothing
    #--------------------------------------------------------------------------

    indices = jnp.arange(N_times - 1)

    def smoother_step(smoothed_state, t):
        update_residual = smoothed_state[t + 1] - predicted_state[t + 1]

        smoothed_state = smoothed_state.at[t].add(
            smoother_gain @ update_residual
        )

        return smoothed_state, None

    smoothed_state, _ = jax.lax.scan(
        smoother_step,
        filtered_state,
        indices,
        reverse=True,
    )
    
    smoother_result = RTSSmootherResult(
        smoothed_state = smoothed_state,   
        smoothed_cov = P_smoothed,
        smoother_gain = smoother_gain,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return smoother_result


def rts_smoother_blas(y, F, R, em_state, use_lapack=True):
    #---------------------------------------------------------------------------
    # rts smoother setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A
    
    # smoother operates on filtered states
    filter_result = forward_filter_blas(y, F, R, em_state, use_lapack)

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

    return smoother_result



def disturbance_smoother_blas(y, F, R, em_state, use_lapack=True):
    #---------------------------------------------------------------------------
    # disturbance smoother setup
    #---------------------------------------------------------------------------

    assert y.shape[1] == F.shape[0]
    N_times = y.shape[0]
    N_sensors, N_sources = F.shape
    A = em_state.A
    
    # smoother operates on filtered states
    filter_result = forward_filter_blas(y, F, R, em_state, use_lapack)

    innovation_precision = filter_result.innovation_precision
    kalman_gain = filter_result.kalman_gain
    predicted_state = filter_result.predicted_state

    # --------------------------------------------------------------------------
    # disturbance smoother
    # --------------------------------------------------------------------------

    proj_innov_precision = F.T @ innovation_precision @ F
    L = A - A @ kalman_gain @ F 
    information_mat = linalg.solve_discrete_lyapunov(L, proj_innov_precision)
    
    AK = A @ kalman_gain

    disturbance_information = proj_innov_precision + \
                                    AK.T @ information_mat @ AK
    

    
    innovation = y - predicted_state @ F.T

    # project innovations into information space
    innovation_information = innovation @ innovation_precision

    # disturbance smoother recursion
    backward_information = np.empty((N_times, N_sources), dtype=np.float64)
    smoothed_disturbance = np.empty_like(y)

    backward_information[-1] = 0.0

    proj_innovation_information = F.T @ innovation_information

    for t in reversed(range(N_times)):
        # smoothed measurement disturbance
        smoothed_disturbance[t] = innovation_information[t] - \
                                  AK.T @ backward_information[t]

        # backward information recursion
        if t > 0:
            backward_information[t - 1] = proj_innovation_information[t] + \
                                          L.T @ backward_information[t]
            

    # sum_t (n_t^T C n_t) where n is disturbance vec and C is disturbance info
    # constructed from precision matrices. Similar to mahalanobis distance
    # x^T\Sigma^{-1}x. this gives us the average disturbances (state noise 
    # residuals) basically scaled by the noise variances
    negative_avg_scaled_disturbance = \
                    -np.sum((smoothed_disturbance @ disturbance_information) * \
                    smoothed_disturbance)

    smoother_result = DisturbanceSmootherResult(
        disturbance = smoothed_disturbance,
        disturbance_information = disturbance_information,
        model_fit = negative_avg_scaled_disturbance,
        negative_log_likelihood = filter_result.negative_log_likelihood,
    )

    return smoother_result



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
