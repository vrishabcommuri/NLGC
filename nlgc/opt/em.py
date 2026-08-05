import numpy as np
from numpy.typing import NDArray
from typing import Union, TypeAlias
import dataclasses
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp  
from nlgc.opt.kalman.filter import (rts_smoother_blas, rts_smoother_jax)  
from nlgc.opt.proximal import proximal_param_update  
from functools import partial  
jax.config.update("jax_enable_x64", True)

Array: TypeAlias = NDArray[np.float64] | jnp.ndarray

@jax.tree_util.register_dataclass
@dataclass 
class EMState:
    A: Union[Array, None] = None  # companion form
    A_mask: Union[Array, None] = None
    Q: Union[Array, None] = None  # companion form
    em_iter: int = 0  # how many EM steps have we done?
    log_likelihood: Union[Array, None] = None # likelihood trajectory
    P0: Union[Array, None] = None
    N0: Union[Array, None] = None

    # companion upper portion; this is marked as meta information
    # so jax can use it at compile time
    N_sources_upper: Union[int, None] = field(default=None,
                                              metadata={"static": True})  

def _triage_em_state(em_state):
    assert em_state.A is not None and len(em_state.A.shape) == 2
    assert em_state.A_mask is not None and len(em_state.A_mask.shape) == 2
    assert em_state.Q is not None and len(em_state.Q.shape) == 2
    assert em_state.P0 is not None and len(em_state.P0.shape) == 2
    assert em_state.N0 is not None and len(em_state.N0.shape) == 2
    assert em_state.N_sources_upper is not None
    assert em_state.log_likelihood is not None


def _copycast_em_state_numpy(em_state):
    return dataclasses.replace(
            em_state,
            A = np.array(em_state.A),
            A_mask = np.array(em_state.A_mask),
            Q = np.array(em_state.Q),
            P0 = np.array(em_state.P0),
            N0 = np.array(em_state.N0),
            log_likelihood = np.array(em_state.log_likelihood),
    )


def _copycast_em_state_jax(em_state):
    return dataclasses.replace(
            em_state,
            A = jnp.array(em_state.A),
            A_mask = jnp.array(em_state.A_mask),
            Q = jnp.array(em_state.Q),
            P0 = jnp.array(em_state.P0),
            N0 = jnp.array(em_state.N0),
            log_likelihood = jnp.array(em_state.log_likelihood),
    )


def solve_params(y, F, R, em_state, config, lambda_):
    """
    top-level function to set up filters and optimizers
    """

    em_state = dataclasses.replace(
        em_state,
        A = zero_entries(em_state.A, em_state.A_mask),
        em_iter=0,
        log_likelihood=np.zeros(config.optimizer.max_iter + 1),
    )

    _triage_em_state(em_state)

    em_state = _copycast_em_state_jax(em_state)
    y = jnp.array(y)
    F = jnp.array(F)
    R = jnp.array(R)

    if config.numerical.verbose:
        print(f"running JAX EM with {config.optimizer.max_iter} "
              "iterations")
        
    em_state, smoother_result = em_jax(y, F, R, em_state, config, lambda_,
                                           config.optimizer.max_iter)
        
    if config.numerical.verbose:
        print(f"EM finished in {em_state.em_iter} iterations")
                  
    return em_state, smoother_result


@partial(jax.jit, static_argnames=("config",))
def em_jax(y, F, R, em_state, config, lambda_, N_iter):
    """
    jax EM which assumes all routines called are JAX-compatible.
    """

    def em_step(iter, carry):
        convergence_count, prev_objective, em_state, _ = carry

        def skip(_):
            return carry

        def update(_):
            prev_iter = em_state.em_iter
            prev_ll = em_state.log_likelihood[prev_iter]
            

            em_new, smoother_result = rts_smoother_jax(y, F, R, em_state)

            em_new, rel_A_change, curr_objective = proximal_param_update(
                                                         em_new, 
                                                         smoother_result,
                                                         config,
                                                         lambda_)
            

            curr_ll = -smoother_result.negative_log_likelihood

            safe_prev_ll = jnp.where(jnp.abs(prev_ll) > 1e-12, prev_ll, 1.0)
            safe_prev_objective = jnp.where(jnp.abs(prev_objective) > 1e-12, 
                                            prev_objective, 1.0)

            rel_change = jnp.where(
                jnp.abs(prev_ll) > 1e-12,
                jnp.abs(curr_ll - prev_ll) / jnp.abs(safe_prev_ll),
                jnp.inf,
            )

            rel_obj_change = jnp.where(
                jnp.abs(prev_objective) > 1e-12,
                jnp.abs(curr_objective - prev_objective) / \
                    jnp.abs(safe_prev_objective),
                jnp.inf,
            )

            below_tol = (rel_obj_change <= config.optimizer.tol) | \
                        jnp.isnan(curr_ll)

            convergence_count_new = jnp.where(below_tol, convergence_count + 1, 
                                              0)

            converged = convergence_count_new >= 10

            curr_iter = jnp.where(converged, prev_iter, prev_iter + 1)

            safe_iter = jnp.minimum(curr_iter, len(em_new.log_likelihood) - 1)
            updated_ll_trajectory = em_new.log_likelihood.at[safe_iter]\
                                        .set(curr_ll)
            
            def print_progress():
                jax.debug.print(
                    "curr_iter={i}: curr_obj={cobj} prev_obj={pobj} rel_obj_change={roc} curr_ll={ll} rel_ll_change(unused)={rc}",
                    i=prev_iter,
                    cobj=curr_objective,
                    pobj=prev_objective,
                    roc=rel_obj_change,
                    ll=curr_ll,
                    rc=rel_change,
                )

            # only run the print callback every N iterations
            jax.lax.cond(
                ((prev_iter % 10) == 0) & config.numerical.verbose,
                print_progress,
                lambda: None
            )

            em_new = dataclasses.replace(
                em_new,
                A=zero_entries(em_new.A, em_new.A_mask),
                em_iter=curr_iter,
                log_likelihood=updated_ll_trajectory,
            )

            return convergence_count_new, curr_objective, em_new, smoother_result

        # must have been below convergence tol threshold 10 times in a row
        # before we declare final convergence. otherwise a transient small step
        # may be responsible for early stopping
        return jax.lax.cond(convergence_count >= 10, skip, update, operand=None)
        
    converged = 0
    init_objective = -jnp.inf
    _, smoother_result_init = rts_smoother_jax(y, F, R, em_state)
    init_carry = (converged, init_objective, em_state, smoother_result_init)

    converged, _, em_state, smoother_result = jax.lax.fori_loop(
        lower=0,
        upper=N_iter,
        body_fun=em_step,
        init_val=init_carry,
    )

    return em_state, smoother_result


def zero_entries(A, A_mask):
    A_masked = A * A_mask
    return A_masked 


def project_psd(Q, eps=1e-8):
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = jnp.linalg.eigh(Q)
    eigvals = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


################################################################################
# defunct em blas implementation preserved for old tests
################################################################################

def em_blas(y, F, R, em_state, config, lambda_, N_iter):
    """
    before passing to jax we do a few iterations with the slow but robust 
    solvers. these use heavy iterative decompositions (e.g., scipy solve dare)
    that are not available in jax. after burn-in, we can move to newton-raphson
    acceleration and jit
    """

    for _ in range(N_iter):
        smoother_result = rts_smoother_blas(y, F, R, em_state, 
                                            config.numerical.use_lapack)
        curr_iter = em_state.em_iter
        prev_ll = em_state.log_likelihood[curr_iter]

        em_state, rel_A_change = proximal_param_update(em_state, 
                                                       smoother_result, 
                                                       config,
                                                       lambda_)

        em_state = _copycast_em_state_numpy(em_state)

        em_state.A = zero_entries(em_state.A, em_state.A_mask)

        curr_ll = -smoother_result.negative_log_likelihood

        # solve_params sizes the trajectory n_warmup_iter + max_iter + 1 and
        # resets em_iter to 0, so curr_iter+1 <= n_warmup_iter is always in range
        # em_state.Q = project_psd(em_state.Q)
        em_state.log_likelihood[curr_iter+1] = curr_ll

        if np.abs(prev_ll) > 1e-12:
            rel_change = np.abs(curr_ll - prev_ll) / np.abs(prev_ll)
        else:
            rel_change = np.inf

        # print(curr_ll, np.abs(rel_change), rel_A_change)
        
        # if np.abs(rel_change) > config.optimizer.tol or \
        #     rel_A_change > config.optimizer.A_tol:
        if np.abs(rel_change) > config.optimizer.tol:
            em_state.em_iter += 1
        else:
            # converged
            return em_state, smoother_result

    return em_state, smoother_result