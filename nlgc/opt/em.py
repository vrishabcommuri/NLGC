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
    
    em_state, smoother_result = _finalize_em_state(y, F, R, em_state)
        
    if config.numerical.verbose:
        print(f"EM finished in {em_state.em_iter} iterations")
                  
    return em_state, smoother_result



@partial(jax.jit, static_argnames=("config",))
def em_jax(y, F, R, em_state, config, lambda_, N_iter):
    """
    JAX EM which assumes all routines called are JAX-compatible.

    convergence is based on the penalized objective. the model must remain below
    the objective tolerance for `convergence_patience` consecutive iterations
    before EM terminates.

    the likelihood trajectory is retained for diagnostics, but the final
    likelihood used for inference should be recomputed from the final returned
    state by `_finalize_em_state`.
    """

    # can be increased if desired
    convergence_patience = 2

    def em_step(iter, carry):
        convergence_count, prev_objective, em_state, _ = carry

        def skip(_):
            return carry

        def update(_):
            prev_iter = em_state.em_iter

            # E-step 
            em_new, smoother_result = rts_smoother_jax(y, F, R, em_state)

            # M-step
            em_new, rel_A_change, curr_objective = proximal_param_update(
                em_new,
                smoother_result,
                config,
                lambda_,
            )

            # likelihood belonging to the smoother_result generated from the OLD
            # parameter state, not em_new.A.
            curr_ll = -smoother_result.negative_log_likelihood

    
            # Increment the iteration whenever we actually perform an EM update.
            curr_iter = prev_iter + 1

            safe_iter = jnp.minimum(curr_iter, len(em_new.log_likelihood) - 1)

            updated_ll_trajectory = em_new.log_likelihood.at[safe_iter]\
                                          .set(curr_ll)

            # ---- ll diagnostic ----
            prev_ll = em_state.log_likelihood[
                jnp.minimum(
                    prev_iter,
                    len(em_state.log_likelihood) - 1,
                )
            ]

            rel_ll_change = jnp.where(
                prev_iter > 0,
                jnp.abs(curr_ll - prev_ll)
                / jnp.maximum(jnp.abs(prev_ll), 1e-12),
                jnp.inf,
            )

            # ---- objective convergence ----
            first_iteration = jnp.isneginf(prev_objective)

            rel_obj_change = jnp.where(
                first_iteration,
                jnp.inf,
                jnp.abs(curr_objective - prev_objective)
                / jnp.maximum(jnp.abs(prev_objective), 1e-12),
            )

            below_tol = (
                (~first_iteration)
                & (rel_ll_change <= config.optimizer.tol)
                & (~jnp.isnan(curr_ll))
            )

            convergence_count_new = jnp.where(
                below_tol,
                convergence_count + 1,
                0,
            )


            def print_progress():
                jax.debug.print(
                    "curr_iter={i}: "
                    "curr_obj={cobj} "
                    "prev_obj={pobj} "
                    "rel_obj_change={roc} "
                    "curr_ll={ll} "
                    "rel_ll_change={rlc} "
                    "rel_A_change={rac} "
                    "conv_count={cc}",
                    i=curr_iter,
                    cobj=curr_objective,
                    pobj=prev_objective,
                    roc=rel_obj_change,
                    ll=curr_ll,
                    rlc=rel_ll_change,
                    rac=rel_A_change,
                    cc=convergence_count_new,
                )

            jax.lax.cond(
                ((curr_iter % 10) == 0) & config.numerical.verbose,
                print_progress,
                lambda: None,
            )

            em_new = dataclasses.replace(
                em_new,
                A=zero_entries(em_new.A, em_new.A_mask),
                em_iter=curr_iter,
                log_likelihood=updated_ll_trajectory,
            )

            return convergence_count_new, curr_objective, em_new, \
                   smoother_result

        # must have been below convergence tol threshold convergence_patience
        # times in a row before we declare final convergence. otherwise a
        # transient small step may be responsible for early stopping
        return jax.lax.cond(
            convergence_count >= convergence_patience,
            skip,
            update,
            operand=None,
        )

    # ---- initialize objective ----
    _, smoother_result_init = rts_smoother_jax(
        y,
        F,
        R,
        em_state,
    )

    # first iteration explicitly treated as non-converged.
    init_objective = -jnp.inf

    init_carry = (
        0,
        init_objective,
        em_state,
        smoother_result_init,
    )

    _, _, em_state, smoother_result = jax.lax.fori_loop(
        lower=0,
        upper=N_iter,
        body_fun=em_step,
        init_val=init_carry,
    )

    return em_state, smoother_result


@jax.jit
def _finalize_em_state(y, F, R, em_state):
    """
    recompute the smoother and likelihood using the final parameter state
    returned by EM.

    this is intentionally separate from the final EM update because the
    smoother_result generated inside an EM iteration corresponds to the
    parameters BEFORE the subsequent proximal parameter update.
    """

    _, smoother_result = rts_smoother_jax(y, F, R,  em_state)

    final_ll = -smoother_result.negative_log_likelihood

    # Store the likelihood at the iteration corresponding to the
    # final parameter state.
    safe_iter = jnp.minimum(em_state.em_iter, len(em_state.log_likelihood) - 1)

    updated_ll_trajectory = (
        em_state.log_likelihood
        .at[safe_iter]
        .set(final_ll)
    )

    em_state = dataclasses.replace(
        em_state,
        log_likelihood=updated_ll_trajectory,
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