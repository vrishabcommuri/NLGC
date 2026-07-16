import numpy as np
from numpy.typing import NDArray
from typing import Union, TypeAlias
import dataclasses
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from nlgc.opt.kalman.filter import (rts_smoother_blas, rts_smoother_jax)
from nlgc.opt.proximal import proximal_param_update
from nlgc.utils.restriction import expand_zeroindex_masks
from functools import partial

Array: TypeAlias = NDArray[np.float64] | jnp.ndarray

@jax.tree_util.register_dataclass
@dataclass 
class EMState:
    A: Union[Array, None] = None  # companion form
    A_mask: Union[Array, None] = None
    Q: Union[Array, None] = None  # companion form
    em_iter: int = 0  # how many EM steps have we done?
    log_likelihood: float = 0.0
    P0: Union[Array, None] = None
    N0: Union[Array, None] = None

    # companion mutable upper portion; this is marked as meta information
    # so jax can use it at compile time
    N_sources_upper: Union[int, None] = field(default=None,
                                              metadata={"static": True})  
    

def _triage_em_state(em_state):
    assert em_state.A is not None
    assert em_state.A_mask is not None
    assert em_state.Q is not None
    assert em_state.P0 is not None
    assert em_state.N0 is not None
    assert em_state.N_sources_upper is not None


def _copycast_em_state_numpy(em_state):
    return dataclasses.replace(
            em_state,
            A = np.array(em_state.A),
            A_mask = np.array(em_state.A_mask),
            Q = np.array(em_state.Q),
            P0 = np.array(em_state.P0),
            N0 = np.array(em_state.N0),
    )

def _copycast_em_state_jax(em_state):
    return dataclasses.replace(
            em_state,
            A = jnp.array(em_state.A),
            A_mask = jnp.array(em_state.A_mask),
            Q = jnp.array(em_state.Q),
            P0 = jnp.array(em_state.P0),
            N0 = jnp.array(em_state.N0),
    )


def solve_params(y, F, R, em_state, config, lambda_, zeroed_index=None):
    """
    top-level function to set up filters and optimizers
    """
    if zeroed_index is not None:
        A_mask = zeroed_index_to_mask(zeroed_index, em_state)
    else:
        A_mask = np.ones_like(em_state.A)
    
    em_state.A_mask = A_mask

    _triage_em_state(em_state)

    em_state, smoother_result = em_blas(y, F, R, em_state, config, lambda_)

    em_state = _copycast_em_state_jax(em_state)
    y = jnp.array(y)
    F = jnp.array(F)
    R = jnp.array(R)

    em_state, smoother_result = em_jax(y, F, R, em_state, config, lambda_)

    return em_state, smoother_result


def em_blas(y, F, R, em_state, config, lambda_):
    """
    before passing to jax we do a few iterations with the slow but robust 
    solvers. these use heavy iterative decompositions (e.g., scipy solve dare)
    that are not available in jax. after burn-in, we can move to newton-raphson
    acceleration and jit
    """

    for blas_iter in range(config.optimizer.n_warmup_iter):
        smoother_result = rts_smoother_blas(y, F, R, em_state, 
                                            config.numerical.use_lapack)
        
        # print(smoother_result)
        # from nlgc.test.viz import plot_transition_single
        # import matplotlib.pyplot as plt

        # print(config.latent.n_orients)
        # print(em_state.N_sources_upper)
        # plot_transition_single(em_state.Q)
        # plt.show()


        em_state = proximal_param_update(em_state, smoother_result, lambda_)
        
        # print(em_state.Q)
        # print()
        # print(em_state.A)
        # plot_transition_single(em_state.Q)
        # plt.show()

        # plot_transition_single(em_state.A[em_state.N_sources_upper:])
        # plt.show()

        em_state = _copycast_em_state_numpy(em_state)

        em_state.A = zero_entries(em_state.A, em_state.A_mask)

        # em_state.Q = project_psd(em_state.Q)
        
        em_state.em_iter = blas_iter
        em_state.log_likelihood = -smoother_result.negative_log_likelihood

    return em_state, smoother_result


@partial(jax.jit, static_argnames=("config",))
def em_jax(y, F, R, em_state, config, lambda_):
    """
    jax EM which assumes all routines called are JAX-compatible.
    """

    def em_step(iter, carry):
        em_state, _ = carry

        smoother_result = rts_smoother_jax(y, F, R, em_state)

        em_state = proximal_param_update(em_state, smoother_result, lambda_)
        # em_state.Q = project_psd(em_state.Q)

        em_state = dataclasses.replace(
            em_state,
            A=zero_entries(em_state.A, em_state.A_mask),
            em_iter=iter,
            log_likelihood=-smoother_result.negative_log_likelihood,
        )

        return em_state, smoother_result
        
    smoother_result = rts_smoother_jax(y, F, R, em_state)

    # dummy initialization; overwritten on the first iteration.
    init_carry = (em_state, smoother_result)

    em_state, smoother_result = jax.lax.fori_loop(
        lower=0,
        upper=config.optimizer.max_iter,
        body_fun=em_step,
        init_val=init_carry,
    )

    return em_state, smoother_result


def zero_entries(A, A_mask):
    A *= A_mask
    return A 


def zeroed_index_to_mask(zeroed_index, em_state):
    assert len(zeroed_index) == 1
    return expand_zeroindex_masks(zeroed_index, em_state)[0]


def project_psd(Q, eps=1e-8):
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = jnp.linalg.eigh(Q)
    eigvals = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T