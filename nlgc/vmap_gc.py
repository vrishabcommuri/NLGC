import jax
import jax.numpy as jnp
import numpy as np
from nlgc.utils.restriction import (jax_expand_zeroindex_masks, 
                                    link_tuples_to_zero_indices)
from nlgc.opt.em import EMState, em_jax
import dataclasses


def batched_test_links(links_to_check, model_f, y, F, R, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    n, m = F.shape
    nx = m // (eff_eigenmodes)
    fullmodel_log_likelihood = em_state.log_likelihood

    lambda_ = model_f.lambda_
    zeroed_indices = link_tuples_to_zero_indices(links_to_check, em_state, 
                                                 config)
    
    K = len(zeroed_indices)
    A_masks = jax_expand_zeroindex_masks(zeroed_indices, em_state)

    batched_state = dataclasses.replace(
        em_state,
        A = jnp.broadcast_to(em_state.A, (K, *em_state.A.shape)),
        Q = jnp.broadcast_to(em_state.Q, (K, *em_state.Q.shape)),
        N_sources_upper = jnp.broadcast_to(em_state.N_sources_upper, 
                                         (K, *em_state.N_sources_upper.shape)),
        smoothed_state = jnp.broadcast_to(em_state.smoothed_state, 
                                        (K, *em_state.smoothed_state.shape)),
        smoothed_cov = jnp.broadcast_to(em_state.smoothed_cov, 
                                        (K, *em_state.smoothed_cov.shape)),
        em_iter = jnp.broadcast_to(em_state.em_iter, 
                                        (K, *em_state.em_iter.shape)),
        log_likelihood = jnp.broadcast_to(em_state.log_likelihood, 
                                        (K, *em_state.log_likelihood.shape)),
        P0 = jnp.broadcast_to(em_state.P0, (K, *em_state.P0.shape)),
        N0 = jnp.broadcast_to(em_state.N0, (K, *em_state.N0.shape)),
        A_mask=A_masks,
    )

    batched_em = jax.vmap(
        em_jax,
        in_axes=(None, None, None, 0, None, None),
    )

    em_states = batched_em(y, F, R, batched_state, config, lambda_)

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    nonconv_flag = np.zeros((nx, nx), dtype=np.bool_)

    for k, (targ, src) in enumerate(links_to_check):
        reduced_em_state = EMState(
            A = em_states.A[k],
            Q = em_states.Q[k],
            N_sources_upper = em_state.N_sources_upper[k],
            smoothed_state = em_state.smoothed_state[k],
            A_mask = em_state.A_mask[k],
            log_likelihood = em_state.log_likelihood[k],
            em_iter = em_state.em_iter[k],
        )

        bias = model_f.compute_bias(reduced_em_state)
        bias_r[targ, src] = bias
        dev_raw[targ, src] = 2 * fullmodel_log_likelihood
        dev_raw[targ, src] -= 2 * reduced_em_state.log_likelihood 
        nonconv_flag[targ, src] = reduced_em_state.em_iter == \
                                  config.optimizer.max_iter
        
    return dev_raw, bias_r, nonconv_flag
    





