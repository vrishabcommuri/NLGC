# NOTE: to run GC link testing across multiple host devices, set
#     XLA_FLAGS=--xla_force_host_platform_device_count=<N>
# in the environment *before* the first jax import (i.e. at the top of your
# driver script, before `import nlgc`). Setting it here is too late -- jax is
# already imported via the nlgc/__init__.py -> nlgc.opt chain.
import numpy as np

from nlgc.utils.restriction import (jax_expand_zeroindex_masks,  
                                    link_tuples_to_zero_indices)
from nlgc.opt.em import em_jax
from nlgc.bias_utils import compute_bias                         
import dataclasses                                               
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
                                         


def batch_em_state(em_state, zeroed_indices):
    K = len(zeroed_indices)
    A_masks = jax_expand_zeroindex_masks(zeroed_indices, em_state)

    batched_state = dataclasses.replace(
        em_state,
        A = jnp.broadcast_to(em_state.A, (K, *em_state.A.shape)),
        Q = jnp.broadcast_to(em_state.Q, (K, *em_state.Q.shape)),
        N_sources_upper = em_state.N_sources_upper, # meta field not batched
        em_iter = jnp.broadcast_to(em_state.em_iter, 
                                        (K,)),
        log_likelihood = jnp.broadcast_to(em_state.log_likelihood, 
                                        (K, *em_state.log_likelihood.shape)),
        P0 = jnp.broadcast_to(em_state.P0, (K, *em_state.P0.shape)),
        N0 = jnp.broadcast_to(em_state.N0, (K, *em_state.N0.shape)),
        A_mask=A_masks,
    )

    batched_state = dataclasses.replace(
        batched_state,
        em_iter = jnp.zeros_like(batched_state.em_iter),
        log_likelihood = jnp.zeros_like(batched_state.log_likelihood),
        P0 = jnp.zeros_like(batched_state.P0), # these depend on previous A
        N0 = jnp.zeros_like(batched_state.N0), # these depend on previous A
    )

    return batched_state


def slice_batched_output(output, idx):
    updates = {}

    for f in dataclasses.fields(output):
        value = getattr(output, f.name)

        if f.metadata.get("static", False):
            updates[f.name] = value
        elif value is None:
            updates[f.name] = None
        else:
            updates[f.name] = value[idx]

    return dataclasses.replace(output, **updates)


def batched_test_links(links_to_check, y, F, R, lambda_, full_em_state, config):
    # gc_extraction routes everything except ModelMultiprocessConfig here, but only
    # ModelShardConfig carries n_devices -- so serial and vmap modes would raise
    # AttributeError. Default to 1, which makes the batch loop below run one link at
    # a time, i.e. the serial behaviour those configs ask for.
    N_devices = getattr(config.parallel, 'n_devices', 1)
    K = len(links_to_check)
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    m = full_em_state.N_sources_upper
    nx = m // (eff_eigenmodes)
    fullmodel_log_likelihood = full_em_state.log_likelihood[full_em_state.em_iter]

    zeroed_indices = link_tuples_to_zero_indices(links_to_check, full_em_state,
                                                 config)

    batched_em = jax.pmap(
        em_jax, 
        in_axes=(None, None, None, 0, None, None, None), 
        # tells pmap that argument index 4 (config) is static
        static_broadcasted_argnums=(4,)    
    )

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    nonconv_flag = np.zeros((nx, nx), dtype=np.bool_)

    for start in range(0, K, N_devices):
        if config.numerical.verbose:
            print(f"running pmapped GC link testing batch #{start//N_devices} "
                  f"on {jax.device_count()} CPUs")
            
        stop = min(start + N_devices, K)

        curr_zeroed = zeroed_indices[start:stop]
        N_valid = len(curr_zeroed)

        if N_valid < N_devices:
            curr_zeroed = (curr_zeroed +
                           [curr_zeroed[-1]] *
                           (N_devices - N_valid))

        curr_state = batch_em_state(full_em_state, curr_zeroed)

        batched_states, batched_smoother_results = \
            batched_em(y, F, R, curr_state, config, lambda_, 
                       config.optimizer.max_iter)

        # may have fewer results in this batch than we have devices since we
        # zero-padded the batch size to align with the number of devices, we
        # just drop the extras
        batched_states = slice_batched_output(batched_states, 
                                              slice(None, N_valid))
        batched_smoother_results = \
            slice_batched_output(batched_smoother_results, slice(None, N_valid))

        if config.numerical.verbose:
            print(f"\nbatch #{start//N_devices} model EM iterations:")

        for batch_idx, (targ, src) in enumerate(links_to_check[start:stop]):
            reduced_em_state = slice_batched_output(batched_states, batch_idx)
            reduced_smoother_result = \
                slice_batched_output(batched_smoother_results, batch_idx)
            
            if config.numerical.verbose:
                print(f"\t model {batch_idx} iters: "
                      f"{reduced_em_state.em_iter}")

            bias = compute_bias(reduced_em_state, reduced_smoother_result, 
                                config)
            
            bias_r[targ, src] = bias
            dev_raw[targ, src] = 2 * fullmodel_log_likelihood
            dev_raw[targ, src] -= 2 * reduced_em_state.log_likelihood[reduced_em_state.em_iter]
            nonconv_flag[targ, src] = reduced_em_state.em_iter == \
                                    config.optimizer.max_iter
        
    return dev_raw, bias_r, nonconv_flag


    
    





