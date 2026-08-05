# NOTE: to run GC link testing across multiple host devices, set
#     XLA_FLAGS=--xla_force_host_platform_device_count=<N>
# in the environment *before* the first jax import (i.e. at the top of your
# driver script, before `import nlgc`). Setting it here is too late -- jax is
# already imported via the nlgc/__init__.py -> nlgc.opt chain.
import numpy as np

from nlgc.utils.restriction import link_to_A_mask
from nlgc.opt.em import em_jax, _copycast_em_state_numpy
from nlgc.bias_utils import compute_bias
import dataclasses
import jax
import jax.numpy as jnp
from nlgc.opt import NeuraLVAR, create_shared_mem, link_share_memory
from joblib import Parallel, delayed
from multiprocessing import current_process
from mne.utils import logger
from threadpoolctl import threadpool_limits
jax.config.update("jax_enable_x64", True)
                                         

def batch_em_state(em_state, A_masks):
    K = len(A_masks)

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
    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    eff_eigenmodes = n_eigenmodes * n_orients
    m = full_em_state.N_sources_upper
    nx = m // (eff_eigenmodes)
    fullmodel_log_likelihood = full_em_state.log_likelihood

    A_masks = []
    for targ, src in links_to_check:
        A_masks.append(link_to_A_mask(targ, src))

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

        curr_zeroed = A_masks[start:stop]
        N_valid = len(curr_zeroed)

        if N_valid < N_devices:
            curr_zeroed = (curr_zeroed +
                           [curr_zeroed[-1]] *
                           (N_devices - N_valid))

        curr_state = batch_em_state(full_em_state, curr_zeroed)

        from jax.tree_util import tree_leaves

        for f in dataclasses.fields(curr_state):
            value = getattr(curr_state, f.name)
            print(f"\n{f.name}:")
            for leaf in tree_leaves(value):
                print("   ", getattr(leaf, "shape", None), type(leaf))

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
            dev_raw[targ, src] = 2 * \
                fullmodel_log_likelihood[full_em_state.em_iter]
            dev_raw[targ, src] -= 2 * \
                reduced_em_state.log_likelihood[reduced_em_state.em_iter]
            nonconv_flag[targ, src] = reduced_em_state.em_iter == \
                                    config.optimizer.max_iter
        
    return dev_raw, bias_r, nonconv_flag


def multiprocess_test_links(links_to_check, y, F, R, lambda_, em_state, config):
    em_state = _copycast_em_state_numpy(em_state)
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    m = em_state.N_sources_upper
    nx = m // (eff_eigenmodes)

    fullmodel_log_likelihood = em_state.log_likelihood[em_state.em_iter]

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    nonconv_flag = np.zeros((nx, nx), dtype=np.bool_)

    if len(links_to_check) == 0:
        return dev_raw, bias_r, nonconv_flag

    # Memory management for Parallel implementation
    _, info_y, shm_y = create_shared_mem(y)
    _, info_f, shm_f = create_shared_mem(F)
    shared_bias_r, info_bias_r, shm_bias_r = create_shared_mem(bias_r)
    shared_ll_r, info_ll_r, shm_ll_r = create_shared_mem(dev_raw)
    shared_nonconv_flag, info_nonconv_flag, shm_nonconv_flag = \
        create_shared_mem(nonconv_flag)
    
    shared_args = (info_y, info_f, info_bias_r, info_ll_r, info_nonconv_flag) 
    args = (R, lambda_, em_state, config)  

   
    n_jobs = min(config.parallel.n_workers, len(links_to_check))

    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_learn_reduced_model_parallel)(
            link, *(shared_args + args)
        ) for link in links_to_check
    )

    ll_r = np.reshape(shared_ll_r, dev_raw.shape).copy()
    bias_r = np.reshape(shared_bias_r, dev_raw.shape).copy()
    nonconv_flag = np.reshape(shared_nonconv_flag, nonconv_flag.shape).copy()

    for shm in (shm_nonconv_flag, shm_bias_r, shm_f, shm_ll_r, shm_y):
        shm.close()
        try:
            shm.unlink()
        except Exception as exc:
            print(f"\nUnlink shared-memory issue: {exc}")

    indices = tuple(z for z in zip(*links_to_check))
    dev_raw[indices] = 2 * fullmodel_log_likelihood
    dev_raw[indices] -= 2 * ll_r[indices]

    return dev_raw, bias_r, nonconv_flag


def _learn_reduced_model(targ, src, y, F, R, lambda_f, em_state, config):   
    if config.numerical.verbose: 
        print(f"reduced model {current_process().name} processing {src}->{targ}")
    
    model_r = NeuraLVAR.from_config(config)

    em_state = dataclasses.replace(
        em_state,
        A_mask = link_to_A_mask(targ, src, em_state, config)
    )
    
    em_state, smoother_result = model_r.fit(y, F, R, lambda_f, em_state)

    ll = em_state.log_likelihood[em_state.em_iter]
    if config.numerical.verbose:
        print(f"\t reduced model {current_process().name} iters: "
                f"{em_state.em_iter} ll: {ll}")
    
    bias = compute_bias(em_state, smoother_result, config)

    nonconv_flag = em_state.em_iter == config.optimizer.max_iter
    return ll, bias, nonconv_flag


def _learn_reduced_model_parallel(link_index, info_y, info_f, info_bias_r, 
                                  info_ll_r, info_nonconv_flag, R, lambda_f, 
                                  em_state, config):
    
    # prevent oversubscription
    with threadpool_limits(limits=1, user_api='blas'):
        try:
            y, shm_y = link_share_memory(info_y)
            F, shm_f = link_share_memory(info_f)
            bias_r, shm_bias_r = link_share_memory(info_bias_r)
            ll_r, shm_ll_r = link_share_memory(info_ll_r)
            nonconv_flag, shm_nonconv_flag = link_share_memory(info_nonconv_flag)
        except BaseException as e:
            logger.error("Could not link to memory")
            raise e

        targ, src = link_index
        ll, bias, flag = _learn_reduced_model(targ, src, y, F, R, lambda_f, 
                                              em_state, config)
        ll_r[targ, src] = ll
        bias_r[targ, src] = bias
        nonconv_flag[targ, src] = flag
        for shm in (shm_y, shm_f, shm_bias_r, shm_ll_r, shm_nonconv_flag):
            shm.close()    
    





