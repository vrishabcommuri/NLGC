import os
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1"
)
os.environ["XLA_FLAGS"] = (
f"--xla_force_host_platform_device_count={4}" # conservative guess
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# from multiprocessing import cpu_count
import numpy as np
import jax
import time
from nlgc.opt.em import (solve_params, em_jax, _copycast_em_state_numpy)
from nlgc.opt.proximal import instantiate_proximal_solvers
from nlgc.parallel_gc import (link_tuples_to_zero_indices, 
                              batch_em_state, slice_batched_output)
from nlgc.config import ModelConfig
from nlgc.test.ssm_gen import gen_sparse_var_ssm
from nlgc.test.test_em import make_initial_em_state
from nlgc.test.viz import (plot_transition_comparison, plot_transition_single, 
                           plot_transition_blurred)
from nlgc.nlgc_utils import gc_extraction
from nlgc.bias_utils import debias_deviances
from nlgc.stat import fdr_control
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

show_plots = False


def make_gc_test_setup(
    n_sources=4,
    n_sensors=8,
    n_orients=3,
    order=3,
    T=2000,
    lambda_=1,
    parallel_mode='shard',
    **ssm_kwargs,
):
    ssm, _, _ = gen_sparse_var_ssm(
        T=T,
        n_sources=n_sources,
        n_sensors=n_sensors,
        order=order,
        n_orients=n_orients,
        seed=0,
        **ssm_kwargs,
    )

    instantiate_proximal_solvers(
        {
            "n_orients": n_orients,
            "order": order,
            "alpha": 0.0,
            "beta": 0.0,
        },
        N_sources=n_sources * n_orients,
    )

    em_state = make_initial_em_state(
        ssm,
        order=order,
        n_orients=n_orients,
    )

    config = ModelConfig.from_legacy_kwargs(
        {
            "order": order,
            "n_orients": n_orients,
            "n_eigenmodes": 1,
            "parallel_mode": parallel_mode,
            "lambda_range": (lambda_,),
            "verbose": True,
            "n_devices": jax.device_count(),
            "n_workers": jax.device_count(),
            "negligible_candidate_link_energy_thr":0.2,
            "tol":1e-5,
            "A_tol":5e3,
        }
    )

    return ssm, em_state, config, lambda_


def _expected_block_mask(
    n_sources,
    n_orients,
    order,
    src,
    dst,
):
    m = n_sources * n_orients

    mask = np.ones((m, m * order), dtype=bool)

    row = slice(dst * n_orients, (dst + 1) * n_orients)

    for lag in range(order):
        col = slice(
            lag * m + src * n_orients,
            lag * m + (src + 1) * n_orients,
        )
        mask[row, col] = False

    return mask


def test_zeroindex_vector_var3():
    n_sources = 4
    n_orients = 3
    order = 3

    _, em_state, config, _ = make_gc_test_setup(n_sources=n_sources,
                                                n_orients=n_orients,
                                                order=order)

    # (target, source) -- zeroes A[ROI 1 rows, ROI 0 cols], i.e. link 0 -> 1
    links = [(1, 0)]

    zeroed_indices = link_tuples_to_zero_indices(links, em_state, config)

    batched = batch_em_state(em_state, zeroed_indices)

    mask = np.array(batched.A_mask[0])

    expected = _expected_block_mask(n_sources, n_orients, order, src=0, dst=1)
    mask_upper = mask[:expected.shape[0]]

    assert np.array_equal(mask_upper, expected)

    if show_plots:
        plot_transition_comparison(expected.astype(int), mask_upper.astype(int),  
                                   titles=(
                                        "True Mask",
                                        "Constructed Mask from Tuple",
                                    ))
        plt.show()


def test_zeroindex_multiple_links():
    n_sources = 4
    n_orients = 3
    order = 2

    _, em_state, config, _ = make_gc_test_setup(n_sources=n_sources, 
                                                  n_orients=n_orients,
                                                  order=order)

    # (target, source): links 0 -> 1 and 2 -> 0
    links = [(1, 0), (0, 2)]

    zeroed = link_tuples_to_zero_indices(links, em_state, config)

    batched = batch_em_state(em_state, zeroed)

    mask1 = np.array(batched.A_mask[0])
    mask2 = np.array(batched.A_mask[1])
    mask = mask1.astype(bool) & mask2.astype(bool)

    expected1 = _expected_block_mask(n_sources, n_orients, order, src=0, dst=1)


    expected2 = _expected_block_mask(n_sources, n_orients, order, 
                                               src=2, dst=0)
    
    expected = expected1 & expected2
    mask_upper = mask[:expected.shape[0]]

    assert np.array_equal(mask_upper, expected)

    if show_plots:
        plot_transition_comparison(expected.astype(int), mask_upper.astype(int),  
                                   titles=(
                                        "True Mask",
                                        "Constructed Mask from Tuple",
                                    ))
        plt.show()


def test_batch_shapes():
    ssm, em_state, config, _ = make_gc_test_setup()

    links = [(0, 1), (1, 2), (2, 0)]

    zeroed = link_tuples_to_zero_indices(links, em_state, config)

    batched = batch_em_state(em_state, zeroed)

    K = len(links)

    assert batched.A.shape[0] == K
    assert batched.Q.shape[0] == K
    assert batched.A_mask.shape[0] == K

    for k in range(K):
        assert np.allclose(
            np.array(batched.A[k]),
            np.array(em_state.A),
        )
        assert np.allclose(
            np.array(batched.Q[k]),
            np.array(em_state.Q),
        )


def test_masks_differ_between_links():
    ssm, em_state, config, _ = make_gc_test_setup(order=2)

    links = [(0, 1), (1, 2), (2, 0)]

    zeroed = link_tuples_to_zero_indices(links, em_state, config)

    batched = batch_em_state(em_state, zeroed)

    masks = np.array(batched.A_mask)

    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])
    assert not np.array_equal(masks[0], masks[2])


def test_reduced_model_enforces_zero_block():
    n_sources = 5
    n_orients = 3
    order = 2
    n_sensors = 20
    T = 3000
    lambda_ = 0.1
    sparsity = 0.075

    ssm, em_state, config, lambda_ = make_gc_test_setup(n_sources=n_sources,
                                                        n_sensors=n_sensors,
                                                        n_orients=n_orients,
                                                        order=order,
                                                        T=T,
                                                        lambda_=lambda_,
                                                        sparsity=sparsity)

    # (target, source): link 0 -> 4, so A[ROI 4 rows, ROI 0 cols] must be zero
    links = [(4, 0)]

    zeroed_indices = link_tuples_to_zero_indices(links, em_state, config)

    reduced_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config,
                                    lambda_, zeroed_index=zeroed_indices)

    reduced_A = np.array(reduced_state.A)
    full_A = ssm.A
    m = ssm.N_sources

    if show_plots:
        plot_transition_comparison(full_A[:m], reduced_A[:m], 
                                titles=("Full A", "Reduced A"))
        plt.show()
    
    assert np.allclose(reduced_A[m-n_orients:m, :n_orients], 0.0, atol=1e-6)


def test_batched_matches_sequential():
    n_sources = 5
    n_orients = 3
    order = 2
    n_sensors = 20
    T = 3000
    lambda_ = 0.1
    sparsity = 0.075

    ssm, em_state, config, lambda_ = make_gc_test_setup(n_sources=n_sources,
                                                        n_sensors=n_sensors,
                                                        n_orients=n_orients,
                                                        order=order,
                                                        T=T,
                                                        lambda_=lambda_,
                                                        sparsity=sparsity)

    links_to_check = [(0, 4), (1, 2), (2, 0)]

    zeroed_indices = link_tuples_to_zero_indices(links_to_check, em_state, 
                                                 config)

    sequential = []

    full_em_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config, 
                                    lambda_, zeroed_index=None)

    for idx, zi in enumerate(zeroed_indices):
        print("fitting sequential reduced model link "
              f"{links_to_check[idx][0]}->{links_to_check[idx][1]}")
        curr_reduced_em_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                                config, lambda_, 
                                                zeroed_index=[zi])
        
        sequential.append(curr_reduced_em_state)

    batched_state = batch_em_state(
        full_em_state,
        zeroed_indices,
    )

    print("fitting pmapped reduced models")
    batched_em = jax.pmap(
        em_jax,
        in_axes=(None, None, None, 0, None, None, None),
        static_broadcasted_argnums=(4,),
    )

    batched_out, _ = batched_em(ssm.y, ssm.F, ssm.R, batched_state, config, 
                                lambda_, config.optimizer.max_iter)
    
    for k in range(len(links_to_check)):
        print(f"reduced model {k} ran for {batched_out.em_iter[k]} iterations")

    m = ssm.N_sources

    for k in range(len(links_to_check)):
        if show_plots:
            plot_transition_comparison(batched_out.A[k][:m], 
                                       sequential[k].A[:m], 
                    titles=(f"Batched A[{k}] (zeroed {links_to_check[k]})", 
                            f"Sequential A[{k}] (zeroed {links_to_check[k]})"))
            plt.show()

            plot_transition_single(batched_out.A[k][:m] - sequential[k].A[:m], 
                    titles=("diff"))
            plt.show()
            
        assert np.allclose(
            np.array(batched_out.A[k]),
            np.array(sequential[k].A),
            atol=5e-2,
            rtol=5e-2,
        )

        assert np.allclose(
            np.array(batched_out.Q[k]),
            np.array(sequential[k].Q),
            atol=5e-2,
            rtol=5e-2,
        )

        # Compare the CONVERGED log-likelihood, not the trajectory. The two arms
        # cannot share a trajectory by construction: `sequential` starts cold
        # from the initial em_state and runs em_blas warmup + em_jax, while
        # `batched` starts from the fitted full model and runs em_jax only. They
        # begin at different likelihoods (~ -9e3 vs ~ +4.6e4) and reach the same
        # optimum after a different number of iterations, which is exactly what
        # the A/Q assertions above already establish.
        ll_batched = float(batched_out.log_likelihood[k][batched_out.em_iter[k]])
        ll_sequential = float(sequential[k].log_likelihood[sequential[k].em_iter])

        assert np.allclose(ll_batched, ll_sequential, atol=5e-2, rtol=5e-2), \
            f"link {links_to_check[k]}: batched converged to {ll_batched}, " \
            f"sequential to {ll_sequential}"


def test_full_ll_exceeds_reduced():
    n_sources = 5
    n_orients = 3
    order = 2
    n_sensors = 20
    T = 3000
    lambda_ = 0.1
    sparsity = 0.075

    ssm, em_state, config, lambda_ = make_gc_test_setup(n_sources=n_sources,
                                                        n_sensors=n_sensors,
                                                        n_orients=n_orients,
                                                        order=order,
                                                        T=T,
                                                        lambda_=lambda_,
                                                        sparsity=sparsity)

    full_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config, 
                                 lambda_, zeroed_index=None)

    links = [(0, 4)]

    zeroed_index = link_tuples_to_zero_indices(links, full_state, config)
    full_state = _copycast_em_state_numpy(full_state)

    reduced_state, _ = solve_params(ssm.y, ssm.F, ssm.R, full_state, config, 
                                    lambda_, zeroed_index=zeroed_index)

    ll_full = float(full_state.log_likelihood[-1])
    ll_reduced = float(reduced_state.log_likelihood[-1])

    assert ll_full >= ll_reduced - 1e-4

    print(f"Full LL: {ll_full:.3f} > Reduced LL: {ll_reduced:.3f}")


def test_pmap_benchmark():
    n_sources = 5
    n_orients = 3
    order = 2
    n_sensors = 20
    T = 3000
    lambda_ = 0.1
    sparsity = 0.075
    K = 20
    print(f"jax device count: {jax.device_count()}")

    ssm, em_state, config, lambda_ = make_gc_test_setup(n_sources=n_sources,
                                                        n_sensors=n_sensors,
                                                        n_orients=n_orients,
                                                        order=order,
                                                        T=T,
                                                        lambda_=lambda_,
                                                        sparsity=sparsity)

    full_em_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config,
                                    lambda_, zeroed_index=None)
    full_em_state = _copycast_em_state_numpy(full_em_state)

    rng = np.random.default_rng(0)

    links_to_check = []
    while len(links_to_check) < K:
        src = rng.integers(n_sources)
        dst = rng.integers(n_sources)

        if src != dst:
            links_to_check.append((src, dst))

    zeroed_indices = link_tuples_to_zero_indices(links_to_check,
                                                 full_em_state,
                                                 config)

    # ------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------

    sequential = []

    t0 = time.perf_counter()

    for idx, zi in enumerate(zeroed_indices):
        print(f"fit sequential {links_to_check[idx][0]}->"
              f"{links_to_check[idx][1]}")
        reduced_state, _ = solve_params(ssm.y, ssm.F, ssm.R, full_em_state,
                                        config, lambda_, zeroed_index=[zi])
        sequential.append(reduced_state)

    t_seq = time.perf_counter() - t0

    # ------------------------------------------------------------
    # pmap
    # ------------------------------------------------------------

    n_devices = jax.device_count()
    print(f"running pmap with {n_devices=}")

    batched_em = jax.pmap(
        em_jax,
        in_axes=(None, None, None, 0, None, None, None),
        static_broadcasted_argnums=(4,)
    )

    print("fit pmap dry run")

    for start in range(0, K, n_devices):
        stop = min(start + n_devices, K)

        curr_zeroed = zeroed_indices[start:stop]
        
        # pad input
        if len(curr_zeroed) < n_devices:
            curr_zeroed = (curr_zeroed +
                           [curr_zeroed[-1]] *
                           (n_devices - len(curr_zeroed)))

        curr_state = batch_em_state(full_em_state, curr_zeroed)

        batched_out, _ = batched_em(ssm.y, ssm.F, ssm.R,
                                    curr_state, config, lambda_, 
                                    config.optimizer.max_iter)

        jax.block_until_ready(batched_out.A)

    print("fit pmap final")

    t0 = time.perf_counter()

    batched_results = []

    for start in range(0, K, n_devices):
        stop = min(start + n_devices, K)

        curr_zeroed = zeroed_indices[start:stop]
        n_valid = len(curr_zeroed)

        if n_valid < n_devices:
            curr_zeroed = (curr_zeroed +
                           [curr_zeroed[-1]] *
                           (n_devices - n_valid))

        curr_state = batch_em_state(full_em_state, curr_zeroed)

        batched_out, _ = batched_em(ssm.y, ssm.F, ssm.R,
                                    curr_state, config, lambda_,
                                    config.optimizer.max_iter)

        jax.block_until_ready(batched_out.A)

        batched_results.append(slice_batched_output(batched_out, 
                                                    slice(None, n_valid)))
        

    t_batch = time.perf_counter() - t0

    print(f"\nReduced models : {K}")
    print(f"Sequential time: {t_seq:.2f} s")
    print(f"Batched time   : {t_batch:.2f} s")
    print(f"Speedup        : {t_seq / t_batch:.2f}x")


def test_gc_extraction(parallel_mode="shard"):
    n_sources=50 
    n_sensors=100
    n_orients=3
    order=2
    T=5000
    lambda_=0.2
    sparsity=0.01
    print(f"jax device count: {jax.device_count()}")

    ssm, em_state, config, _ = make_gc_test_setup(n_sources=n_sources,
                                                  n_sensors=n_sensors,
                                                  n_orients=n_orients,
                                                  order=order,
                                                  T=T,
                                                  lambda_=lambda_,
                                                  sparsity=sparsity,
                                                  parallel_mode=parallel_mode)
    
    plot_transition_blurred(ssm.A, em_state.N_sources_upper, 2)
    plt.show()

    ROIs = list(range(n_sources))
    dev_raw, bias_r, bias_f, model_f, nonconv_flag = \
        gc_extraction(ssm.y, ssm.F, ssm.R, ROIs, em_state, config)
    
    avg_debiased_dev = debias_deviances(dev_raw, bias_f, bias_r)

    n_eigenmodes = 1
    eff_eigenmodes = n_orients * n_eigenmodes
    alpha = 0.1

    J = fdr_control(avg_debiased_dev, order * (eff_eigenmodes**2), alpha)

    if show_plots:
        m = em_state.N_sources_upper
        plot_transition_comparison(ssm.A[:m], 
                                   model_f._ravel_a(model_f._parameters[0]), 
                titles=("Ground Truth A", 
                        "Estimated A"), 
                        bind_colorbars=True)
        plt.show()

        plot_transition_comparison(ssm.A[:m], 
                                  (np.abs(model_f._ravel_a\
                                  (model_f._parameters[0])) > 1e-4).astype(int), 
                titles=("Ground Truth A", 
                        "Nonzero A"), 
                        bind_colorbars=False)
        plt.show()
        plot_transition_comparison(ssm.A[:m], 
                                   J, 
                titles=("Ground Truth A", 
                        "J"), 
                        bind_colorbars=False)
        plt.show()

if __name__ == '__main__':
    show_plots = True

    # print("running test single zeroed index") 
    # # test_zeroindex_vector_var3()
    # print("pass\n\n")

    # print("running test multiple zeroed index") 
    # test_zeroindex_multiple_links()
    # print("pass\n\n")

    # print("running test pmap batch mask shapes") 
    # test_batch_shapes()
    # print("pass\n\n")

    # print("!!! TESTS FROM THIS POINT RELY ON SPECIFIC LINK LOCATIONS IN "
    #       "TRANSITION MATRIX !!!\n"
    #       "(you may have to change the links under test if your transition "
    #       "matrix differs due to version/seeding differences)\n\n")

    # print("running test ensure unique mask per gc test") 
    # test_masks_differ_between_links()
    # print("pass\n\n")

    # print("running test reduced model enforces zeroed block") 
    # test_reduced_model_enforces_zero_block()
    # print("pass\n\n")

    # print("running test reduced model pmap vs sequential fit equality") 
    # test_batched_matches_sequential()
    # print("pass\n\n")

    # print("running test reduced model lowers log likelihood") 
    # test_full_ll_exceeds_reduced()
    # print("pass\n\n")

    # print("running test benchmark pmap vs sequential EM") 
    # test_pmap_benchmark()
    # print("done\n\n")

    print("running test pmapped gc_extraction") 
    test_gc_extraction()
    print("done\n\n")

    # print("running test multiprocess gc_extraction") 
    # test_gc_extraction(parallel_mode="multiprocess")
    # print("done\n\n")