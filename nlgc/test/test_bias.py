from nlgc.test.test_gc import make_gc_test_setup
from nlgc.opt.em import solve_params, _copycast_em_state_numpy
from nlgc.opt.kalman.filter import _copycast_rtssmoother_result_numpy
from nlgc.bias_utils import compute_bias, bias_by_idx, sample_path_bias
import numpy as np


def test_sample_path_bias_total_sparsity():
    n_sources = 4
    n_sensors = 10
    n_orients = 3
    order = 2
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    em_state.A_mask = np.zeros_like(em_state.A_mask)

    bias = compute_bias(em_state, smoother_result, config)

    assert np.isfinite(bias)
    assert np.isclose(bias, 0.0, atol=1e-10)
    print(f"{bias=:.5f}")


def test_sample_path_bias_zero_uncertainty():
    n_sources = 4
    n_sensors = 10
    n_orients = 3
    order = 2
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    m = em_state.N_sources_upper

    zero_P = np.zeros_like(smoother_result.smoothed_cov)
    zero_B = np.zeros_like(smoother_result.smoother_gain)

    bias = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        zero_P,
        zero_B,
        em_state.A_mask[:m],
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    assert np.isfinite(bias)
    assert bias >= 0.0
    print(f"{bias=:.5f}")


def test_sample_path_bias_high_order():
    n_sources = 4
    n_sensors = 10
    n_orients = 3
    order = 5
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    m = em_state.N_sources_upper
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    N = m // eff_eigenmodes

    rng = np.random.default_rng(0)

    A_mask = np.ones_like(em_state.A_mask[:m])

    # randomly remove entire source->target blocks across all lags
    for target in range(N):
        target_slice = slice(
            target * eff_eigenmodes,
            (target + 1) * eff_eigenmodes,
        )

        for source in range(N):
            if rng.random() < 0.5:
                for lag in range(order):
                    source_slice = slice(
                        lag * m + source * eff_eigenmodes,
                        lag * m + (source + 1) * eff_eigenmodes,
                    )
                    A_mask[target_slice, source_slice] = 0

    bias = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        A_mask,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    assert np.isfinite(bias)
    assert bias >= 0.0
    print(f"{bias=:.5f}")


def test_sample_path_bias_ill_conditioned_q_blocks():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    em_state = _copycast_em_state_numpy(em_state)

    m = em_state.N_sources_upper
    
    Q = em_state.Q[:m, :m].copy()

    # nearly-singular covariance for every orientation block
    Q_block = np.array([
        [1.0,    0.9999, 0.9999],
        [0.9999, 1.0,    0.9999],
        [0.9999, 0.9999, 1.0   ],
    ]) + np.eye(3) * 1e-8

    for i in range(0, m, n_orients):
        Q[i:i+n_orients, i:i+n_orients] = Q_block

    bias = sample_path_bias(
        Q,
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        em_state.A_mask[:m],
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    assert np.isfinite(bias)
    assert bias >= 0.0


def test_sample_path_bias_scaling_q():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    m = em_state.N_sources_upper

    bias1 = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        em_state.A_mask[:m],
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    bias2 = sample_path_bias(
        2.0 * em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        em_state.A_mask[:m],
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    bias5 = sample_path_bias(
        5.0 * em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        em_state.A_mask[:m],
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    assert np.allclose(bias2, bias1 / 2, rtol=1e-8)
    assert np.allclose(bias5, bias1 / 5, rtol=1e-8)


def test_sample_path_bias_monotonic_masking():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 4000
    lambda_ = 0.1
    sparsity = 0.05

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)

    m = em_state.N_sources_upper

    # construct three masks at increasing levels of sparsity
    mask_full = np.ones_like(em_state.A_mask[:m])

    rng = np.random.default_rng(0)
    mask_half = mask_full.copy()

    eff = config.latent.n_eigenmodes * config.latent.n_orients
    N = m // eff

    for target in range(N):
        for source in range(N):
            if rng.random() < 0.5:
                for lag in range(order):
                    rs = slice(target * eff, (target + 1) * eff)
                    cs = slice(lag * m + source * eff,
                               lag * m + (source + 1) * eff)
                    mask_half[rs, cs] = 0

    mask_zero = np.zeros_like(mask_full)

    bias_full = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        mask_full,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    bias_half = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        mask_half,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    bias_zero = sample_path_bias(
        em_state.Q[:m, :m],
        em_state.A[:m],
        smoother_result.smoothed_state,
        smoother_result.smoothed_cov,
        smoother_result.smoother_gain,
        mask_zero,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    assert bias_full >= bias_half >= bias_zero


def test_sample_path_bias_additivity():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 1000
    lambda_ = 0.1
    sparsity = 0.1

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)
    
    em_state = _copycast_em_state_numpy(em_state)
    smoother_result = _copycast_rtssmoother_result_numpy(smoother_result)

    m = em_state.N_sources_upper

    Q = em_state.Q[:m, :m]
    A = em_state.A[:m]
    A_mask = em_state.A_mask[:m]

    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    bias_total = sample_path_bias(
        Q,
        A,
        x,
        P,
        B,
        A_mask,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        config.latent.order,
    )

    n_voxels = m // config.latent.n_orients

    bias_sum = 0.0

    # construct masks that keep only one voxel's parameters
    for idx_v in range(n_voxels):

        voxel_mask = np.zeros_like(A_mask)

        # target voxel rows
        row_start = idx_v * config.latent.n_orients
        row_stop = row_start + config.latent.n_orients

        voxel_mask[row_start:row_stop] = A_mask[row_start:row_stop]

        bias_v = sample_path_bias(
            Q,
            A,
            x,
            P,
            B,
            voxel_mask,
            config.latent.n_eigenmodes,
            config.latent.n_orients,
            m,
            config.latent.order,
        )

        bias_sum += bias_v

    assert np.isfinite(bias_total)
    assert np.isfinite(bias_sum)

    assert np.allclose(
        bias_total,
        bias_sum,
        rtol=1e-6,
        atol=1e-8,
    ), (
        f"Bias not additive: total={bias_total}, "
        f"sum of voxel contributions={bias_sum}"
    )


def test_bias_by_idx_finite():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 1000
    lambda_ = 0.1
    sparsity = 0.1

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)
    
    em_state = _copycast_em_state_numpy(em_state)
    smoother_result = _copycast_rtssmoother_result_numpy(smoother_result)

    m = em_state.N_sources_upper
    p = config.latent.order

    Q = em_state.Q[:m, :m]
    A = em_state.A[:m]
    A_mask = em_state.A_mask[:m]

    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    n_voxels = m // config.latent.n_orients

    for targ in range(n_voxels):
        for src in range(n_voxels):
            bias_edge = bias_by_idx(
                src,
                targ,
                Q,
                A,
                x,
                P,
                B,
                A_mask,
                config.latent.n_eigenmodes,
                config.latent.n_orients,
                m,
                p,
            )

            assert np.isfinite(bias_edge), f"bias for {src}->{targ} is not finite"
            assert bias_edge >= 0.0, f"bias for {src}->{targ} is negative: {bias_edge}"

            print(f"bias for {src}->{targ} = {bias_edge}")


def test_bias_by_idx_masked_edge_is_zero():
    n_sources = 4
    n_orients = 3
    order = 2
    n_sensors = 10
    T = 1000
    lambda_ = 0.1
    sparsity = 0.1

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)
    
    em_state = _copycast_em_state_numpy(em_state)
    smoother_result = _copycast_rtssmoother_result_numpy(smoother_result)

    m = em_state.N_sources_upper
    p = config.latent.order
    n_orients = config.latent.n_orients

    Q = em_state.Q[:m, :m]
    A = em_state.A[:m]
    A_mask = em_state.A_mask[:m].copy()

    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    # Select an arbitrary edge to completely mask out
    test_src = 1
    test_targ = 2
    
    targ_start = test_targ * n_orients
    targ_end = (test_targ + 1) * n_orients

    # Zero out all lags for this specific source in the target rows
    for lag in range(p):
        src_start = test_src * n_orients + lag * m
        src_end = (test_src + 1) * n_orients + lag * m
        A_mask[targ_start:targ_end, src_start:src_end] = 0

    bias_edge = bias_by_idx(
        test_src,
        test_targ,
        Q,
        A,
        x,
        P,
        B,
        A_mask,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        p,
    )

    assert bias_edge == 0.0, f"Masked edge bias should be exactly 0.0, got {bias_edge}"


def test_bias_by_idx_single_source_equivalence():
    n_sources = 1
    n_orients = 3
    order = 2
    n_sensors = 5 # Reduced sensors for 1 source
    T = 1000
    lambda_ = 0.1
    sparsity = 0.1

    ssm, em_state, config, lambda_ = make_gc_test_setup(
        n_sources=n_sources,
        n_sensors=n_sensors,
        n_orients=n_orients,
        order=order,
        T=T,
        lambda_=lambda_,
        sparsity=sparsity,
    )

    em_state, smoother_result = solve_params(ssm.y, ssm.F, ssm.R, em_state, 
                                             config, lambda_)
    
    em_state = _copycast_em_state_numpy(em_state)
    smoother_result = _copycast_rtssmoother_result_numpy(smoother_result)

    m = em_state.N_sources_upper
    p = config.latent.order

    Q = em_state.Q[:m, :m]
    A = em_state.A[:m]
    A_mask = em_state.A_mask[:m]

    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    # Compute joint full bias
    bias_total = sample_path_bias(
        Q,
        A,
        x,
        P,
        B,
        A_mask,
        config.latent.n_eigenmodes,
        config.latent.n_orients,
        m,
        p,
    )

    # Compute marginal bias of the ONLY edge (0 -> 0)
    bias_edge = bias_by_idx(
        0, 
        0, 
        Q, 
        A, 
        x, 
        P, 
        B, 
        A_mask, 
        config.latent.n_eigenmodes, 
        config.latent.n_orients, 
        m, 
        p
    )

    assert np.allclose(
        bias_total,
        bias_edge,
        rtol=1e-6,
        atol=1e-8,
    ), (
        f"For n_sources=1, marginal bias must equal joint bias: "
        f"total={bias_total}, edge={bias_edge}"
    )


if __name__ == '__main__':
    print("running test sample path bias total sparsity")
    test_sample_path_bias_total_sparsity()
    print("pass\n\n")

    print("running test sample path bias zero state uncertainty")
    test_sample_path_bias_zero_uncertainty()
    print("pass\n\n")

    print("running test sample path bias high order")
    test_sample_path_bias_high_order()
    print("pass\n\n")

    print("running test sample path bias ill conditioning")
    test_sample_path_bias_ill_conditioned_q_blocks()
    print("pass\n\n")

    print("running test sample path bias Q scaling")
    test_sample_path_bias_scaling_q()
    print("pass\n\n")

    print("running test sample path bias monotonic sparsity increase")
    test_sample_path_bias_monotonic_masking()
    print("pass\n\n")

    print("running test sample path bias additivity")
    test_sample_path_bias_additivity()
    print("pass\n\n")

    print("running test bias by idx finite")
    test_bias_by_idx_finite()
    print("pass\n\n")

    print("running test bias by idx masking")
    test_bias_by_idx_masked_edge_is_zero()
    print("pass\n\n")
    
    print("running test bias by idx vs sample path bias equivalence")
    test_bias_by_idx_single_source_equivalence()
    print("pass\n\n")