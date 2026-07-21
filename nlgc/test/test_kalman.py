from nlgc.opt.kalman.filter import (forward_filter_jax, forward_filter_blas,
                                rts_smoother_jax, rts_smoother_blas)
from nlgc.opt.em import EMState
from nlgc.test.test_em import make_initial_em_state
from nlgc.test.ssm_gen import gen_small_ssm, gen_large_ssm, gen_sparse_var_ssm
import numpy as np
import dataclasses
import jax
import jax.numpy as jnp
import time 




def test_forward_filter_smoke():
    """
    baseline test to ensure that the filter shapes and result vectors are of
    correct dimensions and that the filter is not blowing up
    """
    
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)

    assert result.filtered_state.shape == (ssm.N_times, ssm.N_sources)
    assert result.predicted_state.shape == (ssm.N_times, ssm.N_sources)
    assert result.filtered_cov.shape == (ssm.N_sources, ssm.N_sources)
    assert result.predicted_cov.shape == (ssm.N_sources, ssm.N_sources)

    assert np.isfinite(result.negative_log_likelihood)

    print(f"smoke test: neg log likelihood: {result.negative_log_likelihood}")


def test_forward_filter_estimation_improvement():
    """
    baseline test to ensure that the forward filter actually improves the state
    estimation
    """
    
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)
    pred_rmse = np.sqrt(np.mean((result.predicted_state - ssm.x) ** 2))
    filt_rmse = np.sqrt(np.mean((result.filtered_state - ssm.x) ** 2))
    print(f"{pred_rmse=:.3f} vs {filt_rmse=:.3f}")

    assert filt_rmse < pred_rmse


def test_forward_filter_kalman_gain():
    """
    compare nonparametric innovation covariance to the innovation covariance 
    from state covariance matrix. if not close, the gain is wrong
    """
    
    ssm = gen_small_ssm()
    
    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)
    innovation = ssm.y - result.predicted_state @ ssm.F.T
    S_empirical = np.cov(innovation.T)
    S_theory = ssm.F @ result.predicted_cov @ ssm.F.T + ssm.R

    assert np.allclose(S_empirical, S_theory, atol=0.1, rtol=0.1)
    

def test_model_mismatch_likelihood():
    """
    compare log likelihoods from ground truth and mismatched models
    """
    ssm = gen_small_ssm()
    
    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    true_ll = -forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)\
                .negative_log_likelihood

    wrongA = 0.2 * np.eye(ssm.N_sources)
    wrong_em_state = dataclasses.replace(
        em_state,
        A = wrongA,
    )

    wrong_ll = -forward_filter_blas(ssm.y, ssm.F, ssm.R, wrong_em_state)\
                .negative_log_likelihood
    
    assert true_ll > wrong_ll

    print(f"{true_ll=:.2f} vs {wrong_ll=:.2f}")


def test_jax_filter_equivalence():
    """
    compare state estimates from jax kalman filter implementation against the 
    more robust blas implementation
    """
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    blas_result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)
    
    em_state = EMState(
        A = jnp.array(ssm.A),
        Q = jnp.array(ssm.Q),
        P0 = jnp.zeros_like(blas_result.filtered_cov),
        N0 = jnp.zeros_like(blas_result.filtered_cov),
    )

    jax_result = forward_filter_jax(
        jnp.array(ssm.y), 
        jnp.array(ssm.F), 
        jnp.array(ssm.R),
        em_state,
    )

    np.testing.assert_allclose(
        blas_result.filtered_state,
        np.asarray(jax_result.filtered_state),
        atol=1e-5,
    )
    
    np.testing.assert_allclose(
        blas_result.filtered_cov,
        np.asarray(jax_result.filtered_cov),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        blas_result.predicted_state,
        np.asarray(jax_result.predicted_state),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        blas_result.predicted_cov,
        np.asarray(jax_result.predicted_cov),
        atol=1e-5,
    )


def test_smoother_improves_state_estimate():
    """
    compare latent-state RMSE from the forward filter and RTS smoother.
    the smoother should never perform worse because it uses future observations.
    """
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    filter_result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)

    smoother_result = rts_smoother_blas(ssm.y, ssm.F, ssm.R, em_state)

    filter_rmse = np.sqrt(
        np.mean((filter_result.filtered_state - ssm.x) ** 2)
    )

    smoother_rmse = np.sqrt(
        np.mean((smoother_result.smoothed_state - ssm.x) ** 2)
    )

    assert smoother_rmse <= filter_rmse

    print(f"{filter_rmse=:.4f} >= {smoother_rmse=:.4f}")


def test_smoother_preserves_final_state():
    """
    The final smoothed state should equal the final filtered state since there
    are no future observations after the last time point.
    """
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    filter_result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)

    smoother_result = rts_smoother_blas(ssm.y, ssm.F, ssm.R, em_state)

    np.testing.assert_allclose(
        smoother_result.smoothed_state[-1],
        filter_result.filtered_state[-1],
        atol=1e-12,
    )


def test_jax_smoother_equivalence():
    """
    compare state estimates from jax rts smoother implementation against the 
    more robust blas implementation
    """
    ssm = gen_small_ssm()

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    blas_result = rts_smoother_blas(ssm.y, ssm.F, ssm.R, em_state)

    em_state = EMState(
        A = jnp.array(ssm.A),
        Q = jnp.array(ssm.Q),
        P0 = jnp.zeros_like(blas_result.smoothed_cov),
        N0 = jnp.zeros_like(blas_result.smoothed_cov),
    )

    jax_result = rts_smoother_jax(
        jnp.array(ssm.y), 
        jnp.array(ssm.F), 
        jnp.array(ssm.R),
        em_state,
    )

    np.testing.assert_allclose(
        blas_result.smoothed_state,
        np.asarray(jax_result.smoothed_state),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        blas_result.smoothed_cov,
        np.asarray(jax_result.smoothed_cov),
        atol=1e-5,
    )


def benchmark_filter_speed(ssm, n_repeats=100):
    """
    compare repeated filter evaluations after JAX compilation.
    """

    em_state = EMState(
        A = ssm.A,
        Q = ssm.Q,
    )

    # -----------------------------
    # blas timing
    # -----------------------------
    start = time.perf_counter()

    for _ in range(n_repeats):
        blas_result = forward_filter_blas(ssm.y, ssm.F, ssm.R, em_state)

    blas_time = time.perf_counter() - start


    # -----------------------------
    # jax setup
    # -----------------------------

    em_state = EMState(
        A = jnp.array(ssm.A),
        Q = jnp.array(ssm.Q),
        P0 = np.zeros_like(blas_result.filtered_cov),
        N0 = jnp.zeros_like(blas_result.filtered_cov),
    )

    args = (
        jnp.asarray(ssm.y),
        jnp.asarray(ssm.F),
        jnp.asarray(ssm.R),
        em_state,
    )

    # need to do one run first otherwise we'll be profiling compile time
    jax_result = forward_filter_jax(*args)

    # block until finished
    jax.block_until_ready(jax_result.filtered_state)

    # -----------------------------
    # JAX timing
    # -----------------------------
    start = time.perf_counter()

    for _ in range(n_repeats):
        jax_result = forward_filter_jax(*args)

        # JAX async execution: force completion
        jax.block_until_ready(jax_result.filtered_state)

    jax_time = time.perf_counter() - start

    # -----------------------------
    # perfunctory checks and readout
    # -----------------------------
    np.testing.assert_allclose(
        blas_result.filtered_state,
        np.asarray(jax_result.filtered_state),
        atol=1e-5,
    )
    
    np.testing.assert_allclose(
        blas_result.filtered_cov,
        np.asarray(jax_result.filtered_cov),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        blas_result.predicted_state,
        np.asarray(jax_result.predicted_state),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        blas_result.predicted_cov,
        np.asarray(jax_result.predicted_cov),
        atol=1e-5,
    )

    print(f"BLAS: {blas_time:.3f}s")
    print(f"JAX:  {jax_time:.3f}s")
    print(f"Speedup: {blas_time/jax_time:.2f}x")


if __name__ == '__main__':
    print("running smoke test")
    test_forward_filter_smoke()
    print("pass\n\n")

    print("running forward estimation improvement test")
    test_forward_filter_estimation_improvement()
    print("pass\n\n")

    print("running forward kalman gain test")
    test_forward_filter_kalman_gain()
    print("pass\n\n")

    print("running model mismatch test")
    test_model_mismatch_likelihood()   
    print("pass\n\n")

    print("running jax vs blas filter equivalence test")
    test_jax_filter_equivalence()   
    print("pass\n\n")

    print("running filter vs smoother comparison test")
    test_smoother_improves_state_estimate()   
    print("pass\n\n")

    print("running filter vs smoother final state equivalence test")
    test_smoother_preserves_final_state()   
    print("pass\n\n")

    print("running jax vs blas smoother equivalence test")
    test_jax_smoother_equivalence()
    print("pass\n\n")

    print("running jax vs blas small state space benchmark")
    ssm = gen_small_ssm()
    benchmark_filter_speed(ssm, n_repeats=100)
    print("done\n\n")

    print("running jax vs blas large state space benchmark")
    ssm = gen_large_ssm()
    benchmark_filter_speed(ssm, n_repeats=10)
    print("done\n\n")
