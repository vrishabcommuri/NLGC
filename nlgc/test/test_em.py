from nlgc.opt.em import (EMState, em_blas, em_jax, 
                         _copycast_em_state_jax, solve_params)
from nlgc.test.ssm_gen import gen_small_ssm, gen_sparse_var_ssm
from nlgc.test.viz import plot_transition_comparison
from nlgc.opt.proximal import instantiate_proximal_solvers
import numpy as np
from nlgc.config import ModelConfig
import jax.numpy as jnp
import dataclasses
import matplotlib.pyplot as plt
import time
import jax

def make_initial_em_state(ssm, order=1, n_orients=1):
    em_state = EMState(
        A = np.zeros_like(ssm.A),
        A_mask = np.ones_like(ssm.A),
        Q = np.eye(ssm.Q.shape[0]) * 0.001, # initial guess
        P0 = np.zeros_like(ssm.Q),
        N0 = np.zeros_like(ssm.Q),
        N_sources_upper = ssm.Q.shape[0]
    )
    m = ssm.N_sources 

    A = np.block([[np.zeros_like(ssm.A[:m])],
                  [np.eye(N=m*(order-1), M=m*order)]])
    
    Q = np.zeros_like(ssm.Q)
    Q[:m,:m] = np.eye(m)*0.01

    em_state.A = A
    em_state.Q = Q
    em_state.N_sources_upper = m

    return em_state


def test_em_recovers_A():
    ssm = gen_small_ssm(T=500)

    em_state = make_initial_em_state(ssm)

    config = ModelConfig.from_legacy_kwargs({'order':1})
    lambda_ = 0.1

    final_state, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state, config, lambda_)
    
    A_est = final_state.A
    A_true = ssm.A
    # Overall recovery
    np.testing.assert_allclose(
        A_est,
        A_true,
        atol=0.2,
        rtol=0.2,
    )

    # Diagonal entries are usually much better estimated
    np.testing.assert_allclose(
        np.diag(A_est),
        np.diag(A_true),
        atol=0.15,
    )

    # Off-diagonal should remain small
    offdiag_mask = ~np.eye(A_true.shape[0], dtype=bool)

    assert np.linalg.norm(
        A_est[offdiag_mask]
    ) < 0.5

    print("estimate: \n", A_est.round(2))
    print("ground truth: \n", A_true.round(2))


def test_em_recovers_Q():
    """
    EM should recover process noise covariance approximately.
    """

    ssm = gen_small_ssm(T=500)

    config = ModelConfig.from_legacy_kwargs({"order": 1})

    em_state = make_initial_em_state(ssm)

    final_state, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state, config, lambda_=0.0)

    Q_est = final_state.Q
    Q_true = ssm.Q

    # covariance should remain symmetric
    np.testing.assert_allclose(
        Q_est,
        Q_est.T,
        atol=1e-6,
    )

    # diagonal variances are the meaningful part
    np.testing.assert_allclose(
        np.diag(Q_est),
        np.diag(Q_true),
        atol=0.2,
        rtol=0.3,
    )

    # off-diagonal covariance structure
    offdiag_mask = ~np.eye(Q_true.shape[0], dtype=bool)

    assert np.linalg.norm(
        Q_est[offdiag_mask] - Q_true[offdiag_mask]
    ) < 0.5

    print("estimate: \n", Q_est.round(2))
    print("ground truth: \n", Q_true.round(2))


def test_em_jax_matches_blas():
    ssm = gen_small_ssm(T=1000)

    # do 2 blas iterations for jax stability 
    config = ModelConfig.from_legacy_kwargs({"order": 1, 
                                             'n_warmup_iter':2}) 

    em_state_blas = make_initial_em_state(ssm)

    intermediate_state_blas, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state_blas, 
                                  config, lambda_=0.0)
    
    
    # do 25 blas iterations (default) for final
    config = ModelConfig.from_legacy_kwargs({"order": 1})

    em_state_jax = _copycast_em_state_jax(intermediate_state_blas)

    final_state_jax, _ = em_jax(jnp.array(ssm.y), 
                                    jnp.array(ssm.F), 
                                    jnp.array(ssm.R), 
                                    em_state_jax, 
                                    config, lambda_=0.0)
    
    final_state_blas, _ = em_blas(ssm.y, ssm.F, ssm.R, 
                                         intermediate_state_blas, 
                                         config, lambda_=0.0)

    np.testing.assert_allclose(
        final_state_jax.A,
        final_state_blas.A,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        final_state_jax.Q,
        final_state_blas.Q,
        atol=1e-3,
    )


def test_em_likelihood_increases():
    ssm = gen_small_ssm(T=1000)

    em_state = make_initial_em_state(ssm)

    likelihoods = []

    for _ in range(10):
        # force one iteration at a time for evaluation
        config = ModelConfig.from_legacy_kwargs({
            "order": 1,
            "n_warmup_iter": 1,
        })


        em_state, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state, config, 
                              lambda_=0.0)

        likelihoods.append(em_state.log_likelihood)

    likelihoods = np.asarray(likelihoods)
    assert np.all(
        np.diff(likelihoods) >= -1e-5
    )

    for step, likelihood in enumerate(likelihoods):
        print(f"step: {step} ll = {likelihood}")


def test_em_lambda_controls_sparsity():
    ssm = gen_small_ssm(T=1000)

    config = ModelConfig.from_legacy_kwargs({
        "order":1,
        "n_warmup_iter":50,
    })

    em_state0 = make_initial_em_state(ssm)
    em_state1 = make_initial_em_state(ssm)

    no_penalty_state, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state0, config, 
                                  lambda_=0.0)

    sparse_state, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state1, config, 
                              lambda_=1.0)

    assert np.linalg.norm(sparse_state.A) < np.linalg.norm(no_penalty_state.A)
    print(f"sparse A norm {np.linalg.norm(sparse_state.A):.3f} < "
          f"no penalty A norm {np.linalg.norm(no_penalty_state.A):.3f}")
    

def test_em_A_mask_enforced():
    ssm = gen_small_ssm(T=1000)

    A_mask = np.ones_like(ssm.A)

    A_mask[0,0] = 0.0

    em_state = make_initial_em_state(ssm)
    em_state = dataclasses.replace(
        em_state,
        A_mask=A_mask,
    )

    config = ModelConfig.from_legacy_kwargs({
        "order":1,
        "n_warmup_iter":30,
    })

    result, _ = em_blas(ssm.y, ssm.F, ssm.R, em_state, config, lambda_=0.0)

    assert result.A[0,0] == 0.0


def test_em_stable():
    ssm = gen_small_ssm(T=1000)

    config = ModelConfig.from_legacy_kwargs({"order":1})
    em_state = make_initial_em_state(ssm)
    lambda_ = 0.1

    result, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config, lambda_, 
                             zeroed_index=None)

    assert np.all(np.isfinite(result.A))
    assert np.all(np.isfinite(result.Q))

    eig = np.linalg.eigvals(result.A)

    assert np.max(np.abs(eig)) < 1.2


def test_em_var3_scalar():
    """
    test EM recovery for a scalar VAR(3) model.
    """

    n_sources = 4
    order = 3
    n_orients = 1

    ssm, _, _ = gen_sparse_var_ssm(
        T=1000,
        n_sources=n_sources,
        order=order,
        n_orients=n_orients,
        seed=0,
    )

    instantiate_proximal_solvers(
        {
            "n_orients": n_orients,
            "order": order,
            "alpha": 0.0,
            "beta": 0.0,
        },
        N_sources=n_sources,
    )

    em_state = make_initial_em_state(ssm, order=order, n_orients=n_orients)

    config = ModelConfig.from_legacy_kwargs(
        {
            "order": order,
        }
    )

    lambda_ = 0.1

    final_em_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config, 
                                     lambda_, zeroed_index=None)
    
    A_est = final_em_state.A
    A_true = ssm.A

    # Extract the learnable VAR coefficient matrices
    A_true_var = A_true[:n_sources, :n_sources * order]
    A_est_var = A_est[:n_sources, :n_sources * order]


    assert np.all(np.isfinite(A_est))
    assert np.all(np.isfinite(final_em_state.Q))

    error = np.linalg.norm(A_true_var - A_est_var)
    baseline = np.linalg.norm(A_true_var)

    print(
        f"VAR(3) scalar error: {error:.3f} "
        f"(baseline {baseline:.3f})"
    )

    assert error < baseline

    # Check lag blocks
    for lag in range(order):
        true_block = A_true_var[:, lag*n_sources:(lag+1)*n_sources]
        est_block = A_est_var[:, lag*n_sources:(lag+1)*n_sources]

        print(
            f"Lag {lag+1}: "
            f"true norm={np.linalg.norm(true_block):.3f}, "
            f"estimate norm={np.linalg.norm(est_block):.3f}"
        )

    plot_transition_comparison(
        ssm.A,
        final_em_state.A,
        titles=(
            "True VAR(3)",
            "Recovered VAR(3)",
        ),
    )

    plt.show()


def test_em_var_vector(n_sources=4, n_sensors=8, n_orients=3, 
                       order=3, T=2000, lambda_=0.2, plot_transition=True,
                       **ssm_kwargs):
    """
    test EM recovery for a sparse VAR(3) model with 3 orientations.
    """

    ssm, _, _ = gen_sparse_var_ssm(
        T=T,
        n_sources=n_sources,
        order=order,
        n_orients=n_orients,
        n_sensors=n_sensors,
        seed=0,
        **ssm_kwargs
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

    m = n_sources * n_orients

    em_state = make_initial_em_state(ssm, order=order, n_orients=n_orients)
    em_state.N_sources_upper = m

    config = ModelConfig.from_legacy_kwargs(
        {
            "order": order,
            "n_orients": n_orients,
        }
    )
    start = time.perf_counter()

    final_em_state, _ = solve_params(ssm.y, ssm.F, ssm.R, em_state, config, 
                                     lambda_, zeroed_index=None)

    jax.block_until_ready(final_em_state.A)
    em_time = time.perf_counter() - start
    A_true = ssm.A
    A_est = final_em_state.A

    # ---- checks ----

    # correct shape
    assert A_est.shape == A_true.shape

    # finite parameters
    assert np.all(np.isfinite(A_est))

    # recovery should beat zero initialization
    zero_error = np.linalg.norm(A_true)
    est_error = np.linalg.norm(A_true - A_est)

    print(
        f"VAR(3) orientation recovery error: "
        f"{est_error:.3f}/{zero_error:.3f}"
    )

    assert est_error < zero_error

    # lag-1 dynamics should be strongest
    A_true_lags = [
        A_true[:m, i*m:(i+1)*m]
        for i in range(order)
    ]

    A_est_lags = [
        A_est[:m, i*m:(i+1)*m]
        for i in range(order)
    ]

    true_norms = [
        np.linalg.norm(A)
        for A in A_true_lags
    ]

    est_norms = [
        np.linalg.norm(A)
        for A in A_est_lags
    ]

    # sanity: not all lags collapse to zero
    assert max(est_norms) > 0.1 * max(true_norms)

    print("True lag norms:", true_norms)
    print("Estimated lag norms:", est_norms)

    if plot_transition:
        # Plot recovery
        plot_transition_comparison(
            A_true,
            A_est,
            titles=(
                f"True VAR({order})",
                f"Recovered VAR({order})",
            ),
        )

        plt.show()
    
    return em_time

def test_em_var_vector_medium():
    """
    test EM recovery for a sparse VAR(3) model with 3 orientations and medium
    state dimension
    """
    test_em_var_vector(n_sources=7, 
                       n_sensors=10, 
                       n_orients=3, 
                       order=3, 
                       T=4000,
                       lambda_=0.2)


def test_em_var_vector_large():
    """
    test EM recovery for a sparse VAR(2) model with 3 orientations and large
    state dimension
    """
    runtime = test_em_var_vector(n_sources=15, 
                       n_sensors=30, 
                       n_orients=3, 
                       order=2, 
                       T=5000,
                       lambda_=0.05,
                       sparsity=0.075)
    print(f"runtime: {runtime:.3f}s")
    

def test_em_var_vector_huge():
    """
    test EM recovery for a sparse VAR(2) model with 3 orientations and huge
    state dimension
    """
    
    runtime = test_em_var_vector(n_sources=50, 
                       n_sensors=100, 
                       n_orients=3, 
                       order=2, 
                       T=5000,
                       lambda_=0.05,
                       sparsity=0.075,
                       plot_transition=False)
    
    print(f"runtime: {runtime:.3f}s")
    

if __name__ == '__main__':
    # instantiate_proximal_solvers({
    #     'n_orients': 1,
    #     'order': 1,
    #     'alpha': 0.0,
    #     'beta': 0.0,
    # }, N_sources=4)

    # print("running test EM recovers small ssm A parameter")
    # test_em_recovers_A()
    # print("pass\n\n")

    # print("running test EM recovers small ssm Q parameter")
    # test_em_recovers_Q()
    # print("pass\n\n")

    # print("running test jax vs blas EM equality")
    # test_em_jax_matches_blas()
    # print("pass\n\n")

    # print("running test E< likelihood increasing monotonicity")
    # test_em_likelihood_increases()
    # print("pass\n\n")

    # print("running test EM lambda increases sparsity")
    # test_em_lambda_controls_sparsity()
    # print("pass\n\n")

    # print("running test EM scalar GC mask")
    # test_em_A_mask_enforced()
    # print("pass\n\n")

    # print("running test EM stability")
    # test_em_stable()
    # print("pass\n\n")

    # print("running test EM scalar VAR(3)")
    # test_em_var3_scalar()
    # print("pass\n\n")

    # print("running test EM vector VAR(3)")
    # test_em_var_vector()
    # print("pass\n\n")

    # print("running test EM vector VAR(3) medium") 
    # test_em_var_vector_medium()
    # print("pass\n\n")

    # print("running test EM vector VAR(2) large") 
    # test_em_var_vector_large()
    # print("pass\n\n")

    print("running test EM vector VAR(2) huge") 
    test_em_var_vector_huge()
    print("pass\n\n")