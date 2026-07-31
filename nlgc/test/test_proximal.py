from nlgc.opt.proximal import (instantiate_proximal_solvers, 
                               proximal_param_update,
                               calculate_ss_jax, f)
from nlgc.test.ssm_gen import gen_small_ssm
from nlgc.opt.kalman.filter import rts_smoother_jax
from nlgc.opt.em import EMState
import jax
import jax.numpy as jnp
import numpy as np

def setup_proximal_test():
    """
    shared setup for proximal-gradient tests.
    """
    # proximal_param_update reads the module-level `solver`/`solve_for_Q`
    # globals, so they have to exist before any test touches it. This used to
    # live only in the __main__ block below, which meant the file passed when run
    # as a script and every test errored with "'NoneType' has no attribute 'run'"
    # under pytest -- unless another test module happened to instantiate first.
    instantiate_proximal_solvers({
        'n_orients': 1,
        'order': 1,
        'alpha': 0.0,
        'beta': 0.0,
    }, N_sources=4)

    ssm = gen_small_ssm(T=1000)

    em_state = EMState(
        A = jnp.array(ssm.A),
        Q = jnp.array(ssm.Q),
        P0 = jnp.zeros_like(ssm.Q),
        N0 = jnp.zeros_like(ssm.Q),
        N_sources_upper = 4 # small ssm has 4 latent states
    )

    smoother_result = rts_smoother_jax(
        jnp.array(ssm.y), 
        jnp.array(ssm.F), 
        jnp.array(ssm.R),
        em_state,
    )

    return ssm, em_state, smoother_result


def test_proximal_lambda_zero_recovers_closed_form():
    ssm, em_state, smoother_result = setup_proximal_test()

    s1, s2, *_ = calculate_ss_jax(em_state, smoother_result)

    A_closed = s1 @ jnp.linalg.inv(s2)

    em_new = proximal_param_update(em_state, smoother_result, lambda_=0.0)

    np.testing.assert_allclose(
        np.asarray(em_new.A),
        np.asarray(A_closed),
        atol=1e-3,
    )


def test_proximal_recovers_ground_truth_transition():
    ssm, em_state, smoother_result = setup_proximal_test()

    em_new = proximal_param_update(em_state, smoother_result, lambda_=0.1)

    np.testing.assert_allclose(em_new.A, ssm.A, atol=1e-1)
    

def test_proximal_decreases_objective():
    ssm, em_state, smoother_result = setup_proximal_test()

    s1, s2, *_ = calculate_ss_jax(em_state, smoother_result)

    Qinv = jnp.linalg.inv(em_state.Q)

    before = f(em_state.A, s1, s2, Qinv)

    em_new = proximal_param_update(em_state, smoother_result, lambda_=0.1)

    after = f(em_new.A, s1, s2, Qinv)
    assert after < before
    print(f"{before=} > {after=}")


def test_regularization_reduces_transition_norm():
    ssm, em_state, smoother_result = setup_proximal_test()

    A0 = proximal_param_update(em_state, smoother_result, lambda_=0.0).A

    A1 = proximal_param_update(em_state, smoother_result, lambda_=0.5).A

    assert np.linalg.norm(A1) < np.linalg.norm(A0)

    print(f"regularized norm {np.linalg.norm(A1):.3f} < "
          f"closed form norm {np.linalg.norm(A0):.3f}")


def test_large_lambda_zeroes_transition():
    ssm, em_state, smoother_result = setup_proximal_test()

    A = proximal_param_update(em_state, smoother_result, lambda_=1e6).A

    np.testing.assert_allclose(A, 0, atol=1e-6)


def test_increasing_lambda_increases_sparsity():
    ssm, em_state, smoother_result = setup_proximal_test()

    nnz = []
    test_lams = [0, 0.05, 0.1, 0.5]
    for lam in test_lams:
        A = proximal_param_update(em_state, smoother_result, lambda_=lam).A

        nnz.append(np.count_nonzero(np.abs(A) > 1e-8))

    assert nnz == sorted(nnz, reverse=True)

    print("number of nonzero terms: \n")
    for i in range(len(test_lams)):
        print(f"\t lambda {test_lams[i]}: {nnz[i]} nonzero")


def test_jitted_and_eager_match():
    ssm, em_state, smoother_result = setup_proximal_test()

    eager_state = proximal_param_update(em_state, smoother_result, lambda_=0.1)

    jit_fun = jax.jit(proximal_param_update)

    compiled_state = jit_fun(em_state, smoother_result, lambda_=0.1)

    np.testing.assert_allclose(
        eager_state.A,
        compiled_state.A,
        atol=1e-3
    )


if __name__ == '__main__':
    instantiate_proximal_solvers({
        'n_orients': 1,
        'order': 1,
        'alpha': 0.0,
        'beta': 0.0,
    }, N_sources=4)

    print("running test zero lambda closed form equality")
    test_proximal_lambda_zero_recovers_closed_form()
    print("pass\n\n")

    print("running test small lambda perturbation equality")
    test_proximal_recovers_ground_truth_transition()   
    print("pass\n\n")

    print("running test proximal decreases objective function")
    test_proximal_decreases_objective()
    print("pass\n\n")

    print("running test proximal reduces A matrix norm")
    test_regularization_reduces_transition_norm()
    print("pass\n\n")

    print("running test large lambda zeroes matrix")
    test_large_lambda_zeroes_transition()
    print("pass\n\n")

    print("running test lambda monotonicity")
    test_increasing_lambda_increases_sparsity()
    print("pass\n\n")

    print("running test jax compilation equivalence")
    test_jitted_and_eager_match()
    print("pass\n\n")