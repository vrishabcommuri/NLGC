from nlgc.opt.proximal import (proximal_param_update,
                               calculate_ss_jax)
from nlgc.test.ssm_gen import gen_small_ssm
from nlgc.opt.kalman.filter import rts_smoother_jax
from nlgc.opt.em import EMState
import jax
import jax.numpy as jnp
import numpy as np
from nlgc.config import ModelConfig
from functools import partial


def setup_proximal_test():
    """
    shared setup for proximal-gradient tests.
    """
    ssm = gen_small_ssm(T=1000)

    em_state = EMState(
        A = jnp.array(ssm.A),
        Q = jnp.array(ssm.Q),
        P0 = jnp.zeros_like(ssm.Q),
        N0 = jnp.zeros_like(ssm.Q),
        A_mask =jnp.ones_like(ssm.A),
        N_sources_upper = 4 # small ssm has 4 latent states
    )

    _, smoother_result = rts_smoother_jax(
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

    em_new, _ = proximal_param_update(em_state, smoother_result, lambda_=0.0, 
                                   config=config)

    np.testing.assert_allclose(
        np.asarray(em_new.A),
        np.asarray(A_closed),
        atol=1e-3,
    )


def test_proximal_recovers_ground_truth_transition():
    ssm, em_state, smoother_result = setup_proximal_test()

    em_new, _ = proximal_param_update(em_state, smoother_result, lambda_=0.1, 
                                   config=config)

    np.testing.assert_allclose(em_new.A, ssm.A, atol=1e-1)
    

# def test_proximal_decreases_objective():
#     ssm, em_state, smoother_result = setup_proximal_test()

#     s1, s2, *_ = calculate_ss_jax(em_state, smoother_result)

#     Qinv = jnp.linalg.inv(em_state.Q)

#     d = jnp.sqrt(jnp.diag(s2))
#     d_safe = jnp.maximum(d, 1e-12)
#     s2_tilde = s2 / jnp.outer(d_safe, d_safe)
#     s1_tilde = s1 / d_safe[None, :]

#     # whiten targets by Q^{-1/2} 
#     # (using eigh since Q is symmetric positive definite)
#     evals, evecs = jnp.linalg.eigh(Q)
#     evals_safe = jnp.maximum(evals, 1e-12)
    
#     q_inv_sqrt = evecs @ jnp.diag(1.0 / jnp.sqrt(evals_safe)) @ evecs.T

#     # apply to s1 and initial A
#     s1_tilde = q_inv_sqrt @ s1_tilde

#     def f_fun(x):
#         xs2 = x @ s2_tilde
#         return (jnp.trace(xs2 @ x.T) - 2.0 * jnp.trace(s1_tilde @ x.T))

#     before = f_fun(em_state.A, s1, s2, Qinv)

#     em_new = proximal_param_update(em_state, smoother_result, lambda_=0.1)

#     after = f(em_new.A, s1, s2, Qinv)
#     assert after < before
#     print(f"{before=} > {after=}")


def test_regularization_reduces_transition_norm():
    ssm, em_state, smoother_result = setup_proximal_test()

    A0 = proximal_param_update(em_state, smoother_result, lambda_=0.0, 
                               config=config)[0].A

    A1 = proximal_param_update(em_state, smoother_result, lambda_=0.5, 
                               config=config)[0].A

    assert np.linalg.norm(A1) < np.linalg.norm(A0)

    print(f"regularized norm {np.linalg.norm(A1):.3f} < "
          f"closed form norm {np.linalg.norm(A0):.3f}")


def test_large_lambda_zeroes_transition():
    ssm, em_state, smoother_result = setup_proximal_test()

    A = proximal_param_update(em_state, smoother_result, lambda_=1e6, 
                              config=config)[0].A

    np.testing.assert_allclose(A, 0, atol=1e-6)


def test_increasing_lambda_increases_sparsity():
    ssm, em_state, smoother_result = setup_proximal_test()

    nnz = []
    test_lams = [0, 0.05, 0.1, 0.5]
    for lam in test_lams:
        A = proximal_param_update(em_state, smoother_result, lambda_=lam, 
                                  config=config)[0].A

        nnz.append(np.count_nonzero(np.abs(A) > 1e-8))

    assert nnz == sorted(nnz, reverse=True)

    print("number of nonzero terms: \n")
    for i in range(len(test_lams)):
        print(f"\t lambda {test_lams[i]}: {nnz[i]} nonzero")


# def test_jitted_and_eager_match():
#     ssm, em_state, smoother_result = setup_proximal_test()

#     eager_state, _ = proximal_param_update(em_state, smoother_result, lambda_=0.1, 
#                                         config=config)

#     jit_fun = partial(jax.jit(proximal_param_update), static_argnames=("config",))

#     compiled_state, _ = jit_fun(em_state, smoother_result, lambda_=0.1, 
#                                 config=config)

#     np.testing.assert_allclose(
#         eager_state.A,
#         compiled_state.A,
#         atol=1e-3
#     )


if __name__ == '__main__':
    config = ModelConfig.from_legacy_kwargs({
                "order": 1,
                "n_eigenmodes": 1,
                "n_orients": 1,
            })
    
    print("running test zero lambda closed form equality")
    test_proximal_lambda_zero_recovers_closed_form()
    print("pass\n\n")

    print("running test small lambda perturbation equality")
    test_proximal_recovers_ground_truth_transition()   
    print("pass\n\n")

    # print("running test proximal decreases objective function")
    # test_proximal_decreases_objective()
    # print("pass\n\n")

    print("running test proximal reduces A matrix norm")
    test_regularization_reduces_transition_norm()
    print("pass\n\n")

    print("running test large lambda zeroes matrix")
    test_large_lambda_zeroes_transition()
    print("pass\n\n")

    print("running test lambda monotonicity")
    test_increasing_lambda_increases_sparsity()
    print("pass\n\n")

    # print("running test jax compilation equivalence")
    # test_jitted_and_eager_match()
    # print("pass\n\n")