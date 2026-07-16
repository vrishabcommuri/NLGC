from scipy import linalg
import control
import numpy as np
import jax


def solve_ss_covariance_qz(a, f, q, r):
    try:
        P = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)           
    except np.linalg.LinAlgError:
        P = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    except ValueError:
        try:
            P, _, _ = control.dare(a.T, f.T, q, r, stabilizing=True, 
                                   method=None)
        except ValueError:
            P, _, _ = control.dare(a.T, f.T, q, r, stabilizing=False, 
                                   method=None)
    return P


def solve_ss_covariance_newton_raphson(A, F, Q, R, P0, N0, maxiter=20, tol=1e-5, 
                                       maxerror=0.02):
    """
    see Benner https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=808626
    for the algorithm 
    this is faster than solve_discrete_are for a couple of reasons:
    1. solve_discrete_are constructs a symplectic block matrix and then does 
       a QZ factorization on that (schur decomposition) which scales O(n^3) in 
       the dimension n of the matrix. we use a solver for the lyapunov 
       equation that also does a schur decomposition, but does not construct
       the block matrix to do the decomposition, this scales O((n/3)^3) which
       is a significant saving at our matrix sizes.
    2. solve_discrete_are does not take into account any prior knowledge, 
       but we do know the smoother covariance from the previous iteration, so 
       we use that as the starting point and then do a newton step to update 
       it. 
    """

    def newton_raphson_update(_, state_cov):
        N = A.shape[0]
        X_k, N_k = state_cov

        # (R + B^T X_k B)^{-1} B^T X_k A
        # rkinv = linalg.inv(r + f.T @ x_k @ f)
        # clgain_k = rkinv @ f.T @ x_k @ a
        # replace inv with cholesky solve
        M = F.T @ X_k @ F + R
        L = jax.scipy.linalg.cholesky(M, lower=True)
        clgain_k = jax.scipy.linalg.cho_solve((L, True), F.T @ X_k @ A)

        A_k = A - F @ clgain_k 

        # DARE residual: R(X_k) = Q - X_k + A^T X_k A - A^T X_k B (R + B^T X_k B)^{-1} B^T X_k A
        Dr_k = Q - X_k + A.T @ X_k @ A - A.T @ X_k @ F @ clgain_k

        # newton-raphson first linearizes R(X_k ​+ N_k​) ≈ R(X_k​)+ DR(X_k​)[N_k​]
        # where D..[N_k] is the directional derivative (similar to jacobian) 
        # evaluated at direction N_k.
        # it is known that DR(X_k)[N] = A_k^TXA_k - N where A_k = A - BK_k above
        # set equal to zero: R(X_k​)+ DR(X_k​)[N_k​] = 0 ==> DR(X_k​)[N_k​] = -R(X_k)
        # 
        # ideally, we solve lyapunov equation A_k^T N_k A_k - N_k + R(X_k) = 0
        # N_k = linalg.solve_discrete_lyapunov(A_k.T, Dr_k)
        # but this doesn't exist in jax ecosystem, so we use a slightly more 
        # expensive and less robust conjugate gradient approach.
        #
        # let L(X) = N_k - A_k^T N_k A_k 
        # then, L(X) = Dr_k can be solved via CG by vectorizing X and Dr_k
        def lyap_operator(x):
            X = x.reshape(N, N) # CG takes vector input
            Y = X - A_k.T @ X @ A_k
            return Y.ravel()

        b = Dr_k.ravel() # GC rhs must be vector

        x, info = jax.scipy.sparse.linalg.bicgstab(
            lyap_operator,
            b,
            x0=N_k.ravel(),      # warm start from previous EM iteration
            tol=1e-8,
            maxiter=50,
        )

        N_k = x.reshape(N, N)

        t_k = 1
        X_k = X_k + t_k * N_k
        X_k = 0.5 * (X_k + X_k.T)
        return (X_k, N_k) 

    (P_k, N_k) = jax.lax.fori_loop(lower=0, 
                            upper=maxiter, 
                            body_fun=newton_raphson_update, 
                            init_val=(P0, N0))

    return (P_k, N_k)
