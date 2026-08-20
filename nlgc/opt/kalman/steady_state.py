from scipy import linalg
import control
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

DEBUG = True

def solve_ss_covariance_newton_raphson(a, f, q, r, s_init, maxiter=50, 
                                        tol=1e-7, maxerror=0.02, verbose=True):
    # see Benner https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=808626
    # for the algorithm 
    # this is faster than solve_discrete_are for a couple of reasons:
    # 1. solve_discrete_are constructs a symplectic block matrix and then does 
    #    a QZ factorization on that (schur decomposition) which scales O(n^3) in 
    #    the dimension n of the matrix. we use a solver for the lyapunov 
    #    equation that also does a schur decomposition, but does not construct
    #    the block matrix to do the decomposition, this scales O((n/3)^3) which
    #    is a significant saving at our matrix sizes.
    # 2. solve_discrete_are does not take into account any prior knowledge, 
    #    but we do know the smoother covariance from the previous iteration, so 
    #    we use that as the starting point and then do a newton step to update 
    #    it. 

    x_k = s_init
    errors = []
    for k in range(maxiter):
        # (R + B^T X_k B)^{-1} B^T X_k A
        # rkinv = linalg.inv(r + f.T @ x_k @ f)
        # clgain_k = rkinv @ f.T @ x_k @ a

        # replace inv with cholesky solve
        M = r + f.T @ x_k @ f
        
        L = linalg.cholesky(M, lower=True)
        clgain_k = linalg.cho_solve((L, True), f.T @ x_k @ a)

        a_k = a - f @ clgain_k 

        # DARE residual: R(X_k) = Q - X_k + A^T X_k A - A^T X_k B (R + B^T X_k B)^{-1} B^T X_k A
        dr_k = q - x_k + a.T @ x_k @ a - a.T @ x_k @ f @ clgain_k

        # newton-raphson first linearizes R(X_k ​+ N_k​) ≈ R(X_k​)+ DR(X_k​)[N_k​]
        # where D..[N_k] is the directional derivative (similar to jacobian) 
        # evaluated at direction N_k.
        # it is known that DR(X_k)[N] = A_k^TXA_k - N where A_k = A - BK_k above
        # set equal to zero: R(X_k​)+ DR(X_k​)[N_k​] = 0 ==> DR(X_k​)[N_k​] = -R(X_k)
        # 
        # thus, solve lyapunov equation A_k^T N_k A_k - N_k + R(X_k) = 0
        n_k = linalg.solve_discrete_lyapunov(a_k.T, dr_k)
        
        # compute t_k line search learning rate
        # g_k = f @ rkinv @ f.T
        # v_k = (a - f @ clgain_k).T @ n_k @ g_k @ n_k @ (a - f @ clgain_k)
        # f_k = lambda t: linalg.norm((1-t)*dr_k - t**2*v_k)
        # # t_k = min_t || f_k(t) ||_F for t in [0, 2]
        # t_k = minimize(f_k, x0=1, bounds=[(0,2)]).x[0] 
        t_k = 1
        
        x_kp1 = x_k + t_k * n_k

        error = linalg.norm(dr_k, ord='fro')
        errors.append(error)

        x_k = x_kp1 # update x_k for possible next iteration

        if error < tol:
            return x_k, n_k

    if DEBUG:
        print("ss cov newton raphson nonconvergence, try qz solver")
    return solve_ss_covariance_qz(a.T, f.T, q, r)


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
    return P, np.zeros_like(P)


def solve_ss_covariance(a, f, q, r, s_init=None):
    if s_init is None:
        if DEBUG:
            print("ss covariance uninitialized, using qz solver")
        P, N = solve_ss_covariance_qz(a, f, q, r)
    else:
        try:
            P, N = solve_ss_covariance_newton_raphson(a.T, f.T, q, r, s_init)
        except np.linalg.LinAlgError:
            if DEBUG:
                print("ss cov newton raphson failed! try qz solver")
            P, N = solve_ss_covariance_qz(a, f, q, r)

    return P, N
