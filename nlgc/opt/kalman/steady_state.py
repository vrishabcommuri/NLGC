from scipy import linalg
import control
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

DEBUG = False

def dare_exact_line_search(dr_k, n_k, a_k, f, chol_M, tmax=1.0):
    """
    Minimize ||R(X_k + t N_k)||_F over 0 <= t <= tmax, using

        R(X_k + t N_k) = (1 - t) dr_k - t^2 v_k. 
        
    which is the linearized second-order expression.

    The expression ||R(X_k + t N_k)||_F is quartic, so we can foil and make the
    appropriate substitutions:
    
    phi(t) = ||R(X_k + t N_k)||_F = a(1-t)^2 - 2bt^2(1-t) + ct^4  
    
    A cheap minimization is to pick the smallest root of the cubic derivative:

    phi'(t) = 4*c*t^3 + 6*b*t^2 + 2*(a - 2*b)*t - 2*a

    we use the best candidate t as the learning rate in t * N_k. this usage is
    apparent from the relation above, where t=1 cancels out the newton residual
    term dr_k and leaves only the -v_k, which is appropriate *only* when N_k is
    small. 
    """
    L, lower = chol_M

    # G = F M^{-1} F^T, formed without an explicit inverse.
    # equivalent to F @ cho_solve((L, lower), F.T).
    g_k = f @ linalg.cho_solve((L, lower), f.T)

    v_k = a_k.T @ n_k @ g_k @ n_k @ a_k

    a0 = np.vdot(dr_k, dr_k).real
    b0 = np.vdot(dr_k, v_k).real
    c0 = np.vdot(v_k, v_k).real

    # phi'(t) = 4*c*t^3 + 6*b*t^2 + 2*(a - 2*b)*t - 2*a
    coeff = [4.0 * c0, 6.0 * b0, 2.0 * (a0 - 2.0 * b0), -2.0 * a0]

    # degenerate/near-converged case: any choice is effectively equivalent.
    scale = max(a0, abs(b0), c0, 1.0)
    if max(abs(x) for x in coeff) <= 1e-14 * scale:
        return 1.0, v_k

    roots = np.roots(coeff)
    candidates = [0.0, float(tmax)]

    for z in roots:
        if abs(z.imag) <= 1e-10 * max(1.0, abs(z.real)):
            t = float(z.real)
            if 0.0 <= t <= tmax:
                candidates.append(t)

    def phi(t):
        return a0 * (1.0 - t)**2 - 2.0 * b0 * t**2 * (1.0 - t) + c0 * t**4

    t_best = min(candidates, key=phi)
    return t_best, v_k


def solve_ss_covariance_newton_raphson(a, f, q, r, s_init, maxiter=25,
    tol=1e-7, maxerror=0.02, verbose=True):
    """
    Solve the stabilizing DARE in the convention

        R(X) = Q - X + A.T X A
               - A.T X F (R + F.T X F)^(-1) F.T X A = 0.

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

    # symmetrize
    x_k = 0.5 * (s_init + s_init.T)
    errors = []

    norm_scale = max(1.0, linalg.norm(q, ord="fro"))

    for k in range(maxiter):
        if verbose and DEBUG:
            print(f"NR iter {k + 1}/{maxiter}")

        # keep roundoff from slowly destroying symmetry.
        x_k = 0.5 * (x_k + x_k.T)

        M = r + f.T @ x_k @ f
        M = 0.5 * (M + M.T)

        try:
            L = linalg.cholesky(M, lower=True, check_finite=False)
        except linalg.LinAlgError:
            if verbose and DEBUG:
                print("R + F.T @ X @ F is not SPD; falling back to QZ.")
            return solve_ss_covariance_qz(a.T, f.T, q, r)

        # K = M^{-1} F.T X A
        clgain_k = linalg.cho_solve(
            (L, True), f.T @ x_k @ a, check_finite=False
        )

        a_k = a - f @ clgain_k

        # DARE residual: R(X_k) = Q - X_k + A^T X_k A - 
        #                A^T X_k B (R + B^T X_k B)^{-1} B^T X_k A
        dr_k = q - x_k + a.T @ x_k @ a - a.T @ x_k @ f @ clgain_k
        dr_k = 0.5 * (dr_k + dr_k.T)

        residual = linalg.norm(dr_k, ord="fro")
        rel_residual = residual / max(norm_scale, linalg.norm(x_k, ord="fro"))
        errors.append(residual)

        if verbose and DEBUG:
            rho = max(abs(linalg.eigvals(a_k)))
            print(
                f"  ||R||_F={residual:.3e}, relative={rel_residual:.3e}, "
                f"rho(A_cl)={rho:.6f}"
            )

        if rel_residual < tol:
            return x_k, np.zeros_like(x_k)

        # newton-raphson first linearizes R(X_k ​+ N_k​) ≈ R(X_k​)+ DR(X_k​)[N_k​]
        # where D..[N_k] is the directional derivative (similar to jacobian) 
        # evaluated at direction N_k.
        # it is known that DR(X_k)[N] = A_k^TXA_k - N where A_k = A - BK_k above
        # set equal to zero: R(X_k​)+ DR(X_k​)[N_k​] = 0 ==> DR(X_k​)[N_k​] = -R(X_k)
        # 
        # thus, solve lyapunov equation A_k^T N_k A_k - N_k + R(X_k) = 0
        # This assumes A_k is Schur stable.
        try:
            n_k = linalg.solve_discrete_lyapunov(a_k.T, dr_k, method="bilinear")
        except linalg.LinAlgError:
            if verbose and DEBUG:
                print("Lyapunov solve failed; falling back to QZ.")
            return solve_ss_covariance_qz(a.T, f.T, q, r)

        n_k = 0.5 * (n_k + n_k.T)

        # residual-minimizing line search, initially constrained to a
        # non-extrapolating Newton step.
        t_k, _ = dare_exact_line_search(
            dr_k=dr_k,
            n_k=n_k,
            a_k=a_k,
            f=f,
            chol_M=(L, True),
            tmax=1.0,
        )

        # numerical safeguard: require actual residual reduction and SPD M.
        # usually t_k from the exact polynomial is accepted immediately.
        old_residual = residual
        accepted = False

        for _ in range(20):
            x_trial = x_k + t_k * n_k
            x_trial = 0.5 * (x_trial + x_trial.T)

            M_trial = r + f.T @ x_trial @ f
            M_trial = 0.5 * (M_trial + M_trial.T)

            try:
                linalg.cholesky(M_trial, lower=True, check_finite=False)

                K_trial = linalg.cho_solve(
                    linalg.cho_factor(M_trial, lower=True, check_finite=False),
                    f.T @ x_trial @ a,
                    check_finite=False,
                )
                dr_trial = (
                    q - x_trial + a.T @ x_trial @ a
                    - a.T @ x_trial @ f @ K_trial
                )
                dr_trial = 0.5 * (dr_trial + dr_trial.T)
                trial_residual = linalg.norm(dr_trial, ord="fro")

                if np.isfinite(trial_residual) and trial_residual < old_residual:
                    accepted = True
                    break
            except linalg.LinAlgError:
                pass

            t_k *= 0.5

        if not accepted:
            if verbose and DEBUG:
                print("Line search could not reduce residual; "
                      "falling back to QZ.")
            return solve_ss_covariance_qz(a.T, f.T, q, r)

        if verbose and DEBUG:
            print(f"  step={t_k:.4g}, trial ||R||_F={trial_residual:.3e}")

        x_k = x_trial

    if verbose and DEBUG:
        print("Steady-state covariance Newton iteration did not converge; "
              "using QZ.")
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
