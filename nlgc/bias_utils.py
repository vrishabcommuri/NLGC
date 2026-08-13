import numpy as np
import warnings
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg
from nlgc.opt.proximal import calculate_ss


def compute_bias(em_state, smoother_result, config):
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    Q = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    n_orients = config.latent.n_orients
    p = config.latent.order
    
    bias = sample_path_bias(Q, A, x, P, B, A_mask, 
                            n_orients, m, p)
    return bias


def compute_bias_idx(source, target, em_state, smoother_result, config):
    p = config.latent.order
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    Q = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain
    
    n_orients = config.latent.n_orients
    n_eigenmodes = config.latent.n_eigenmodes

    bias = bias_by_idx(source, target, Q, A, x, P, B, A_mask, 
                       n_eigenmodes, n_orients, m, p)

    return bias


def sample_path_bias(q, a, x_bar, s_bar, b, A_mask, n_orients, m, p):
    """Computes the expected complete-data bias in the deviance"""
    _, dtot = a.shape
    # TODO: should this be smoother stats (current) or plug-in estimate as in
    # nlgc_test_utils branch?
    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)

    bias = 0

    s2_factor = linalg.cho_factor(s2)
    
    # sources couple covariance structure only within shared orients
    n_sources = m // n_orients 

    # remove sources one by one and test the perturbation/optimism. 
    # sources are coupled via shared orientations are removed as blocks, but
    # otherwise sources are independent so we can proceed blockwise and sum to
    # get the total bias
    for idx_s in range(n_sources):
        # block encompasses the entire source structure
        block = slice(idx_s * n_orients, (idx_s + 1) * n_orients)
        block_mask = A_mask[block, :] 
        keep_idx = block_mask.reshape(-1).astype(bool)
        n_free = keep_idx.sum()

        ai = a[block, :]
        Q_block = q[block, block]
        Q_inv_block = np.linalg.inv(Q_block)

        # residual
        diff = s1[block, :] - ai @ s2

        # ---- gradient ----
        ldot_matrix = Q_inv_block @ diff
        ldot = ldot_matrix.reshape(-1)

        # slice down to only free parameters
        ldot_free = ldot[keep_idx]

        # ---- Hessian ----
        # The old hessian code explicity constructed the hessian operator from
        # the gradient: ldot = Qinv (s1 - A s2)
        # 
        # The hessian ldotdot is the gradient of ldot wrt to A. This is done by
        # vectorizing the expression and then using the vec property 
        # vec(ABC) = kron(C^T, B) * vec(A)
        #
        # Taking the gradient explicitly would yield
        # ldotdot = -np.kron(Q_inv_block, s2)
        # but this matrix representation would scale poorly for even modest Q. 
        # Instead, the Hessian can be viewed as a linear operator; the matrix 
        # representation simply encodes a linear transformation of an input, so 
        # we can construct an operator that does this and solve for the input 
        # using conjugate gradients, similar to what was done in the previous 
        # versions of em_jax lyapunov and riccati solvers. So instead of solving
        # ldotdot * x = ldot
        # which constructed both ldotdot and ldot as matrices and using lstsq we
        # can simply construct 
        # H[x] = ldot 
        # and use CG to solve for x.
        def neg_hess_mv(v):
            # here X is basically the perturbation of A along gradient dirs
            X = np.zeros((n_orients, dtot), dtype=v.dtype)
            # effectively ldotdot_free = ldotdot[keep_idx][:, keep_idx] we can
            # use the same indices for keep_idx, since this is pertubation of A
            X.reshape(-1)[keep_idx] = v

            HX = Q_inv_block @ X @ s2

            return HX.reshape(-1)[keep_idx]

        H_op = LinearOperator(
            shape=(n_free, n_free),
            matvec=neg_hess_mv,
            dtype=np.result_type(Q_inv_block.dtype, s2.dtype),
        )

        # ---- preconditioner ---- 
        # For the unrestricted problem the inverse Hessian has form 
        # H(X) = Q_block^{-1} @ X @ s2 = Y, therefore
        # H^{-1}(Y) = Q_block @ Y @ s2^{-1} = X
        #
        # For the restricted problem, this preconditioner will get us close to
        # the restricted operator. 

        def precond_mv(v):
            Y = np.zeros((n_orients, dtot), dtype=v.dtype)
            Y.reshape(-1)[keep_idx] = v

            X = Q_block @ Y
            X = linalg.cho_solve(s2_factor, X.T).T

            return X.reshape(-1)[keep_idx]

        M_op = LinearOperator(
            shape=(n_free, n_free),
            matvec=precond_mv,
            dtype=np.result_type(Q_block.dtype, s2.dtype),
        )

        n_iter = 0

        residuals = []

        def callback(xk):
            nonlocal n_iter
            n_iter += 1
            r = ldot_free - H_op @ xk
            residuals.append(np.linalg.norm(r))

        step, info = cg(
            H_op,
            ldot_free,
            M=M_op,
            callback=callback,
            rtol=1e-10,
            atol=0.0,
        )
        
        assert info == 0, f"bias estimation Hessian CG exit with code {info}"

        bias += n * np.dot(ldot_free, step)
    return bias


def _voxel_block(v, n_orient=3):
    """Return slice for RAS components of voxel v."""
    return slice(n_orient * v, n_orient * (v + 1))


def bias_by_idx(src, targ, q, a, x_bar, s_bar, b, A_mask, n_eigenmodes, 
                n_orients, m, p):
    """Computes the expected bias for a specific source -> target connection."""
    # TODO !!!this is probably a bug where the blocks are not correctly aligned
    # to the effective eigenmodes, resulting in incorrect steps in the iteration
    # for eigenmodes > 1. since this func is vestigial for now, we can leave it,
    # but we should come back to it
    
    _, dtot = a.shape

    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)

    eff_eigenmodes = n_eigenmodes * n_orients

    # identify target block (rows in A)
    targ_block = _voxel_block(targ, n_orients)

    # identify source columns in A (across all lags)
    src_start = src * n_orients
    src_end = (src + 1) * n_orients
    src_cols = []
    for lag in range(p):
        src_cols.extend(range(src_start + lag * m, src_end + lag * m))
        
    # determine which of these columns are explicitly masked out by A_mask
    delete_cols = set()
    if A_mask is not None:
        idx_large_voxel = targ // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes
        for idx in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = transition_mask_to_parameter_indices(A_mask, idx, m, dtot)
            delete_cols.update(removed_idx)

    # keep only the valid (unmasked) source columns
    valid_src_cols = [c for c in src_cols if c not in delete_cols]
    
    # if the entire connection is masked, there are no parameters to perturb,
    # bias is 0
    if len(valid_src_cols) == 0:
        return 0.0

    # extract blocks for computation
    ai = a[targ_block, :]
    Q_block = q[targ_block, targ_block]
    Q_inv_block = np.linalg.inv(Q_block)

    #  residual 
    diff = s1[targ_block, :] - ai @ s2
    
    # restrict residual and s2 to just the valid source columns
    diff_src = diff[:, valid_src_cols]
    s2_src = s2[np.ix_(valid_src_cols, valid_src_cols)]
    
    # gradient
    ldot_matrix = Q_inv_block @ diff_src
    ldot = ldot_matrix.reshape(-1)

    # Hessian
    ldotdot = -np.kron(Q_inv_block, s2_src)

    try:
        c, low = linalg.cho_factor(-ldotdot)
        temp = linalg.cho_solve((c, low), ldot)
        bias = n * (ldot @ temp)
    except linalg.LinAlgError:
        warnings.warn('Connection src={:d} -> targ={:d} ldotdot is not negative definite: '
                      'setting positive eigenvalues equal to zero, '
                      'result may not be accurate.'.format(src, targ), 
                      RuntimeWarning, stacklevel=2)
        e, v = linalg.eigh(-ldotdot)
        temp = v.T @ ldot 
        idx = e > 0
        bias = np.sum(temp[idx] ** 2 / e[idx])
        bias *= n

    return bias


def wald_by_idx(src, targ, q, a, x_bar, s_bar, b, A_mask, n_eigenmodes, 
                n_orients, m, p):
    """
    similar to bias_by_idx, computes the Wald statistic for pruning a
    source->target connection.

    Wald score \theta^T H(\theta) \theta measures perturbation for a given link
    using the smoother statistics; i.e., how much does the likelihood change
    when we zero out this link? 
    """
    
    _, dtot = a.shape
    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)
    eff_eigenmodes = n_eigenmodes * n_orients

    targ_block = slice(targ * eff_eigenmodes, (targ + 1) * eff_eigenmodes)

    src_cols = []
    for lag in range(p):
        src_cols.extend(range(src * eff_eigenmodes + lag * m, (src + 1) * eff_eigenmodes + lag * m))
        
    # TODO: replace complicated delete logic with simpler mask raveling like
    # what is done in sample_path_bias
    delete_cols = set()
    if A_mask is not None:
        idx_large_voxel = targ // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes
        for idx in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = transition_mask_to_parameter_indices(A_mask, idx, m, dtot)
            delete_cols.update(removed_idx)

    valid_src_cols = [c for c in src_cols if c not in delete_cols]

    if len(valid_src_cols) == 0:
        return 0.0, 0

    # extract parameter values for this specific link
    ai_edge = a[targ_block, :][:, valid_src_cols]
    
    # extract the Hessian components
    Q_block = q[targ_block, targ_block]
    Q_inv_block = np.linalg.inv(Q_block)
    s2_src = s2[np.ix_(valid_src_cols, valid_src_cols)]
    
    # compute the Hessian strictly for this connection
    ldotdot = -np.kron(Q_inv_block, s2_src)

    # compute the Wald Statistic: n * (theta^T * H * theta)
    theta = ai_edge.reshape(-1) 
    wald_score = n * (theta.T @ (-ldotdot) @ theta)

    # degrees of freedom used for pre-screening using central chi2 dist
    k_dof = len(theta)

    return wald_score, k_dof


def debias_deviances(dev_raw, bias_f, bias_r):
    d = dev_raw.copy()
    bias_mat = bias_r - bias_f
    d[bias_r != 0] -= bias_mat[bias_r != 0]
    np.fill_diagonal(d, 0)
    d[d < 0] = 0
    return d


def transition_mask_to_parameter_indices(A_mask, target_idx, dxm, dtot):
    """
    Return flattened companion-matrix parameter indices
    removed by fixing transitions to zero.
    """
    removed = []

    sources = np.where(A_mask[target_idx] == 0)[0]

    for src in sources:
        removed.extend(range(int(src), int(dtot), int(dxm)))

    return np.asarray(removed, dtype=int)


