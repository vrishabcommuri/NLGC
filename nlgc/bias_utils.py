import numpy as np
import warnings
from scipy import linalg

from nlgc.opt.proximal import calculate_ss


def compute_bias(em_state, smoother_result, config):
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    Q = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    p = config.latent.order
    
    bias = sample_path_bias(Q, A, x, P, B, A_mask, 
                            n_eigenmodes, n_orients, 
                            m, p)
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


def _voxel_block(v, n_orient=3):
    """Return slice for RAS components of voxel v."""
    return slice(n_orient * v, n_orient * (v + 1))


def sample_path_bias(q, a, x_bar, s_bar, b, A_mask, n_eigenmodes, n_orients, 
                     m, p):
    """Computes the expected complete-data bias in the deviance"""
    _, dtot = a.shape
    
    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)

    eff_eigenmodes = n_eigenmodes * n_orients
    bias = 0
    
    dxn_voxels = m // n_orients

    for idx_v in range(dxn_voxels):
        block = _voxel_block(idx_v, n_orients)

        ai = a[block, :]
        Q_block = q[block, block]
        Q_inv_block = np.linalg.inv(Q_block)

        # residual 
        diff = s1[block, :] - ai @ s2
        
        # gradient
        ldot_matrix = Q_inv_block @ diff
        ldot = ldot_matrix.reshape(-1)

        # Hessian 
        ldotdot = -np.kron(Q_inv_block, s2)

        delete_idxs = []

        idx_large_voxel = idx_v // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes

        for idx in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = transition_mask_to_parameter_indices(A_mask, idx, m, dtot)
            delete_idxs.extend(removed_idx)

        delete_idxs_3d = []
        for r in range(n_orients):
            for idx in delete_idxs:
                delete_idxs_3d.append(r * dtot + idx)

        delete_idxs_3d = sorted(set(delete_idxs_3d))

        if len(delete_idxs_3d) > 0:
            ldot = np.delete(ldot, delete_idxs_3d)
            ldotdot = np.delete(ldotdot, delete_idxs_3d, axis=0)
            ldotdot = np.delete(ldotdot, delete_idxs_3d, axis=1)

        step, _, _, _ = np.linalg.lstsq(-ldotdot, ldot, rcond=None)
        bias += n * (ldot @ step)

    return bias


def bias_by_idx(src, targ, q, a, x_bar, s_bar, b, A_mask, n_eigenmodes, 
                n_orients, m, p):
    """Computes the expected bias for a specific source -> target connection."""
    
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
        
    delete_cols = set()
    if A_mask is not None:
        idx_large_voxel = targ // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes
        for idx in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = transition_mask_to_parameter_indices(A_mask, idx, m, dtot)
            delete_cols.update(removed_idx)

    valid_src_cols = [c for c in src_cols if c not in delete_cols]

    if len(valid_src_cols) == 0:
        return 0.0

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
    d[bias_r != 0] += bias_mat[bias_r != 0]
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


