import numpy as np
import warnings
from scipy import linalg

from .opt.proximal import calculate_ss


def compute_bias(em_state, smoother_result, config):
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    Q = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain
    
    bias = sample_path_bias(Q, A, x, P, B, A_mask, 
                            config.latent.n_eigenmodes, 
                            config.latent.n_orients, 
                            m, config.latent.order)
    return bias


def compute_bias_idx(source, em_state, smoother_result, config):
    p = config.latent.order
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    Q = em_state.Q[:m, :m]
    A_mask = em_state.A_mask[:m]
    x = smoother_result.smoothed_state
    P = smoother_result.smoothed_cov
    B = smoother_result.smoother_gain

    if isinstance(source, int):
        # Map region index to the actual row indices in the A matrix
        # e.g., Region 0 -> Rows [0, 1, 2]
        source = [source * config.latent.n_orients + i 
                  for i in range(config.latent.n_orients)]
        
    bias = sum([bias_by_idx(i, Q, A, x, P, B, m, p, A_mask) 
                for i in source])
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


def bias_by_idx(idx_src, q, a, x_bar, s_bar, b, m, p, A_mask=None):
    """Computes the bias in the deviance (proloy@umd.edu)

    Parameters
    ----------
    q:  ndarray of shape (n_sources*mo, n_sources*mo)
    a:  ndarray of shape (n_sources*mo, n_sources*order*mo)
    x_bar:  ndarray of shape (t, n_sources*mo)
    idx_src: source index

    Returns
    -------
    bias

    """
    warnings.filterwarnings('always')
    _, dtot = a.shape

    ### These uses the whole distribution.
    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)

    ai = a[idx_src]  # in python slicing returns 1d array, so transpose is meaningless.
    qi = q[idx_src, idx_src]

    ldot = np.empty((dtot))
    ldotdot = np.empty((dtot, dtot))

    temp1 = s2 @ ai
    temp1 -= s1[idx_src]

    ldot[:] = - temp1
    ldot[:] /= qi

    ldotdot[:, :] = - s2 / qi

    if A_mask is not None:
        removed_idx = transition_mask_to_parameter_indices(A_mask, idx_src, 
                                                           m, dtot)

        ldot = np.delete(ldot, removed_idx)
        ldotdot = np.delete(ldotdot, removed_idx, axis=0)
        ldotdot = np.delete(ldotdot, removed_idx, axis=1)

    try:
        c, low = linalg.cho_factor(-ldotdot)
        temp = linalg.cho_solve((c, low), ldot)
        bias = n * ldot @ temp
    except linalg.LinAlgError:
        warnings.warn('source-index {:d} ldotdot is not negative definite: '
                      'setting positive eigenvalues equal to zero, '
                      'result may not be accurate.'.format(idx_src), RuntimeWarning, stacklevel=2)
        e, v = linalg.eigh(-ldotdot)
        temp = v @ ldot
        idx = e > 0
        bias = np.sum(temp[idx] ** 2 / e[idx])
        bias *= n

    return bias


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
        # dxm and dtot must be integers for range() to work
        removed.extend(range(int(src), int(dtot), int(dxm)))

    return np.asarray(removed, dtype=int)