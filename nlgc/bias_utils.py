# Author: Behrad Soleimani <behrad@umd.edu>
# Author: Proloy Das <proloy@umd.edu>
"Deviance calculation"

import numpy as np
import warnings
from scipy import linalg

from .opt.proximal import calculate_ss


def _voxel_block(v, n_orient=3):
    """Return slice for RAS components of voxel v."""
    return slice(n_orient * v, n_orient * (v + 1))


def sample_path_bias(q, a, x_bar, A_mask, n_eigenmodes, n_orients):
    """Computes the bias in the deviance

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
    t, dxm = x_bar.shape
    _, dtot = a.shape
    
    p = dtot // dxm
    eff_eigenmodes = n_eigenmodes * n_orients

    bias = 0
    cx = np.zeros((t - p, dtot))
    dxn_voxels = dxm // n_orients

    for idx_v in range(dxn_voxels):
        block = _voxel_block(idx_v, n_orients)

        ai = a[block, :]
        xi = x_bar[p:, block]

        Q_block = q[block, block]
        Q_inv_block = np.linalg.inv(Q_block)

        for k in range(p):
            cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]

        res = xi - cx.dot(ai.T)

        # gradient of log - likelihood
        ldot = Q_inv_block.dot(res.T.dot(cx))

        # hessian of log - likelihood
        # ldotdot = -cx.T.dot(cx) / qd[block, block]
        Hx = cx.T.dot(cx) 
        ldotdot = -np.kron(Q_inv_block, Hx)

        ldot = ldot.reshape(-1)

        delete_idxs = []

        # Large voxel block containing all eigenmodes and all RAS orientations
        idx_large_voxel = idx_v // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes

        for idx in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = transition_mask_to_parameter_indices(A_mask, idx,
                                                               dxm, dtot)

            delete_idxs.extend(removed_idx)

        # Expand deletes across the 3 target RAS equations
        delete_idxs_3d = []

        for r in range(n_orients):
            for idx in delete_idxs:
                delete_idxs_3d.append(r * dtot + idx)

        delete_idxs_3d = sorted(set(delete_idxs_3d))

        if len(delete_idxs_3d) > 0:
            ldot = np.delete(ldot, delete_idxs_3d)
            ldotdot = np.delete(ldotdot, delete_idxs_3d, axis=0)
            ldotdot = np.delete(ldotdot, delete_idxs_3d, axis=1)

        bias += ldot.dot(np.linalg.solve(ldotdot, ldot))
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

    temp1 = s2.dot(ai)
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
        bias = n * ldot.dot(temp)
    except linalg.LinAlgError:
        warnings.warn('source-index {:d} ldotdot is not negative definite: '
                      'setting positive eigenvalues equal to zero, '
                      'result may not be accurate.'.format(idx_src), RuntimeWarning, stacklevel=2)
        e, v = linalg.eigh(-ldotdot)
        temp = v.dot(ldot)
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
        removed.extend(range(src, dtot, dxm))

    return np.asarray(removed)