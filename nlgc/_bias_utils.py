# Author: Behrad Soleimani <behrad@umd.edu>
# Author: Proloy Das <proloy@umd.edu>
"Deviance calculation"

import numpy as np
import warnings
from scipy import linalg

from .opt.m_step import calculate_ss


def _voxel_block(v, n_orient=3):
    """Return slice for RAS components of voxel v."""
    return slice(n_orient * v, n_orient * (v + 1))


def sample_path_bias(q, a, x_bar, zeroed_index, n_eigenmodes, n_orients):
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
        print(f'dtot {dtot}')
        print(f'dxm {dxm}')
        print(f'ai shape {ai.shape}')
        xi = x_bar[p:, block]
        print(f'xi shape {xi.shape}')
        print(f'cx shape {cx.shape}')
        Q_block = q[block, block]
        Q_inv_block = np.linalg.inv(Q_block)

        for k in range(p):
            cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]


        res = xi - cx.dot(ai.T)
        print(f'res shape is {res.shape}')
        # gradient of log - likelihood
        ldot = Q_inv_block.dot(res.T.dot(cx))

        # hessian of log - likelihood
        # ldotdot = -cx.T.dot(cx) / qd[block, block]
        Hx = cx.T.dot(cx) 
        ldotdot = -np.kron(Q_inv_block, Hx)

        ldot = ldot.reshape(-1)
        # if zeroed_index is not None:
        #     x_index, y_index = zeroed_index
        #     if idx_src in x_index:
        #         removed_idx = list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])
        #         ldot = np.delete(ldot, removed_idx)
        #         ldotdot = np.delete(ldotdot, removed_idx, axis=0)
        #         ldotdot = np.delete(ldotdot, removed_idx, axis=1)

        # FIX removing cross-talk components (that forced to be zero)
        # for l in range(0, dxm, eff_eigenmodes):
        #     for u in range(eff_eigenmodes):
        #         for v in range(eff_eigenmodes):
        #             if v != u and idx_src == l + v:
        #                 removed_idx = list(range(l + u, dtot, dxm))
        #                 print(f"removed_idx = {removed_idx}")
        #                 if zeroed_index is not None:
        #                     print(f"zeroed_index = {zeroed_index}")
        #                     x_index, y_index = zeroed_index
        #                     if idx_src in x_index:
        #                         removed_idx.extend(list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src]))
        #                     print(f"extended removed_idx = {removed_idx}")
        #                 ldot = np.delete(ldot, removed_idx)
        #                 print(f"ldot.shape after delete = {ldot.shape}")
        #                 ldotdot = np.delete(ldotdot, removed_idx, axis=0)
        #                 print(f"ldotdot.shape after delete1 = {ldotdot.shape}")
        #                 ldotdot = np.delete(ldotdot, removed_idx, axis=1)
        #                 print(f"ldotdot.shape after delete2 = {ldotdot.shape}")

        # delete_idxs = []
        # for l in range(0, dxm, eff_eigenmodes):
        #     for u in range(eff_eigenmodes):
        #         for v in range(eff_eigenmodes):
        #             if v != u and idx_v == l + v:
        #                 removed_idx = list(range(l + u, dtot, dxm))
        #                 if zeroed_index is not None:
        #                     x_index, y_index = zeroed_index
        #                     if idx_v in x_index:
        #                         removed_idx.extend(list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_v]))
        #                 delete_idxs.extend(removed_idx)
        
        # ldot = np.delete(ldot, delete_idxs)
        # ldotdot = np.delete(ldotdot, delete_idxs, axis=0)
        # ldotdot = np.delete(ldotdot, delete_idxs, axis=1)

        # bias += ldot.dot(np.linalg.solve(ldotdot, ldot))



        delete_idxs = []

        # Large voxel block containing all eigenmodes and all RAS orientations
        idx_large_voxel = idx_v // n_eigenmodes
        voxel_start = idx_large_voxel * eff_eigenmodes

        for l in range(voxel_start, voxel_start + eff_eigenmodes):
            removed_idx = list(range(l, dtot, dxm))

            if zeroed_index is not None:
                x_index, y_index = zeroed_index
                x_index = np.asarray(x_index)
                y_index = np.asarray(y_index)

                if l in x_index:
                    removed_idx.extend(list(y_index[x_index == l]))

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


def bias_by_idx(idx_src, q, a, x_bar, s_bar, b, m, p, zeroed_index=None):
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

    if zeroed_index is not None:
        x_index, y_index = zeroed_index
        if idx_src in x_index:
            removed_idx = list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])
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
