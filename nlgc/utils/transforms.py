import numpy as np
import mne

def get_region_vec_ori(hemi, region, w, src_origin, neig):
    # apply the weights for the current patch to all of the ico-4 vectors in that same patch
    # (n_curr_patch_eigs, n_ico4_sources) x (n_ico4_sources, RAS) = (n_curr_patch_eigs, RAS)
    return w[hemi][region][0][:, :neig].T @ src_origin[hemi]['nn'][src_origin[hemi]['vertno']][w[hemi][region][1]]

def imbue_vector_direction(src_target, src_origin, weights, neig):
    # data: (n eig groups, 3)
    # vertnos: (2, 42), axis 0 indexes hemisphere 
    # vertloc: (n_eigenmodes)
    vec_src = {'data': [], 'vertnos': [], 'vertloc': []}

    vec_src['vertnos'] = [src_target[0]['vertno'], src_target[1]['vertno']]
    vec_src['vertloc'] = [src_target[0]['rr'][src_target[0]['vertno']], 
                          src_target[1]['rr'][src_target[1]['vertno']]]
    for hemi in range(2):
        for region in range(42):
            ori = get_region_vec_ori(hemi, region, weights, src_origin, neig)
            assert(len(ori) == neig)
            vec_src['data'].extend(ori)
    vec_src['data'] = np.array(vec_src['data'])
    
    return vec_src


def surface_eigs_to_vector_eigs(data, src_target, src_origin, weights, neigs, model):
    """
    data: array of shape (n_eigenmodes * n_sources, ...)
    weights: list with elements [[left_hemi_transform, left_hemi_vertices], [right_hemi_transform, right_hemi_vertices]]
        where transform is a matrix of shape (n_eigs, n_ico4_sources)
    returns array of shape (n_sources, RAS, ...) 
    where are RAS are right, anterior, superior projections
    """
    # generate RAS orientations for all eigenmodes
    vec_src = imbue_vector_direction(src_target, src_origin, weights, neigs)

    patch_vec_ts = []
    for pi in range(84):
        if hasattr(data, '_model_f'):
            # get the eigenmode time series for the current patch
            ei = data.model_f[0]._parameters[4][:, :neigs*84].T\
                   [pi*neigs:(pi+1)*neigs] # (n_eigs, n_time)
        else:
            ei = data[pi*neigs:(pi+1)*neigs] # (n_eigs, n_time)
        
        ev = vec_src['data'][pi*neigs:(pi+1)*neigs] # (n_eigs, RAS)
        patch_vec_ts.append(ev.T @ ei) # (RAS, n_time)

    return np.array(patch_vec_ts) # (n_sources, RAS, n_time)


def surface_eigs_to_surface_ico4(data, weights, neigs, model):
    """
    data: array of shape (n_eigenmodes * n_sources, ...)
    weights: list with elements [[left_hemi_transform, left_hemi_vertices], [right_hemi_transform, right_hemi_vertices]]
        where transform is a matrix of shape (n_eigs, n_ico4_sources)
    returns array of shape (n_ico4_sources, ...) 
    """
    patch_vec_ts = []
    for hemi in range(2):
        for region in range(42):
            pi = (hemi * 42) + 42
            if hasattr(data, '_model_f'):
                # get the eigenmode time series for the current patch
                ei = data.model_f[0]._parameters[4][:, :neigs*84].T\
                    [pi*neigs:(pi+1)*neigs] # (n_eigs, n_time)
            else:
                ei = data[pi*neigs:(pi+1)*neigs] # (n_eigs, n_time)

            ev = weights[hemi][region][0][:, :neigs].T # (n_eigs, n_ico4_sources)
        
            patch_vec_ts.append(ev.T @ ei) # (n_ico4_sources, n_time)

    return np.array(patch_vec_ts) # (n_ico4_sources, n_time)


def surface_ico4_to_surface_eigs(data, weights, neigs):
    """
    data: array of shape (n_ico4_sources, ...)
    weights: list with elements [[left_hemi_transform, left_hemi_vertices], [right_hemi_transform, right_hemi_vertices]]
        where transform is a matrix of shape (n_eigs, n_ico4_sources)
    returns array of shape (n_ico4_sources, ...) 
    """
    n_total_rows = neigs * 84
    n_time = data.data.shape[1]
    _x = np.zeros((n_time, n_total_rows), dtype=np.float64)
    col_idx = 0

    for hemi in range(2):
        for region in range(42):
            ev = weights[hemi][region][0][:, :neigs] # (n_ico4_sources, n_eigs)
            assert(isinstance(data, mne.SourceEstimate))
            if hemi == 0:
                hemidata = data.data[:5124//2]
            else:
                hemidata = data.data[5124//2:]

            patchdata = hemidata[weights[hemi][region][1]].T # (n_time, n_ico4_sources)
            n_cols = ev.shape[1]
            _x[:,  col_idx:col_idx + n_cols] = patchdata @ ev
            col_idx += n_cols

    assert(_x.flags['C_CONTIGUOUS'])
    return _x # (n_time, n_eigs * n_sources)