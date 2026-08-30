from scipy.spatial import cKDTree
from mne.forward import is_fixed_orient
from mne.inverse_sparse.mxne_inverse import _prepare_gain
from mne.source_space import SourceSpaces
from scipy import linalg
import numpy as np
import copy


def _triage_rank(rank):
    assert rank is not None, "rank must be provided! ICA silently "\
        "rank-reduces data so rank must be manually computed and passed to NLGC"
    
    assert isinstance(rank, int), "rank will be internally marshaled to dict, "\
        "pass in the integer rank"
    
    assert rank > 100, "rank too low, check preprocessing pipeline"


def _prep_surface_ss_eigs(info, forward, noise_cov, labels, n_eigenmodes, 
                        n_orients, prepargs):
    print('fixed orientation')
    forward, gain, gain_info, whitener, source_weighting, mask = \
        _prepare_gain(forward, info, noise_cov, **prepargs)
    
    eff_eigenmodes = n_orients * n_eigenmodes
    if n_orients != 1:
        assert n_eigenmodes == 1
        print("\nGot fixed orientation source space but with"
                f" n_orients = {n_orients}! "
                "Treating orientations as temporally coupled eigenmodes "
                "rather than independent eigenmodes (default).\n")
        
    weights, G, label_vertidx, src_flip, singular_values = \
        _reduce_lead_field_surface(forward['src'], labels, eff_eigenmodes, 
                                   data=gain.T)
    
    label_names = []
    
    for i, label in enumerate(labels):
        label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
   
    return G, label_names, label_vertidx, weights, gain_info, whitener, \
        src_flip, singular_values


def _prep_vol_ss_eigs(info, forward, noise_cov, labels, n_eigenmodes, 
                        n_orients, prepargs):
    print("volume")
    prepargs['loose'] = 1.0 # must be 1 for volume
    eff_eigenmodes = n_eigenmodes * n_orients

    forward, gain, gain_info, whitener, source_weighting, mask = \
        _prepare_gain(forward, info, noise_cov, **prepargs)
    
    weights, G, label_vertidx, singular_values = \
        _reduce_lead_field_vol(forward['src'], labels, eff_eigenmodes, 
                               data=gain.T)
    label_names = []
    for i, label in enumerate(labels):
        label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))

    return G, label_names, label_vertidx, weights, gain_info, whitener, None, \
           singular_values 


def _prep_mixed_ss_eigs(info, forwards, noise_cov, labels, n_eigenmodes, 
                        n_orients, prepargs):
    print("mixed source space (loose required but only applied to surface)")
    G = []
    ss = None
    eff_eigenmodes = n_eigenmodes * n_orients

    for fwd_idx, fwd in enumerate(forwards):
        if fwd_idx == 0:
            # this will error later too upon ss concatenation if there are some
            # shenanigans here
            assert fwd['src'][0]['type'] == 'surf', \
                'mixed source space forwards must have surface forward(s) first'
            
        prepargs = copy.deepcopy(prepargs)
        
        if not is_fixed_orient(fwd):
            prepargs['loose'] = 1.0 # must be 1 for volume
        else:
            assert prepargs['loose'] > 0.0

        forward, gain, gain_info, whitener, source_weighting, mask = \
            _prepare_gain(fwd, info, noise_cov, **prepargs)
        
        print(f"{gain.shape=}")
        G.append(gain)

        if ss is None:
            ss = fwd['src']
        else:
            ss += fwd['src']
    
    # (sensors, mixed sources)
    G = np.concatenate(G, axis=1)
    
    weights, G, label_vertidx, singular_values = \
        _reduce_lead_field_vol(ss, labels, eff_eigenmodes, 
                               data=G.T)
    label_names = []
    for i, label in enumerate(labels):
        label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))

    return G, label_names, label_vertidx, weights, gain_info, whitener, None, \
           singular_values


def prepare_eigenmodes(info, forward, noise_cov, labels, rank, n_eigenmodes=2, 
                       n_orients = 1, loose=0.0, depth=0.0, pca=True,
                       mode='svd_flip'):
    
    _triage_rank(rank)
    rank = {'mag': rank}

    assert isinstance(labels, SourceSpaces), "labels must be mne source space"

    depth_dict = {'exp': depth, 
                  'limit_depth_chs': 'whiten', 
                  'combine_xyz': 'fro', 
                  'limit': None}
        
    prepargs = {
        'pca': pca,
        'depth': depth_dict,
        'loose': loose,
        'rank': rank
    }
        
    if isinstance(forward, list):
        print('mixed source space')
        G, label_names, label_vertidx, weights, gain_info, whitener, src_flip, \
            singular_values = \
            _prep_mixed_ss_eigs(info, forward, noise_cov, labels, n_eigenmodes, 
                                n_orients, prepargs)

    elif is_fixed_orient(forward):
        print('fixed orientation source space')
        G, label_names, label_vertidx, weights, gain_info, whitener, src_flip, \
            singular_values = \
            _prep_surface_ss_eigs(info, forward, noise_cov, labels, 
                                  n_eigenmodes, n_orients, prepargs)
    else:
        print('volume source space')
        G, label_names, label_vertidx, weights, gain_info, whitener, src_flip, \
            singular_values = \
            _prep_vol_ss_eigs(info, forward, noise_cov, labels, n_eigenmodes, 
                              n_orients, prepargs)
        
    # test if there are empty columns
    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
    label_vertidx = [i for select, i in zip(sel, label_vertidx) if select]
    
    if not isinstance(forward, list) and is_fixed_orient(forward):
        src_flip = [i for select, i in zip(sel, src_flip) if select]

    discarded_labels = []
    j = 0
    eff_eigenmodes = n_eigenmodes * n_orients
    for i, sel_ in enumerate(sel[::eff_eigenmodes]):
        if not sel_:
            discarded_labels.append(labels.pop(i - j))
            label_vertidx.pop(i - j)
            j += 1
    assert j == len(discarded_labels)
    if j > 0:
        print('No sources were found in following {:d} ROIs:\n'\
                    .format(len(discarded_labels)) + \
                    '\n'.join(map(lambda x: str(x.name), discarded_labels)))

    return weights, G, label_vertidx, label_names, gain_info, whitener, \
           singular_values


def _reduce_lead_field_surface(fwd_src, src, n_eigenmodes, data=None):    
    grouped_vertidx_no_offset, grouped_vertidx, n_groups, n_verts = \
        _prepare_leadfield_reduction(src, fwd_src)
    
    group_eigenmodes = np.zeros((sum(n_groups) * n_eigenmodes,) + \
                                data.shape[1:], dtype=data.dtype)
    singular_values = np.zeros((sum(n_groups), n_eigenmodes))
    
    lhweights = []
    rhweights = []
    
    for i, (this_grouped_vertidx, this_grouped_vertidx_no_offset) in \
                enumerate(zip(grouped_vertidx, grouped_vertidx_no_offset)):
        
        eig_src_weights, svals, this_group_eigenmodes, percentage_explained = \
            _truncatedsvd(data[this_grouped_vertidx], n_eigenmodes)
        
        singular_values[i] = svals
        
        print(
            f"patch {i}\n"
            f"  vertices: {data[this_grouped_vertidx].shape[0]}\n"
            f"  eigenmodes: {n_eigenmodes}\n"
            f"  variance explained: {percentage_explained * 100:.3f}%"
        )
        
        group_eigenmodes[i * n_eigenmodes:(i + 1) * n_eigenmodes] = \
            this_group_eigenmodes
        
        if i < n_groups[0]:
            lhweights.append([eig_src_weights, this_grouped_vertidx_no_offset])
        else:
            rhweights.append([eig_src_weights, this_grouped_vertidx_no_offset])

    weights = [lhweights, rhweights]
    src_flips = [None] * sum(n_groups)

    # all eigenmodes have fine source contribution
    assert np.all(singular_values != 0)

    return weights, group_eigenmodes.T, grouped_vertidx, src_flips, \
           singular_values


def _reduce_lead_field_vol(fwd_ss, src, eff_eigenmodes, data):
    groups, coarse_rr = _prepare_leadfield_reduction_vol(src, fwd_ss)
 
    group_eigenmodes = np.zeros((len(groups) * eff_eigenmodes, 
                                 data.shape[-1]), dtype=data.dtype)
    singular_values = np.zeros((len(groups), eff_eigenmodes))

    
    print(f'Reduced leadfield shape is {group_eigenmodes.shape}')
    print(f"{data.shape=}")
    weights = []
    
    for coarse_idx, members in groups.items():
        idxs = np.empty(0, dtype=int)

        # forward source space is a list of sub-sourcespaces, one monolithic one
        # for vol source spaces, but we must loop over indices for mixed case
        for i in range(len(fwd_ss)):
            if len(members[i]) > 0:
                offset = int(np.sum([fwd_ss[j]['nuse'] for j in range(i)]))
                idxs = np.append(idxs, np.array(members[i]) + offset) 
            else:
                continue
        
        # valid since we require loose > 0 for mixed source space
        ras_idxs = np.concatenate([3 * idxs[:, None] + np.arange(3)], 
                                  axis=1).ravel()
        print(f"{ras_idxs.shape=}")
        
        subvoxels = data[ras_idxs]

        eig_src_weights, svals, this_group_eigenmodes, percentage_explained = \
            _truncatedsvd(subvoxels, eff_eigenmodes)
        
        singular_values[coarse_idx] = svals
        
        group_eigenmodes[coarse_idx * eff_eigenmodes:(coarse_idx + 1)\
                          * eff_eigenmodes] = this_group_eigenmodes
        
        print(
            f"patch {coarse_idx}: vertices {subvoxels.shape[0]} -> "
            f"{eff_eigenmodes} leadfield reduction explained " 
            f"{percentage_explained*100:.3f}% variance"
        )
        
        weights.append(eig_src_weights)

    print(singular_values)

    # all eigenmodes have fine source contribution
    assert np.all(singular_values != 0)

    return weights, group_eigenmodes.T, groups, singular_values


def _prepare_leadfield_reduction(src_target, src_origin):
    vertno_origin = [s['vertno'] for s in src_origin]
    vertno_target = [s['vertno'] for s in src_target]
    pinfo_target = [s['pinfo'] for s in src_target]
    n_verts = [s['nuse'] for s in src_origin]
    n_groups = [s['nuse'] for s in src_target]
    grouped_vertidx = []
    grouped_vertidx_no_offset = []
    
    for k, (this_vertno_target, this_pinfo_target, this_vertno_origin) in \
        enumerate(zip(vertno_target, pinfo_target, vertno_origin)):

        offset = 0 if k == 0 else n_verts[k - 1]
        for this_vert, this_pinfo in zip(this_vertno_target, this_pinfo_target):
            this_vertices = np.intersect1d(this_vertno_origin, this_pinfo)
            
            # offset ensures that rh indices are sequential with the lh indices,
            # but for indexing into the rh sources spaces object, we don't want
            # this offset since the indices overlap with the lh source spaces
            # indices. just create another list for this
            vertidx_no_offset = np.searchsorted(this_vertno_origin, 
                                                this_vertices)
            vertidx = offset + vertidx_no_offset

            if len(vertidx) == 0:
                vertidx = None
                vertidx_no_offset = None
            grouped_vertidx.append(vertidx)
            grouped_vertidx_no_offset.append(vertidx_no_offset)
            
    return grouped_vertidx_no_offset, grouped_vertidx, n_groups, n_verts
    

def _prepare_leadfield_reduction_vol(vol_target, src_origin):
    # fine and coarse coordinates in RAS
    # in case we use multiple spaces/labels
    coarse_rr = []
    for i in range(len(vol_target)):
        coarse_rr.extend(vol_target[i]['rr'][vol_target[i]['inuse'] > 0])

    tree = cKDTree(coarse_rr)

    groups = {i: {j : [] for j in range(len(src_origin))} 
              for i in range(len(coarse_rr))}

    idxs = []
    for i, s in enumerate(src_origin):
        
        src_rr = s['rr'][s['inuse'] > 0]

        _, src_idx = tree.query(src_rr)
        idxs.append(src_idx)

        for fine_idx, coarse_idx in enumerate(src_idx):
            groups[coarse_idx][i].append(fine_idx)

    return groups, coarse_rr


def _truncatedsvd(a, n_components=2):
    if n_components > min(*a.shape):
        raise ValueError('n_components={:d} should be smaller than '
                         'min({:d}, {:d})'.format(n_components, *a.shape))
    u, s, vh = linalg.svd(a, full_matrices=False, compute_uv=True,
                          overwrite_a=True, check_finite=True,
                          lapack_driver='gesdd')    

    percentage_explained = s[:n_components].sum() / s.sum()
    sv = s[:n_components]
    eigs = vh[:n_components]

    # no scaling by singular values
    return u, sv, eigs, percentage_explained
               
