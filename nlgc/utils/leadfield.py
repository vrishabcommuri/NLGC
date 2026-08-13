from scipy.spatial import cKDTree
from mne import (Forward, Label)
from mne.forward import is_fixed_orient
from mne.inverse_sparse.mxne_inverse import _prepare_gain
from mne.source_estimate import _prepare_label_extraction as \
    _prepare_label_extraction_mne
from mne.source_estimate import (_BaseVolSourceEstimate,
                                 _BaseVectorSourceEstimate,
                                 SourceEstimate,
                                 MixedSourceEstimate, 
                                 VolSourceEstimate)
from mne.source_space import SourceSpaces
from mne.utils import (logger, _check_option, _validate_type)
from scipy import linalg, sparse
import numpy as np
import copy


def prepare_eigenmodes(info, forward, noise_cov, labels, n_eigenmodes=2, 
                       n_orients = 1, loose=0.0, depth=0.0, pca=True, rank=None,
                       mode='svd_flip'):
    if not is_fixed_orient(forward):
        depth_dict = None
    else:
        depth_dict = {'exp': depth, 
                      'limit_depth_chs': 'whiten', 
                      'combine_xyz': 'fro', 
                      'limit': None}

    if not is_fixed_orient(forward) and loose == 0.0:
        print('Loose orientation must be set to 1.0 to be applied to free-orientation forward solutions, changing it to 1.0. If unsure set loose to auto')
        loose = 1.0

    forward, gain, gain_info, whitener, source_weighting, mask = \
        _prepare_gain(forward, info, noise_cov, pca, depth_dict, loose, rank)
    # whiten the data
    logger.info('Whitening data matrix.')
    print('check orientation')
    if not is_fixed_orient(forward):
        if n_orients <= 1:
            raise ValueError('Number of orientations is less than or equal to 1 for not fixed orientation forward. Please use accurate number of orientations')
        print('The lead field is not fixed-orientation using mixed source space method')
        if isinstance(labels, Forward):
            weights, G, label_vertidx = \
                _reduce_lead_field_vol(forward, labels, n_eigenmodes, 
                                       n_orients, data=gain.T)
            label_names = []
            for i, label in enumerate(labels['src']):
                label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
        elif isinstance(labels, SourceSpaces):
            weights, G, label_vertidx = \
                _reduce_lead_field_vol(forward, labels, n_eigenmodes, 
                                       n_orients, data=gain.T)
            label_names = []
            for i, label in enumerate(labels):
                label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
        else:
            raise ValueError('Not supported {:s}: labels are expected to be either an mne.SourceSpace or'
                             'mne.Forward object.'.format(labels))
    else:
        eff_eigenmodes = n_orients * n_eigenmodes
        if n_orients != 1:
            assert n_eigenmodes == 1
            print("\nGot fixed orientation source space but with"
                  f" n_orients = {n_orients}! "
                  "Treating orientations as temporally coupled eigenmodes "
                  "rather than independent eigenmodes (default).\n")
            
        print('fixed orientation')
        if isinstance(labels, Forward):
            weights, G, label_vertidx, src_flip = \
                _reduce_lead_field(forward, labels, eff_eigenmodes, data=gain.T)
            label_names = []
            for i, label in enumerate(labels['src']):
                label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
        elif isinstance(labels, SourceSpaces):
            weights, G, label_vertidx, src_flip = \
                _reduce_lead_field(forward, labels, eff_eigenmodes, data=gain.T)
            label_names = []
            for i, label in enumerate(labels):
                label_names.extend(map(lambda x: f'{i}-{x}', label['vertno']))
        elif isinstance(labels, list):
            if isinstance(labels[0], Label):
                weights = None # not implemented
                G, label_vertidx, src_flip = \
                    _extract_label_eigenmodes(forward, labels, gain.T, mode, 
                                              eff_eigenmodes, allow_empty=True)
                
                label_names = [label.name for label in labels]
            else:
                raise ValueError('Not supported {:s}: elements of labels are expected to be mne.Labels, '
                                'if a list is provided.'.format(type(labels[0])))
        else:
            raise ValueError('Not supported {:s}: labels are expected to be either an mne.SourceSpace or'
                            'mne.Forward object or list of mne.Labels.'.format(labels))
        

    # test if there are empty columns
    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
    label_vertidx = [i for select, i in zip(sel, label_vertidx) if select]
    if is_fixed_orient(forward):
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
        logger.info('No sources were found in following {:d} ROIs:\n'\
                    .format(len(discarded_labels)) + \
                    '\n'.join(map(lambda x: str(x.name), discarded_labels)))

    return weights, G, label_vertidx, label_names, gain_info, whitener


def _reduce_lead_field(forward, src, n_eigenmodes, data=None):
    import mne
    if data is None:
        logger.info('Using the raw forward solution')
        data = np.swapaxes(forward['sol']['data'], 0, 1)  # (n_sources, n_channels)
    data = data.copy()
    print(f'Data shape is {data.shape}')
    if isinstance(src, mne.Forward):
        src = src['src']

    grouped_vertidx_no_offset, grouped_vertidx, n_groups, n_verts = \
        _prepare_leadfield_reduction(src, forward['src'])
    group_eigenmodes = np.zeros((sum(n_groups) * n_eigenmodes,) + \
                                data.shape[1:], dtype=data.dtype)
    
    lhweights = []
    rhweights = []
    
    for i, (this_grouped_vertidx, this_grouped_vertidx_no_offset) in \
                enumerate(zip(grouped_vertidx, grouped_vertidx_no_offset)):
        eig_src_weights, this_group_eigenmodes, percentage_explained = \
            _truncatedsvd(data[this_grouped_vertidx], n_eigenmodes, 
                          return_pecentage_explained=True)
        
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
        print(this_group_eigenmodes.shape)
    weights = [lhweights, rhweights]
    src_flips = [None] * sum(n_groups)
    return weights, group_eigenmodes.T, grouped_vertidx, src_flips


def _reduce_lead_field_vol(forward, src, n_eigenmodes, n_orients, data=None):
    import mne
    if data is None:
        print('data is None')
        logger.info('Using the raw forward solution')
        data = np.swapaxes(forward['sol']['data'], 0, 1) 
    print(f'Data shape is {data.shape}')
    if isinstance(src, mne.Forward):
        src = src['src']


    print(f'Data Reshaped shape is {data.shape}')
    print(src)
    print(forward['src'])
    groups, coarse_rr = _prepare_leadfield_reduction_vol(src, forward['src'])
 
    group_eigenmodes = np.zeros((len(groups)*n_eigenmodes * n_orients, 
                                 data.shape[-1]), dtype=data.dtype)
    print(f'Group Eigenmodes Shape {group_eigenmodes.shape}')
    
    weights = []
    
    for coarse_idx, members in groups.items():
        print(f'coarse_idx: {coarse_idx}, and members: {members}')
        idxs = np.empty(0, dtype=int)
        for i in range(len(forward['src'])):
            if len(members[i]) > 0:
                idxs = np.append(idxs, np.array(members[i]) + \
                    int(np.sum([forward['src'][j]['nuse'] 
                                for j in range(i)])))
            else:
                continue
        ras_idxs = np.concatenate([3 * idxs[:, None] + np.arange(3)], 
                                  axis=1).ravel()
        subvoxels = data[ras_idxs]
        print(f'subvoxel shape: {subvoxels.shape}')
        eig_src_weights, this_group_eigenmodes, percentage_explained = \
            _truncatedsvd_vol(subvoxels, n_eigenmodes, n_orients, 
                              return_pecentage_explained=True)
        
        print(f'this_group_eigenmodes shape {this_group_eigenmodes.shape}')
        group_eigenmodes[coarse_idx * n_eigenmodes * n_orients:(coarse_idx + 1)\
                          * n_eigenmodes * n_orients] = this_group_eigenmodes
        
        print(
            f"patch {coarse_idx}: vertices {subvoxels.shape[0]} -> "
            f"{n_eigenmodes} leadfield reduction explained" 
            f"{percentage_explained*100:.3f}% variance"
        )
        
        weights.append(eig_src_weights)

    return weights, group_eigenmodes.T, groups



def _prepare_label_extraction(labels, src):
    vertno = [s['vertno'] for s in src]
    label_vertidx = []
    for label in labels:
        if label.hemi == 'lh':
            this_vertices = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertices)
        elif label.hemi == 'rh':
            this_vertices = np.intersect1d(vertno[1], label.vertices)
            vertidx = len(vertno[0]) + np.searchsorted(vertno[1], this_vertices)
        if len(vertidx) == 0:
            vertidx = None
        label_vertidx.append(vertidx)
    return label_vertidx


def assign_labels(labels, src_target, src_origin, thresh=0):
    """Assign the patch indices of the corresponding labels from origin into 
    target source space

    This function returns the patch indices of the (ROI) labels in the target 
    source space (e.g. 'ico-1') from the origin source space (e.g. 'ico-4')

    Parameters
    ----------
    labels:  mne.Labels | mne.Label
        labels in standard MNE-python format
    src_target: mne.SourceSpaces
        target source space, e.g. ico-4
    src_origin: mne.SourceSpaces
        origin source space, e.g. ico-4

    Returns
    -------
    label_vertidx: list
        vertex(patch) index
    """
    label_vertidx_origin = _prepare_label_extraction(labels, src_origin)
    _, group_vertidx, _, _ = _prepare_leadfield_reduction(src_target, 
                                                          src_origin)
    label_vertidx = []
    for this_label_vertidx_origin in label_vertidx_origin:
        this_label_vertidx = []
        for i, this_group_vertidx in enumerate(group_vertidx):
            this_vertices = np.intersect1d(this_group_vertidx, 
                                           this_label_vertidx_origin)
            if len(this_vertices) > thresh:
                this_label_vertidx.append(i)
        this_label_vertidx = np.asanyarray(this_label_vertidx)
        label_vertidx.append(this_label_vertidx)
    return label_vertidx


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
    coarse_rr = vol_target[0]['rr'][vol_target[0]['inuse'] > 0]
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
    

def _extract_label_eigenmodes(fwd, labels, data=None, mode='mean', 
                              n_eigenmodes=2, allow_empty=False, trans=None, 
                              mri_resolution=True):
    "Zero columns corresponds to empty labels"
    src = fwd['src']
    _validate_type(src, SourceSpaces)
    _check_option('mode', mode, ['svd', 'svd_flip'] + ['auto'])
    func = _svd_funcs[mode]

    if len(src) > 2:
        if src[0]['type'] != 'surf' or src[1]['type'] != 'surf':
            raise ValueError('The first 2 source spaces have to be surf type')
        if any(np.any(s['type'] != 'vol') for s in src[2:]):
            raise ValueError('source spaces have to be of vol type')

        n_aparc = len(labels)
        n_aseg = len(src[2:])
        n_labels = n_aparc + n_aseg
    else:
        n_labels = len(labels)

    # create a dummy stc
    kind = src.kind
    vertno = [s['vertno'] for s in src]
    nvert = np.array([len(v) for v in vertno])
    if kind == 'surface':
        stc = SourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    elif kind == 'mixed':
        stc = MixedSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    else:
        stc = VolSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    stcs = [stc]

    vertno = None
    for si, stc in enumerate(stcs):
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = \
                _prepare_label_extraction_mne(stc, labels, src, 
                                              mode.replace('svd', 'mean'),
                                              allow_empty)
        if isinstance(stc, (_BaseVolSourceEstimate,
                            _BaseVectorSourceEstimate)):
            _check_option(
                'mode', mode, ('svd',),
                'when using a volume or mixed source space')
            mode = 'svd' if mode == 'auto' else mode
        else:
            mode = 'svd_flip' if mode == 'auto' else mode

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        if data is None:
            logger.info('Using the raw forward solution')
            data = np.swapaxes(fwd['sol']['data'], 0, 1)  # (n_sources, n_channels)
        data = data.copy()

        # do the extraction
        label_eigenmodes = np.zeros((n_labels * n_eigenmodes,) + data.shape[1:], 
                                    dtype=data.dtype)
        for i, (vertidx, flip, label) in enumerate(zip(label_vertidx, src_flip, 
                                                       labels)):
            if vertidx is not None:
                if isinstance(vertidx, sparse.csr_matrix):
                    assert mri_resolution
                    assert vertidx.shape[1] == data.shape[0]
                    this_data = np.reshape(data, (data.shape[0], -1))
                    this_data = vertidx * this_data
                    this_data.shape = \
                        (this_data.shape[0],) + stc.data.shape[1:]
                else:
                    this_data = data[vertidx]
                label_eigenmodes[i * n_eigenmodes:(i + 1) * n_eigenmodes] = \
                    func(flip, this_data, n_eigenmodes)

        return label_eigenmodes.T, label_vertidx, src_flip


def _truncatedsvd_vol(a, n_components=2, n_orients = 3,
                      return_pecentage_explained=False):
    n_total, n_sensors = a.shape
    n_voxels = n_total // n_orients

    # The spatial weighting comes from an SVD over voxels, so at most n_voxels
    # components exist -- not min(*a.shape), which the old bound assumed.
    if n_components > n_voxels:
        raise ValueError('n_components={:d} should not exceed the number of voxels '
                         'in the patch ({:d})'.format(n_components, n_voxels))

    # (n_voxels, n_orients, n_sensors): each row is one voxel's full response
    a3 = a.reshape(n_voxels, n_orients, n_sensors)
    u, s, _ = linalg.svd(a3.reshape(n_voxels, n_orients * n_sensors),
                         full_matrices=False, compute_uv=True,
                         check_finite=True, lapack_driver='gesdd')

    modes = np.empty((n_components * n_orients, n_sensors), dtype=a.dtype)
    for m in range(n_components):
        for r in range(n_orients):
            modes[m * n_orients + r] = u[:, m] @ a3[:, r, :]

    if return_pecentage_explained:
        return u, modes, s[:n_components].sum() / s.sum()
    return u, modes

def _truncatedsvd(a, n_components=2, return_pecentage_explained=False):
    if n_components > min(*a.shape):
        raise ValueError('n_components={:d} should be smaller than '
                         'min({:d}, {:d})'.format(n_components, *a.shape))
    u, s, vh = linalg.svd(a, full_matrices=False, compute_uv=True,
                          overwrite_a=True, check_finite=True,
                          lapack_driver='gesdd')    

    if return_pecentage_explained:
        return u, vh[:n_components] * s[:n_components][:, None], \
               s[:n_components].sum() / s.sum()
    return u, vh[:n_components] * s[:n_components][:, None]


_svd_funcs = {
    'svd_flip': lambda flip, data, n_components: \
        _truncatedsvd(flip * data, n_components),
    'svd': lambda flip, data, n_components: _truncatedsvd(data, n_components)
}