import re
import itertools
import jax.numpy as jnp
import numpy as np


def roi_to_link_restriction(ROIs, sparsity, eff_eigenmodes, config):
    """Candidate links as (target, source) ROI index pairs."""
    if config.numerical.verbose:
        print(f"nonzero sparsity entries: {np.count_nonzero(sparsity)}")
        print(f"{len(ROIs)=}")
        print(f"{config.sparsity.sparsity_factor=}")
    links_to_check = []
    for i, j in itertools.product(ROIs, repeat=2):
        # Exclude i == j cases
        if i == j:
            continue

        if sparsity[j, i] <= config.sparsity.sparsity_factor * sparsity[j, j]:
            continue

        links_to_check.append((j, i))
    return links_to_check


def restriction_to_zeroed_index(restriction, m, p):
    """
    expand zero indices of i->j restriction across lags
    A[zeroed_index] = 0.0 will zero all lags for i->j
    for A in companion or raveled form
    """
    if restriction is not None:
        i_s, j_s = re.split(r'->', restriction)
        i_s = [int(i) for i in re.split(r',', i_s)]
        j_s = [int(j) for j in re.split(r',', j_s)]
        # check for i, j's proper range
        if any(i >= m or i < 0 for i in i_s) or \
                any(j >= m or j < 0 for j in j_s):
            raise ValueError(f"restriction {restriction}: i or j needs to be in range of neural sources, {m}")
        x_index = []
        y_index = []
        for i, j in itertools.product(i_s, j_s):
            x_index.extend([j] * p)
            y_index.extend(list(range(i, m * p, m)))
        zeroed_index = (x_index, y_index)
    else:
        zeroed_index = None
    
    return zeroed_index


#-------------------------------------------------------------------------------
# EM-related operations (zeroed index -> A masks)
# jax accelerated and python versions
#-------------------------------------------------------------------------------

def link_tuples_to_zero_indices(links_to_check, em_state, config):
    """Expand (target, source) ROI pairs into A-matrix zeroed indices.

    Tuple order is (target, source) throughout: roi_to_link_restriction emits
    pairs that way after testing sparsity[target, source], and batched_test_links
    stores results as dev_raw[target, source]. Each entry zeroes the block
    A[target_rows, source_cols] across all lags, i.e. removes source -> target.
    """
    m = em_state.N_sources_upper
    p = config.latent.order
    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    eff_eigenmodes = n_eigenmodes * n_orients

    zeroed_indices = []

    for targ, src in links_to_check:
        target = _expand_roi_indices_as_tup(targ, eff_eigenmodes)
        source = _expand_roi_indices_as_tup(src, eff_eigenmodes)
        link = '->'.join(map(lambda x: ','.join(map(str, x)), (source, target)))

        zeroed_indices.append(restriction_to_zeroed_index(link, m, p))

    return zeroed_indices


def _expand_roi_indices_as_tup(reg_idx, emod, n_orients = 1):
    eff_emod = emod* n_orients
    return tuple(range(reg_idx * eff_emod, reg_idx * eff_emod + eff_emod))


def jax_expand_zeroindex_masks(zeroed_indices, em_state):
    """
    applies each zeroed index to a mask of the A matrix from the full model.
    this sets up for vmap which will apply the masks to the A matrices in
    parallel.
    """

    A_mask = jnp.ones_like(em_state.A)
    A_masks = jnp.repeat(A_mask[jnp.newaxis, :, :], len(zeroed_indices), axis=0)

    for k, (rows, cols) in enumerate(zeroed_indices):
        rows = jnp.array(rows)
        cols = jnp.array(cols)

        A_masks = A_masks.at[
            k,
            rows[:, None],
            cols[None, :]
        ].set(0.0)

    return A_masks


def expand_zeroindex_masks(zeroed_indices, em_state):
    """
    applies each zeroed index to a mask of the A matrix from the full model.
    this sets up for vmap which will apply the masks to the A matrices in
    parallel.
    """

    A_mask = np.ones_like(em_state.A)
    A_masks = np.repeat(A_mask[np.newaxis, :, :], len(zeroed_indices), axis=0)

    for k, (rows, cols) in enumerate(zeroed_indices):
        rows = np.array(rows)
        cols = np.array(cols)

        A_masks[
            k,
            rows[:, None],
            cols[None, :]
        ] = 0.0

    return A_masks