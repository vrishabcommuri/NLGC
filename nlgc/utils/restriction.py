import itertools
import numpy as np


def roi_to_link_restriction(ROIs, sparsity, eff_eigenmodes, config):
    """Candidate links as (target, source) ROI index pairs."""
    if config.numerical.verbose:
        print(f"nonzero sparsity entries: {np.count_nonzero(sparsity)}")
        print(f"{len(ROIs)=}")
        print(f"{config.sparsity.sparsity_factor=}")
    links_to_check = []
    for src, targ in itertools.product(ROIs, repeat=2):
        # Exclude i == j cases
        if src == targ:
            continue

        if sparsity[targ, src] <= config.sparsity.sparsity_factor * sparsity[targ, src]:
            continue

        links_to_check.append((targ, src))
    return links_to_check


def link_to_A_mask(targ, src, em_state, config):
    """
    Expand (target, source) ROI pairs into A-matrix zeroed indices.

    Tuple order is (target, source) throughout
    """
    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    p = config.latent.order
    m = em_state.N_sources_upper

    eff_eigenmodes = n_eigenmodes * n_orients

    targ_block = slice(targ * eff_eigenmodes, (targ + 1) * eff_eigenmodes)

    src_cols = []
    for lag in range(p):
        src_cols.extend(range(src * eff_eigenmodes + lag * m, (src + 1) * eff_eigenmodes + lag * m))

    A_mask = np.ones_like(em_state.A)

    A_mask[targ_block, :][:, src_cols] = 0.0

    return A_mask


def mask_to_zeroed_index(A_mask):
    """
    Convert an A_mask into indices of coefficients to be forced to zero.

    Parameters
    ----------
    A_mask : ndarray, shape (m, m*p)
        Boolean or binary mask. False entries indicate coefficients to zero.

    Returns
    -------
    tuple of lists
        Tuple containing row and column indices:
        ([row1, row2, ...], [col1, col2, ...])
    """
    zero_rows, zero_cols = np.where(~A_mask.astype(bool))
    return (zero_rows.tolist(), zero_cols.tolist())
