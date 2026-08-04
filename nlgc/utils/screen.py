import numpy as np
from nlgc.bias_utils import wald_by_idx
from scipy.stats import chi2


def sparsity_screen(em_state, smoother_result, ROIs, config):
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    smoothed_state = smoother_result.smoothed_state
    order = config.latent.order
    n_orients = config.latent.n_orients
    n_eigenmodes = config.latent.n_eigenmodes
    eff_eigenmodes = n_orients * n_eigenmodes
    energy_thresh = config.sparsity.negligible_candidate_link_energy_thr

    N = m // eff_eigenmodes

    A_blocks = A.reshape(N, eff_eigenmodes, order, N, eff_eigenmodes)

    block_strength = np.sqrt(np.sum(A_blocks**2, axis=(1,2,4)))

    # don't include self-links in strength calculation
    block_strength = block_strength * (~np.eye(N).astype(bool)).astype(float)

    if config.sparsity.negligible_candidate_link_energy_thr < 1 and \
            block_strength.sum() > 0:
        link_power = block_strength.ravel() ** 2

        sorted_idx = np.argsort(link_power)[::-1]

        cumul_power = np.cumsum(link_power[sorted_idx])
        cumul_power /= cumul_power[-1]

        idx = np.searchsorted(cumul_power, energy_thresh)

        keep_idx = sorted_idx[:idx + 1]

        sparsity_mask = np.zeros_like(link_power, dtype=bool)
        sparsity_mask[keep_idx] = 1.0

        sparsity_mask = sparsity_mask.reshape(N, N)
        if config.numerical.verbose:
            print(f"retained {np.count_nonzero(sparsity_mask)}/"
                  f"{np.count_nonzero(block_strength)} candidate "
                  "links (dropped lowest  "
                  f"{(1-energy_thresh)*100:.5f}% of total off diag energy)")
            
        sparsity = block_strength * sparsity_mask

        np.count_nonzero(sparsity), np.count_nonzero(sparsity_mask)

        assert np.count_nonzero(sparsity) == \
               np.count_nonzero(sparsity_mask)
    else:
        sparsity = block_strength

    if config.sparsity.var_thr < 1:
        x = smoothed_state[:, :em_state.N_sources_upper]

        # energy of each latent state
        state_power = np.sum(x**2, axis=0)

        # group all eigenmodes/orientations belonging to one ROI
        N_roi = em_state.N_sources_upper // eff_eigenmodes

        roi_power = state_power.reshape(N_roi, eff_eigenmodes).sum(axis=1)

        sorted_idx = np.argsort(roi_power)[::-1]

        cumul_power = np.cumsum(roi_power[sorted_idx])
        cumul_power /= cumul_power[-1]

        idx = np.searchsorted(cumul_power, config.sparsity.var_thr)

        ROIs = sorted_idx[:idx + 1]

    return sparsity, ROIs


def wald_screen(links_to_check, em_state, smoother_result, config):
    wald_scores = []

    target_alpha = 0.05
    
    surviving_links = []
    screened_links_to_check = []

    if config.numerical.verbose:
        print(f"screening {len(links_to_check)} links at alpha={target_alpha}")

    for idx, (targ, src) in enumerate(links_to_check):
        curr_wald, k_dof = wald_by_idx(src, targ, em_state.Q, em_state.A, 
                                       smoother_result.smoothed_state, 
                                       smoother_result.smoothed_cov, 
                                       smoother_result.smoother_gain, 
                                       em_state.A_mask, 
                                       config.latent.n_eigenmodes, 
                                       config.latent.n_orients, 
                                       em_state.N_sources_upper, 
                                       config.latent.order)
        
        wald_scores.append((curr_wald, targ, src))

        # calculate p-value of this Wald score 
        p_val = chi2.sf(curr_wald, df=k_dof)

        if config.numerical.verbose:
            print(f"({idx}/{len(links_to_check)}) "
                  f"screening {src}->{targ}: Score={curr_wald:}")
        
        if p_val < target_alpha:
            surviving_links.append((curr_wald, targ, src))

            screened_links_to_check.append((targ, src))

    if config.numerical.verbose:
        print(f"\nscreening complete: {len(surviving_links)} out of"
              f" {len(links_to_check)} links survived Wald screening.")

        print("\n--- retained link Wald scores ---")
        for score, targ, src in surviving_links:
            print(f"{src}->{targ}: {score:.2f}")

    return screened_links_to_check