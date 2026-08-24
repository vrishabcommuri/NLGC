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
    if not config.sparsity.use_wald_screen:
        if config.numerical.verbose:
            print(f"Wald screening disabled; testing all "
                  f"{len(links_to_check)} candidate links")
        return links_to_check

    if len(links_to_check) == 0:
        return links_to_check

    alpha = config.sparsity.wald_screen_alpha

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"wald_screen_alpha must be in (0, 1), got {alpha}")

    if config.numerical.verbose:
        method = ("empirical null" if config.sparsity.use_empirical_null
                  else "chi2(k_dof)")
        print(f"screening {len(links_to_check)} links at alpha={alpha} "
              f"against the {method}")

    use_empirical_null = config.sparsity.use_empirical_null

    # Score every candidate. The empirical-null threshold is a quantile of the
    # whole score distribution, so it can only be set once all scores are in --
    # hence scoring and deciding are separate passes. The chi2 path could decide
    # inline, but shares this loop so both methods see identical scores.
    wald_scores = []
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

        wald_scores.append((curr_wald, targ, src, k_dof))

        if config.numerical.verbose:
            print(f"({idx}/{len(links_to_check)}) "
                  f"screening {src}->{targ}: Score={curr_wald:}")

    scores = np.array([s for s, _, _, _ in wald_scores])

    if use_empirical_null:
        # Fit the null, turn alpha into an absolute score cut, threshold.
        # Strictly greater than, so a degenerate all-equal distribution (e.g.
        # every score 0) keeps nothing rather than everything.
        null_fit = _fit_empirical_null(scores)

        if null_fit is None:
            # no usable spread in the scores -- nothing is distinguishable from
            # the null, so keep nothing rather than silently passing everything
            # through
            threshold = np.inf
        else:
            c, k = null_fit
            threshold = c * chi2.isf(alpha, k)

        surviving_links = [(s, targ, src) for s, targ, src, _ in wald_scores
                           if s > threshold]
    else:
        # Original per-link test against the nominal central chi2. k_dof varies
        # per link, since wald_by_idx drops columns already zeroed by A_mask.
        surviving_links = [(s, targ, src) for s, targ, src, k_dof in wald_scores
                           if chi2.sf(s, df=k_dof) < alpha]

    screened_links_to_check = [(targ, src) for _, targ, src in surviving_links]

    if config.numerical.verbose:
        if use_empirical_null:
            _report_empirical_null(scores, null_fit, threshold, alpha)
        else:
            _report_chi2_null(scores, wald_scores, alpha)

        print(f"\nscreening complete: {len(surviving_links)} out of"
              f" {len(links_to_check)} links survived Wald screening.")

        print("\n--- retained link Wald scores ---")
        for score, targ, src in sorted(surviving_links, reverse=True):
            print(f"{src}->{targ}: {score:.2f}")

    return screened_links_to_check


# Fraction of the scores treated as the null bulk when fitting. The true links
# live in the upper tail; including them would drag the fitted null up and make
# the screen too strict. This is a robustness detail of the fit, not a knob on
# the threshold -- the resulting cut is still an absolute score.
_NULL_TRIM = 0.90


def _fit_empirical_null(scores, trim=_NULL_TRIM):
    """Moment-match a scaled chi2 to the null bulk of the observed scores.

    For s ~ c * chi2(k): mean = c*k and var = 2*c^2*k, hence c = var/(2*mean)
    and k = mean/c. Returns None when the scores are degenerate (no spread, or
    non-positive mean), which no scaled chi2 can describe.
    """
    if len(scores) < 2:
        return None

    bulk = scores[scores <= np.quantile(scores, trim)]

    if len(bulk) < 2:
        return None

    mean, var = bulk.mean(), bulk.var()

    if mean <= 0 or var <= 0:
        return None

    c = var / (2 * mean)
    return c, mean / c


def _report_empirical_null(scores, null_fit, threshold, alpha):
    """Diagnostics for the empirical calibration.

    The fitted k is worth watching next to the nominal k_dof (n_eigenmodes *
    n_orients * that again * order = 72 in the usual config): it shows how far
    the group-lasso-shrunk estimator sits from the unpenalized chi2 the
    statistic would otherwise be compared against.
    """
    print("\n--- Wald screen calibration ---")
    print(f"scores: min {scores.min():.3g}  median {np.median(scores):.3g}  "
          f"mean {scores.mean():.3g}  max {scores.max():.3g}")

    if null_fit is None:
        print("empirical null: NOT FITTABLE (scores have no usable spread) -- "
              "keeping no links")
    else:
        c, k = null_fit
        print(f"empirical null ~ {c:.3g} * chi2({k:.2f})   "
              f"[fitted on the lower {_NULL_TRIM:.0%} of scores]")

    print(f"threshold at alpha={alpha}: {threshold:.4g}")


def _report_chi2_null(scores, wald_scores, alpha):
    """Diagnostics for the nominal central-chi2 test.

    Prints the score each link would have to beat. When the critical value sits
    above the largest observed score, no link can pass at any alpha -- which is
    the failure mode the empirical null exists to work around, so make it
    visible rather than reporting an empty screen with no explanation.
    """
    k_dofs = np.array([k for _, _, _, k in wald_scores])

    print("\n--- Wald screen calibration ---")
    print(f"scores: min {scores.min():.3g}  median {np.median(scores):.3g}  "
          f"mean {scores.mean():.3g}  max {scores.max():.3g}")

    k_lo, k_hi = k_dofs.min(), k_dofs.max()
    k_desc = f"{k_lo}" if k_lo == k_hi else f"{k_lo}-{k_hi}"
    print(f"nominal null: chi2(k_dof), k_dof={k_desc}")

    crit = chi2.isf(alpha, k_dofs)
    print(f"critical score at alpha={alpha}: {crit.min():.4g}-{crit.max():.4g}")

    if crit.min() > scores.max():
        print(f"WARNING: the smallest critical value ({crit.min():.4g}) exceeds "
              f"the largest observed score ({scores.max():.4g}) -- no link can "
              "pass at any alpha. The statistic is built from group-lasso-"
              "shrunk coefficients, so chi2(k_dof) is not its null "
              "distribution. See WALD_SCREEN_NOTES.md.")