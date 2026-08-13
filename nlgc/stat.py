import numpy as np
from scipy.stats import chi2, ncx2


def fdr_control(d, k, alpha, method='BY'):
    """
    CFDR control based on BY procedure

    Parameters
    ----------
    d:  ndarray of shape (n_sources, n_sources)
        deviance matrix
    k:  ndarray of shape (n_sources, n_sources)
        degrees of freedom
    alpha:  FDR rate

    Returns
    -------
    j_val: ndarray of shape (n_sources, n_sources)
    """
    
    if isinstance(k, (list, tuple, np.ndarray)) == 0:
        k = k * np.ones_like(d, dtype=int)

    N_sources = d.shape[0]

    # drop diagonal since we don't test it
    N_tests = N_sources * (N_sources - 1)
    if method == 'BY':
        alpha_bar = alpha * (N_tests + 1) / (2 * N_tests * np.log(N_tests))
    elif method == 'BH':
        alpha_bar = alpha * (N_tests + 1) / (2 * N_tests)
    else:
        raise Exception(f"FDR control method {method} not supported")

    # isolate the off-diagonals (valid tests) to avoid ranking the diagonals
    mask = ~np.eye(N_sources, dtype=bool)
    d_off = d[mask]
    k_off = k[mask]
    
    # sort p-vals
    p_off = chi2.sf(d_off, k_off)
    sorted_idx = np.argsort(p_off)
    p_val = p_off[sorted_idx]

    # threshold for each test (unity indexed)
    threshold = np.arange(1, N_tests + 1) * alpha / (N_tests * np.log(N_tests))

    below_thresh, = np.nonzero(p_val <= threshold)

    if below_thresh.size == 0:
        reject_idx = 0
        gc_test_indices = []
    else:
        reject_idx = below_thresh[-1] + 1       # +1 because 0-indexed
        gc_idx_off = sorted_idx[:reject_idx] 

        # map to link mask
        row_indices, col_indices = np.nonzero(mask)
        gc_test_indices = list(zip(row_indices[gc_idx_off], 
                                   col_indices[gc_idx_off]))

    print(f'p_val {p_val}')

    j_val = np.zeros_like(d, dtype=float)

    for row, col in gc_test_indices:
        non_centrality = d[row, col] - k[row, col] \
            if d[row, col] > k[row, col] \
            else 0
        
        # J = TP/(TP + FN) + TN/(TN + FP) - 1
        # let critical point be C = 0.95 and non-centrality parameter v
        #
        # under the null:
        # TP/(TP + FN) = P(X < C) = 1 - alpha 
        # (this alpha is alpha_bar above, the familywise error rate)
        #
        # under the alternative:
        # TN/(TN + FP) = P(X > C | v) = 1 - P(X < C | v) 
        #                             = 1 - nc_chi2_cdf(C; v) 
        #                             = nc_sf(alpha; v)
        # 
        # to get the critical value, we do inv_chi2cdf(C) or inv_sf(alpha)
        # this gives the critical value that we defined under the null that will 
        # be compared to the alternative. so
        # 1 - nc_inv_sf(inv_sf(alpha); v)
        # 
        # putting the whole expression together we get:
        # J = TP/(TP + FN) + TN/(TN + FP) - 1
        #   = (1 - alpha) + (1 - nc_cdf(inv_sf(alpha); v)) - 1
        #   =  1 - alpha - nc_sf(inv_sf(alpha); v)
        #   =  1 - alpha + power - 1 = power - alpha

        crit = chi2.isf(alpha_bar, k[row, col])
        power = ncx2.sf(crit, k[row, col], non_centrality)

        j_val[row, col] = power - alpha_bar

    return j_val


