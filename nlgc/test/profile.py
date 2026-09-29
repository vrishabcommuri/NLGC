import numpy as np

def pretty_print_elapsed(elapsed):
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days:
        print(f"Execution time: {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif hours:
        print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes:
        print(f"Execution time: {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"Execution time: {seconds:.2f}s")
        

def _svd_rank_report(x, rtol=(1e-12, 1e-10, 1e-8, 1e-6)):
    """Return singular values and numerical ranks for x: (features, samples)."""
    x = np.asarray(x, dtype=np.float64)

    x_centered = x - x.mean(axis=1, keepdims=True)
    s = np.linalg.svd(x_centered, compute_uv=False)

    if s[0] <= np.finfo(float).tiny:
        ranks = {tol: 0 for tol in rtol}
    else:
        ranks = {tol: int(np.sum(s > tol * s[0])) for tol in rtol}

    return s, ranks


def _summary(name, x):
    x = np.asarray(x, dtype=np.float64)

    finite = np.isfinite(x)
    if not finite.all():
        raise ValueError(f"{name} has non-finite values.")

    print(
        f"{name}:"
        f"\n  shape={x.shape}"
        f"\n  mean={x.mean():.6e}"
        f"\n  std={x.std():.6e}"
        f"\n  rms={np.sqrt(np.mean(x**2)):.6e}"
        f"\n  absmax={np.max(np.abs(x)):.6e}"
        f"\n  Fro={np.linalg.norm(x):.6e}"
    )


def report_em_boundary(subject_label, M_raw, M, G, whitener, singular_values,
    gain_info, evoked):
    """
    Diagnostics at the pre-EM boundary.

    Assumptions:
        M_raw: (n_good_sensor_coords, n_times)
        whitener: (rank, n_good_sensor_coords) or square
        M: whitener @ M_raw, shape (rank, n_times)
        G: whitened/reduced gain, shape (rank, n_states)
    """
    M_raw = np.asarray(M_raw, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    W = np.asarray(whitener, dtype=np.float64)
    singular_values = np.asarray(singular_values, dtype=np.float64)

    print("\n" + "=" * 80)
    print(f"PRE-EM SUBJECT REPORT: {subject_label}")
    print("=" * 80)

    print("\nChannel basis")
    print(f"  evoked channels: {len(evoked.ch_names)}")
    print(f"  gain_info channels: {len(gain_info['ch_names'])}")
    print(f"  M_raw rows: {M_raw.shape[0]}")
    print(f"  whitener shape: {W.shape}")
    print(f"  M rows: {M.shape[0]}")
    print(f"  G rows: {G.shape[0]}")
    print(f"  first gain channels: {gain_info['ch_names'][:5]}")
    print(f"  first evoked channels: {evoked.ch_names[:5]}")

    assert M_raw.shape[0] == W.shape[1], (
        "M_raw must use the input channel basis of the whitener."
    )
    assert M.shape[0] == W.shape[0], (
        "M must use the output basis of the whitener."
    )
    assert G.shape[0] == M.shape[0], (
        "Whitened gain and whitened data must have the same row basis."
    )

    _summary("M_raw", M_raw)
    _summary("M whitened", M)
    _summary("G whitened/reduced", G)
    _summary("Whitener", W)

    print("\nWhitened data scale")
    mode_var = np.var(M, axis=1)
    mode_rms = np.sqrt(np.mean(M**2, axis=1))

    print(f"  median mode variance: {np.median(mode_var):.6e}")
    print(f"  mean mode variance: {np.mean(mode_var):.6e}")
    print(f"  min/max mode variance: "
          f"{mode_var.min():.6e} / {mode_var.max():.6e}")
    print(f"  median mode RMS: {np.median(mode_rms):.6e}")
    print(f"  global RMS: {np.sqrt(np.mean(M**2)):.6e}")

    s_raw, rank_raw = _svd_rank_report(M_raw)
    s_white, rank_white = _svd_rank_report(M)
    s_gain = np.linalg.svd(G, compute_uv=False)
    s_w = np.linalg.svd(W, compute_uv=False)

    print("\nNumerical rank")
    print(f"  M_raw ranks: {rank_raw}")
    print(f"  M ranks: {rank_white}")
    print(f"  whitener singular max/min: "
          f"{s_w[0]:.6e} / {s_w[-1]:.6e}")
    print(f"  gain singular max/min: "
          f"{s_gain[0]:.6e} / {s_gain[-1]:.6e}")
    print(f"  gain condition: "
          f"{s_gain[0] / max(s_gain[-1], 1e-30):.6e}")

    print("\nRetained noise-eigenvalue information")
    print(f"  singular_values shape: {singular_values.shape}")
    print(f"  min/median/max: "
          f"{singular_values.min():.6e} / "
          f"{np.median(singular_values):.6e} / "
          f"{singular_values.max():.6e}")

    print("\nLead-field column norms")
    g_col = np.linalg.norm(G, axis=0)
    print(f"  min/median/max: "
          f"{g_col.min():.6e} / "
          f"{np.median(g_col):.6e} / "
          f"{g_col.max():.6e}")
    print(
        "  fraction < 1e-6 * max: "
        f"{np.mean(g_col < 1e-6 * max(g_col.max(), 1e-30)):.6f}"
    )

    print("\nTemporal structure of whitened data")
    M0 = M[:, :-1] - M[:, :-1].mean(axis=1, keepdims=True)
    M1 = M[:, 1:] - M[:, 1:].mean(axis=1, keepdims=True)

    rho1 = np.sum(M0 * M1, axis=1) / np.maximum(
        np.sqrt(np.sum(M0**2, axis=1) * np.sum(M1**2, axis=1)),
        1e-30,
    )

    print(
        f"  lag-1 mode correlation min/median/mean/max: "
        f"{rho1.min():.4f} / "
        f"{np.median(rho1):.4f} / "
        f"{rho1.mean():.4f} / "
        f"{rho1.max():.4f}"
    )

    return {
        "raw_rms": float(np.sqrt(np.mean(M_raw**2))),
        "white_rms": float(np.sqrt(np.mean(M**2))),
        "white_mode_var_median": float(np.median(mode_var)),
        "white_rank": rank_white,
        "gain_singular_values": s_gain,
        "gain_condition": float(s_gain[0] / max(s_gain[-1], 1e-30)),
        "lag1_mode_rho_median": float(np.median(rho1)),
    }