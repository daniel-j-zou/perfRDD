import numpy as np
from scipy.stats import norm
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


def pGen(n, beta, gamma, rng=None):
    """
    Generates:
      x ~ N(0,1)
      a ~ +/- Exp(1) (bimodal)
      eta | a ~ N(a, 1)
      nu  | a ~ N(3*exp(a)/(1+exp(a)), 1)
      w   | eta ~ N(2*exp(eta)/(1+exp(eta)), 0.1)
      q = gamma * x + eta
      y = 1{q > 0} * w + beta * x + nu
    """
    if rng is None:
        rng = np.random.default_rng()

    x = rng.normal(0.0, 1.0, size=n)

    signs = rng.choice([-1.0, 1.0], size=n)
    a = signs * rng.exponential(scale=1.0, size=n)

    eta = rng.normal(loc=a, scale=1.0, size=n)
    nu_mean = 3 * np.exp(a) / (1 + np.exp(a))
    nu = rng.normal(loc=nu_mean, scale=1.0, size=n)

    w_mean = 2 * np.exp(eta) / (1 + np.exp(eta))
    w = rng.normal(loc=w_mean, scale=0.1, size=n)

    q = gamma * x + eta
    s = (q > 0).astype(float)

    y = s * w + beta * x + nu
    return x, y, eta, nu, w, q


def _basis_params(q, kn, basis, bwf, support=None):
    """Compute basis parameters from q.

    support : optional (lo, hi) tuple. If given, B-spline knots are placed
              over this fixed range instead of the data range.
    """
    ql = np.min(q)
    qr = np.ptp(q)
    if qr == 0:
        qr = 1.0

    if basis == "gaussian":
        idx = np.arange(1, kn + 1) - 0.5
        mu = ql + idx * qr / kn
        bw = bwf * qr / kn
        return {"type": "gaussian", "mu": mu, "bw": bw}
    elif basis == "bspline":
        degree = 3
        if support is not None:
            lo, hi = support
        else:
            lo, hi = ql, ql + qr
        interior = np.linspace(lo, hi, kn + 2)[1:-1]
        t = np.concatenate([
            np.repeat(lo, degree + 1),
            interior,
            np.repeat(hi, degree + 1),
        ])
        return {"type": "bspline", "t": t, "degree": degree,
                "ql": lo, "qr_max": hi}
    else:
        raise ValueError(f"Unknown basis: {basis}")


def _eval_basis(pts, info):
    """Evaluate basis functions at pts."""
    pts = np.asarray(pts, dtype=float)
    if info["type"] == "gaussian":
        return norm.pdf((pts[:, None] - info["mu"][None, :]), scale=info["bw"])
    elif info["type"] == "bspline":
        pts_c = np.clip(pts, info["ql"], info["qr_max"])
        return BSpline.design_matrix(pts_c, info["t"], info["degree"]).toarray()


def _in_support(pts, info_tr, info_con):
    """Check which points are in the support of both basis fits."""
    pts = np.asarray(pts, dtype=float)
    if info_tr["type"] == "gaussian":
        dmt = _eval_basis(pts, info_tr)
        dmc = _eval_basis(pts, info_con)
        return (dmt.sum(axis=1) > 1.0) & (dmc.sum(axis=1) > 1.0)
    elif info_tr["type"] == "bspline":
        in_tr = (pts >= info_tr["ql"]) & (pts <= info_tr["qr_max"])
        in_con = (pts >= info_con["ql"]) & (pts <= info_con["qr_max"])
        return in_tr & in_con


def _ridge_lstsq(H, y, lam):
    """Solve ridge regression: min ||y - H @ b||^2 + lam * ||b||^2."""
    p = H.shape[1]
    return np.linalg.solve(H.T @ H + lam * np.eye(p), H.T @ y)


def TSplm(y, x, q, kn, bwf, basis="gaussian", support=None,
          regularization="threshold", ridge_lam=1.0, col_mask=None):
    """
    Partial linear model:
        y = b * x + h(q) + error
    where h(q) is approximated by a series expansion with 'kn' knots.

    basis: "gaussian" or "bspline".
    support: optional (lo, hi) for fixed B-spline knot range.
    regularization:
        "threshold" - drop low-variance columns (original method)
        "ridge"     - L2 penalty, no column dropping
        "pooled_threshold" - use externally supplied col_mask
    col_mask: boolean masks (mask1, mask2) for pooled_threshold mode.

    Returns:
      h, b, info, (mask1, mask2)
    """
    y = np.asarray(y)
    x = np.asarray(x)
    q = np.asarray(q)
    n = len(y)

    info = _basis_params(q, kn, basis, bwf, support=support)
    Phi = _eval_basis(q, info)

    # First regression: y ~ x + Phi
    H = np.column_stack((x, Phi))

    if regularization == "none":
        be1, *_ = np.linalg.lstsq(H, y, rcond=None)
        mask1 = np.ones(H.shape[1], dtype=bool)
    elif regularization == "ridge":
        be1 = _ridge_lstsq(H, y, ridge_lam)
        mask1 = np.ones(H.shape[1], dtype=bool)
    elif regularization == "pooled_threshold" and col_mask is not None:
        mask1 = col_mask[0]
        H_sel = H[:, mask1]
        be1 = np.zeros(H.shape[1])
        if H_sel.shape[1] > 0:
            be1_sel, *_ = np.linalg.lstsq(H_sel, y, rcond=None)
            be1[mask1] = be1_sel
    else:  # "threshold"
        v = H.var(axis=0, ddof=1)
        thresh1 = 0.03 * np.max(v[1:]) if len(v) > 1 else 0.0
        mask1 = v > thresh1
        H_sel = H[:, mask1]
        be1 = np.zeros(H.shape[1])
        if H_sel.shape[1] > 0:
            be1_sel, *_ = np.linalg.lstsq(H_sel, y, rcond=None)
            be1[mask1] = be1_sel

    # Second regression with shifted basis
    Phi_shift = _eval_basis(q + 0.01 * x, info)
    Hs = np.column_stack((x, Phi_shift))
    diff = (Hs - H) @ be1
    H2 = np.column_stack((x, diff, H[:, 1:]))

    if regularization == "none":
        be, *_ = np.linalg.lstsq(H2, y, rcond=None)
        mask2 = np.ones(H2.shape[1], dtype=bool)
    elif regularization == "ridge":
        be = _ridge_lstsq(H2, y, ridge_lam)
        mask2 = np.ones(H2.shape[1], dtype=bool)
    elif regularization == "pooled_threshold" and col_mask is not None:
        mask2 = col_mask[1]
        H2_sel = H2[:, mask2]
        be = np.zeros(H2.shape[1])
        if H2_sel.shape[1] > 0:
            be_sel, *_ = np.linalg.lstsq(H2_sel, y, rcond=None)
            be[mask2] = be_sel
    else:  # "threshold"
        v2 = H2.var(axis=0, ddof=1)
        thresh2 = 0.03 * np.max(v2[2:]) if len(v2) > 2 else 0.0
        mask2 = v2 > thresh2
        H2_sel = H2[:, mask2]
        be = np.zeros(H2.shape[1])
        if H2_sel.shape[1] > 0:
            be_sel, *_ = np.linalg.lstsq(H2_sel, y, rcond=None)
            be[mask2] = be_sel

    b = be[0]
    h = be[2:]
    return h, b, info, (mask1, mask2)


def _run_plugin(etaTr, etaCon, y, x, iTr, iCon, kn, bwf, basis,
                support=None, regularization="threshold", ridge_lam=1.0,
                out_of_support="clip"):
    """Run the plug-in estimator for a given basis type.

    out_of_support: "clip" (truncate to boundary) or "drop" (exclude).
    Returns (ewA, hTr, hCon, infoTr, infoCon)."""

    if regularization == "pooled_threshold":
        # Compute column masks from pooled (treated + control) data
        etaAll = np.concatenate([etaTr, etaCon])
        xAll = np.concatenate([x[iTr], x[iCon]])
        yAll = np.concatenate([y[iTr], y[iCon]])
        _, _, _, (mask1_pool, mask2_pool) = TSplm(
            yAll, xAll, etaAll,
            kn, bwf, basis=basis, support=support, regularization="threshold"
        )
        col_mask = (mask1_pool, mask2_pool)

        hCon, bCon, infoCon, _ = TSplm(
            y[iCon], x[iCon], etaCon, kn, bwf, basis=basis, support=support,
            regularization="pooled_threshold", col_mask=col_mask
        )
        hTr, bTr, infoTr, _ = TSplm(
            y[iTr], x[iTr], etaTr, kn, bwf, basis=basis, support=support,
            regularization="pooled_threshold", col_mask=col_mask
        )
    else:
        hCon, bCon, infoCon, _ = TSplm(
            y[iCon], x[iCon], etaCon, kn, bwf, basis=basis, support=support,
            regularization=regularization, ridge_lam=ridge_lam
        )
        hTr, bTr, infoTr, _ = TSplm(
            y[iTr], x[iTr], etaTr, kn, bwf, basis=basis, support=support,
            regularization=regularization, ridge_lam=ridge_lam
        )

    # Handle out-of-support etaTr for evaluation
    if support is not None and out_of_support == "drop":
        in_range = (etaTr >= support[0]) & (etaTr <= support[1])
        etaTr_eval = etaTr[in_range]
    elif support is not None:
        etaTr_eval = np.clip(etaTr, support[0], support[1])
        in_range = None
    else:
        etaTr_eval = etaTr
        in_range = None

    if len(etaTr_eval) == 0:
        return np.nan, hTr, hCon, infoTr, infoCon

    dmt = _eval_basis(etaTr_eval, infoTr)
    dmc = _eval_basis(etaTr_eval, infoCon)
    insupp = _in_support(etaTr_eval, infoTr, infoCon)
    if insupp.any():
        return np.mean(dmt[insupp] @ hTr - dmc[insupp] @ hCon), hTr, hCon, infoTr, infoCon
    else:
        return np.nan, hTr, hCon, infoTr, infoCon


def main():
    rng = np.random.default_rng(2025)

    # Parameters
    Gamma = 2.0
    Beta = 2.0
    KN = 22
    ETA_SUPPORT = (-2.8, 2.8)   # ~90% of eta distribution
    RIDGE_C = 1.0   # ridge lambda = RIDGE_C / sqrt(N)
    C = 1.0

    # Sample sizes and replications per size
    N_grid = [2000, 5000, 10000, 50000, 100000]
    R = 200
    M = len(N_grid) * R
    N_vals = np.repeat(N_grid, R)

    # Arrays to store results — 3 methods
    ew0 = np.zeros(M)         # oracle
    ewA_g = np.zeros(M)       # Gaussian (threshold, baseline)
    ewA_none = np.zeros(M)    # B-spline, no reg, clip (truncate)
    ewA_ridge = np.zeros(M)   # (unused, placeholder)
    ewA_pool = np.zeros(M)    # B-spline, no reg, drop

    # Big sample (needed for rng state consistency)
    x_big, y_big, eta_big, nu_big, w_big, q_big = pGen(200000, Beta, Gamma, rng=rng)

    # Monte Carlo loop
    for m in range(M):
        n = N_vals[m]
        x, y, eta, nu, w, q = pGen(n, Beta, Gamma, rng=rng)

        s = (q > 0)
        idx_all = np.arange(n)
        iTr = idx_all[s]
        iCon = idx_all[~s]

        # Oracle E[W|Q>0]
        if len(iTr) > 0:
            ew0[m] = w[iTr].mean()
        else:
            ew0[m] = np.nan

        # First stage: regress q on [1, x] to recover etaHat
        X_design = np.column_stack((np.ones(n), x))
        coef_q, *_ = np.linalg.lstsq(X_design, q, rcond=None)
        b0_hat, b1_hat = coef_q
        etaHat = q - b0_hat - b1_hat * x

        etaCon = etaHat[iCon]
        etaTr = etaHat[iTr]

        kn_s = max(4, int(round(n ** (1.0 / 3.0))))

        # 1) Gaussian plug-in (original threshold method)
        ewA_g[m], _, _, _, _ = _run_plugin(
            etaTr, etaCon, y, x, iTr, iCon, KN, 1.0, "gaussian",
            regularization="threshold"
        )

        # 2) B-spline, no reg, clip (truncate out-of-support to boundary)
        ewA_none[m], _, _, _, _ = _run_plugin(
            etaTr, etaCon, y, x, iTr, iCon, kn_s, 1.0, "bspline",
            support=ETA_SUPPORT, regularization="none", out_of_support="clip"
        )

        # 3) B-spline, no reg, drop (exclude out-of-support)
        ewA_pool[m], _, _, _, _ = _run_plugin(
            etaTr, etaCon, y, x, iTr, iCon, kn_s, 1.0, "bspline",
            support=ETA_SUPPORT, regularization="none", out_of_support="drop"
        )

        if (m + 1) % 50 == 0:
            print(f"Completed replication {m+1}/{M}")

    # ---- Post-processing ----

    def _mad(arr):
        """Median absolute deviation (scaled to match Std for normal data)."""
        return 1.4826 * np.nanmedian(np.abs(arr - np.nanmedian(arr)))

    print("\n--- Bias and Std by N ---")
    print(f"{'N':>8s} {'KN_S':>5s}  "
          f"{'Bias_G':>8s} {'Std_G':>8s}  "
          f"{'Bias_Cl':>8s} {'Std_Cl':>8s}  "
          f"{'Bias_Dr':>8s} {'Std_Dr':>8s}")
    for ni in N_grid:
        mask = (N_vals == ni)
        ok = mask & (np.abs(ewA_g - ew0) < 1.0) & (np.abs(ewA_pool - ew0) < 1.0) & (np.abs(ewA_none - ew0) < 1.0)
        bg = np.nanmean(ewA_g[ok] - ew0[ok])
        sg = np.nanstd(ewA_g[ok] - ew0[ok])
        bc = np.nanmean(ewA_none[ok] - ew0[ok])
        sc = np.nanstd(ewA_none[ok] - ew0[ok])
        bd = np.nanmean(ewA_pool[ok] - ew0[ok])
        sd = np.nanstd(ewA_pool[ok] - ew0[ok])
        kn_s = max(4, int(round(ni ** (1.0 / 3.0))))
        print(f"{ni:>8d} {kn_s:>5d}  "
              f"{bg:>8.4f} {sg:>8.4f}  "
              f"{bc:>8.4f} {sc:>8.4f}  "
              f"{bd:>8.4f} {sd:>8.4f}")

    print(f"\n--- sqrt(N) * Std  and  sqrt(N) * MAD ---")
    print(f"{'N':>8s}  {'sN*Std_G':>9s} {'sN*MAD_G':>9s}  "
          f"{'sN*StdCl':>9s} {'sN*MADCl':>9s}  "
          f"{'sN*StdDr':>9s} {'sN*MADDr':>9s}")
    for ni in N_grid:
        mask = (N_vals == ni)
        ok = mask & (np.abs(ewA_g - ew0) < 1.0) & (np.abs(ewA_pool - ew0) < 1.0) & (np.abs(ewA_none - ew0) < 1.0)
        sn = np.sqrt(ni)
        dg = ewA_g[ok] - ew0[ok]
        dc = ewA_none[ok] - ew0[ok]
        dd = ewA_pool[ok] - ew0[ok]
        print(f"{ni:>8d}  "
              f"{sn * np.nanstd(dg):>9.2f} {sn * _mad(dg):>9.2f}  "
              f"{sn * np.nanstd(dc):>9.2f} {sn * _mad(dc):>9.2f}  "
              f"{sn * np.nanstd(dd):>9.2f} {sn * _mad(dd):>9.2f}")

    # ---- Plots ----

    # Collect stats for bar chart
    labels = ["Gaussian", "Clip", "Drop"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    all_ew = [ewA_g, ewA_none, ewA_pool]

    mean_bias = {l: [] for l in labels}
    std_bias = {l: [] for l in labels}
    for ni in N_grid:
        ok = ((N_vals == ni) & (np.abs(ewA_g - ew0) < 1.0)
              & (np.abs(ewA_pool - ew0) < 1.0)
              & (np.abs(ewA_none - ew0) < 1.0))
        for l, ew in zip(labels, all_ew):
            mean_bias[l].append(np.nanmean(ew[ok] - ew0[ok]))
            std_bias[l].append(np.nanstd(ew[ok] - ew0[ok]))

    # Figure 1: bias bar chart
    plt.figure(1, figsize=(10, 5))
    plt.clf()
    x_pos = np.arange(len(N_grid))
    nbar = len(labels)
    w = 0.8 / nbar
    for i, (l, c) in enumerate(zip(labels, colors)):
        offset = (i - (nbar - 1) / 2) * w
        plt.bar(x_pos + offset, mean_bias[l], w, yerr=std_bias[l],
                capsize=3, label=l, alpha=0.7, color=c)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xticks(x_pos, [str(n) for n in N_grid])
    plt.xlabel("N")
    plt.ylabel("Bias (estimate - oracle)")
    plt.title("Bias of plug-in estimators by sample size")
    plt.legend()

    # Figure 2: boxplots
    plt.figure(2, figsize=(12, 5))
    plt.clf()
    bp_data = {l: [] for l in labels}
    for ni in N_grid:
        ok = ((N_vals == ni) & (np.abs(ewA_g - ew0) < 1.0)
              & (np.abs(ewA_pool - ew0) < 1.0)
              & (np.abs(ewA_none - ew0) < 1.0))
        for l, ew in zip(labels, all_ew):
            bp_data[l].append(ew[ok] - ew0[ok])

    group_width = 3.5
    box_w = 0.7
    bps = []
    for i, (l, c) in enumerate(zip(labels, colors)):
        positions = np.arange(len(N_grid)) * group_width + i * (box_w + 0.1)
        bp = plt.boxplot(bp_data[l], positions=positions, widths=box_w, patch_artist=True)
        for box in bp['boxes']:
            box.set_facecolor(c)
            box.set_alpha(0.5)
        bps.append(bp)

    plt.axhline(0, color="black", linewidth=0.5)
    center_positions = np.arange(len(N_grid)) * group_width + (nbar - 1) * (box_w + 0.1) / 2
    plt.xticks(center_positions, [str(n) for n in N_grid])
    plt.xlabel("N")
    plt.ylabel("Estimate - Oracle")
    plt.title("Boxplots by N")
    plt.legend([bp['boxes'][0] for bp in bps], labels)
    plt.tight_layout()

    # Figure 3: normality check — histograms of sqrt(N)*(est - oracle) at each N
    from scipy.stats import shapiro, kurtosis, skew
    ncols = len(labels)
    fig3, axes = plt.subplots(len(N_grid), ncols, figsize=(4 * ncols, 3 * len(N_grid)))
    fig3.suptitle("sqrt(N) * (estimate - oracle):  normality check", fontsize=13)
    for row, ni in enumerate(N_grid):
        ok = ((N_vals == ni) & (np.abs(ewA_g - ew0) < 1.0)
              & (np.abs(ewA_pool - ew0) < 1.0)
              & (np.abs(ewA_none - ew0) < 1.0))
        sn = np.sqrt(ni)
        for col, (l, c, ew) in enumerate(zip(labels, colors, all_ew)):
            d = sn * (ew[ok] - ew0[ok])
            d = d[np.isfinite(d)]
            ax = axes[row, col]
            ax.hist(d, bins=30, density=True, alpha=0.6, color=c)
            # overlay normal fit
            mu_d, std_d = np.mean(d), np.std(d)
            xg = np.linspace(mu_d - 4 * std_d, mu_d + 4 * std_d, 200)
            ax.plot(xg, norm.pdf(xg, mu_d, std_d), 'k--', lw=1)
            # Shapiro-Wilk (on up to 5000 samples)
            d_sw = d[:5000] if len(d) > 5000 else d
            if len(d_sw) >= 20:
                sw_stat, sw_p = shapiro(d_sw)
            else:
                sw_stat, sw_p = np.nan, np.nan
            sk = skew(d)
            ku = kurtosis(d)  # excess kurtosis (0 for normal)
            ax.set_title(f"{l}, N={ni}\nSW p={sw_p:.3f}, skew={sk:.2f}, kurt={ku:.2f}",
                         fontsize=8)
            if row == len(N_grid) - 1:
                ax.set_xlabel("sqrt(N)*(est-oracle)")
    fig3.tight_layout()

    # Print normality summary
    print(f"\n--- Normality diagnostics: Shapiro-Wilk p-value, skewness, excess kurtosis ---")
    print(f"{'N':>8s}  {'SW_G':>6s} {'sk_G':>6s} {'ku_G':>6s}  "
          f"{'SW_Cl':>6s} {'skCl':>6s} {'kuCl':>6s}  "
          f"{'SW_Dr':>6s} {'skDr':>6s} {'kuDr':>6s}")
    for ni in N_grid:
        ok = ((N_vals == ni) & (np.abs(ewA_g - ew0) < 1.0)
              & (np.abs(ewA_pool - ew0) < 1.0) & (np.abs(ewA_none - ew0) < 1.0))
        sn = np.sqrt(ni)
        vals = []
        for ew in all_ew:
            d = sn * (ew[ok] - ew0[ok])
            d = d[np.isfinite(d)]
            if len(d) >= 20:
                _, p = shapiro(d[:5000])
            else:
                p = np.nan
            vals.extend([p, skew(d), kurtosis(d)])
        print(f"{ni:>8d}  {vals[0]:>6.3f} {vals[1]:>6.2f} {vals[2]:>6.2f}  "
              f"{vals[3]:>6.3f} {vals[4]:>6.2f} {vals[5]:>6.2f}  "
              f"{vals[6]:>6.3f} {vals[7]:>6.2f} {vals[8]:>6.2f}")

    plt.figure(1).savefig("/Users/zoudj/Documents/Research/perfRDD/general/fig1_estimators_vs_N.png", dpi=150, bbox_inches="tight")
    plt.figure(2).savefig("/Users/zoudj/Documents/Research/perfRDD/general/fig2_histograms.png", dpi=150, bbox_inches="tight")
    fig3.savefig("/Users/zoudj/Documents/Research/perfRDD/general/fig3_normality.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
