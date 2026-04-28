"""
plm_dual_threshold_fixed.py
===========================
Windowed dual-threshold partially linear model.

Main changes vs the previous version
------------------------------------
1) Window restriction is applied by default:
       |score1 - c1| <= fit_window1   with default fit_window1 = 10
       |score2 - c2| <= fit_window2   with default fit_window2 = 100000

2) The old 2D "shift trick" that moved both eta dimensions using only
   x_mat[:, 0] has been removed. It is replaced by a more principled
   partially linear sieve / FWL residualization step:

       y = X b + h(eta1, eta2) + e

   using a 2D Gaussian product basis for h(·,·).

   Concretely:
   - project y on Phi(eta1, eta2)
   - project each column of X on Phi(eta1, eta2)
   - regress residualized y on residualized X to get b
   - recover h from y - X b projected back on Phi

This keeps the same user-facing API as much as possible.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# ── timer helper ──────────────────────────────────────────────────────────────
def _p(msg, t0):
    print(f"  {msg}  [{time.time()-t0:.1f}s]", flush=True)


# ── linear algebra helpers ───────────────────────────────────────────────────
def _solve_ridge(xtx, xty, ridge=1e-6):
    """
    Solve (X'X + ridge * I) beta = X'Y.
    Works for vector or matrix right-hand side.
    """
    xtx = np.asarray(xtx, float)
    xty = np.asarray(xty, float)
    k = xtx.shape[0]
    reg = ridge * np.eye(k)
    return np.linalg.solve(xtx + reg, xty)


# ── 2D Gaussian product basis ────────────────────────────────────────────────
def _make_basis_2d(e1, e2, kn, bwf):
    e1 = np.asarray(e1, float).reshape(-1)
    e2 = np.asarray(e2, float).reshape(-1)

    e1_min, e1_rng = e1.min(), np.ptp(e1)
    e2_min, e2_rng = e2.min(), np.ptp(e2)

    e1_rng = e1_rng if e1_rng > 0 else 1.0
    e2_rng = e2_rng if e2_rng > 0 else 1.0

    idx = np.arange(1, kn + 1) - 0.5
    mu1 = e1_min + idx * e1_rng / kn
    mu2 = e2_min + idx * e2_rng / kn

    bw1 = bwf * e1_rng / kn
    bw2 = bwf * e2_rng / kn
    bw1 = bw1 if bw1 > 1e-12 else 1.0
    bw2 = bw2 if bw2 > 1e-12 else 1.0

    A = norm.pdf(e1[:, None] - mu1[None, :], scale=bw1)
    B = norm.pdf(e2[:, None] - mu2[None, :], scale=bw2)
    Phi = (A[:, :, None] * B[:, None, :]).reshape(len(e1), kn * kn)
    return Phi, mu1, bw1, mu2, bw2


# ─────────────────────────────────────────────────────────────────────────────
# TSplm2D  — partially linear sieve estimator using FWL residualization
# ─────────────────────────────────────────────────────────────────────────────
def TSplm2D(y, x_mat, eta1, eta2, kn, bwf, ridge=1e-6, label=""):
    """
    Fit y = X b + h(eta1, eta2) + e using a 2D Gaussian product basis.

    Returns
    -------
    h, b, mu1, bw1, mu2, bw2
    """
    t0 = time.time()
    y = np.asarray(y, float).reshape(-1)
    xm = np.asarray(x_mat, float)
    e1 = np.asarray(eta1, float).reshape(-1)
    e2 = np.asarray(eta2, float).reshape(-1)

    if xm.ndim != 2:
        raise ValueError(f"x_mat must be 2D, got shape={xm.shape}")

    n, p = xm.shape
    if not (len(y) == len(e1) == len(e2) == n):
        raise ValueError(
            f"Length mismatch in TSplm2D: y={len(y)}, x_mat={xm.shape}, "
            f"eta1={len(e1)}, eta2={len(e2)}"
        )

    tag = f"[{label}] " if label else ""

    Phi, mu1, bw1, mu2, bw2 = _make_basis_2d(e1, e2, kn, bwf)
    k = Phi.shape[1]
    _p(f"{tag}basis {kn}x{kn}={k}  n={n:,}  p={p}", t0)

    # Step 1: partial out the nonparametric component from y and X
    G = Phi.T @ Phi
    gamma_y = _solve_ridge(G, Phi.T @ y, ridge=ridge)                 # (k,)
    gamma_x = _solve_ridge(G, Phi.T @ xm, ridge=ridge)                # (k,p)

    y_tilde = y - Phi @ gamma_y                                       # (n,)
    X_tilde = xm - Phi @ gamma_x                                      # (n,p)

    # Drop near-zero residualized columns for numerical stability
    xvar = X_tilde.var(axis=0)
    keep = xvar > 1e-10
    if not np.any(keep):
        raise ValueError("All residualized x columns are near-zero in TSplm2D.")

    b_small, *_ = np.linalg.lstsq(X_tilde[:, keep], y_tilde, rcond=None)
    b = np.zeros(p)
    b[keep] = b_small
    _p(f"{tag}FWL step kept {keep.sum()}/{p} x cols", t0)

    # Step 2: recover h on the basis from y - Xb
    y_res = y - xm @ b
    h = _solve_ridge(G, Phi.T @ y_res, ridge=ridge)
    _p(f"{tag}h recovered on {k} basis terms", t0)

    return h, b, mu1, bw1, mu2, bw2


# ─────────────────────────────────────────────────────────────────────────────
# ewA_dual
# ─────────────────────────────────────────────────────────────────────────────
def ewA_dual(
    y,
    x_mat,
    r1,
    r2,
    treated,
    kn=6,
    bwf=1.0,
    support_pct=10,
    cost=0.0,
    fit_window1=10,
    fit_window2=100000,
    ridge=1e-6,
):
    """
    Dual-threshold ewA estimation with optional window restriction.

    Parameters
    ----------
    fit_window1 : float or None
        Keep only rows with |r1| <= fit_window1.
    fit_window2 : float or None
        Keep only rows with |r2| <= fit_window2.
    ridge : float
        Ridge penalty used in nonparametric projection steps.
    """
    t0 = time.time()

    y = np.asarray(y, dtype=float).reshape(-1)
    x_mat = np.asarray(x_mat, dtype=float)
    r1 = np.asarray(r1, dtype=float).reshape(-1)
    r2 = np.asarray(r2, dtype=float).reshape(-1)
    treated = np.asarray(treated).reshape(-1).astype(bool)

    if x_mat.ndim != 2:
        raise ValueError(f"x_mat must be 2D, got shape={x_mat.shape}")

    n = len(y)
    if not (len(r1) == len(r2) == len(treated) == n and x_mat.shape[0] == n):
        raise ValueError(
            f"Length mismatch: y={len(y)}, x_mat={x_mat.shape}, "
            f"r1={len(r1)}, r2={len(r2)}, treated={len(treated)}"
        )

    print("\n[1/3] First stage ...", flush=True)

    # clean jointly
    x_mat = np.where(np.isfinite(x_mat), x_mat, np.nan)
    r1 = np.where(np.isfinite(r1), r1, np.nan)
    r2 = np.where(np.isfinite(r2), r2, np.nan)
    y = np.where(np.isfinite(y), y, np.nan)

    row_mask = (
        np.isfinite(y)
        & np.isfinite(r1)
        & np.isfinite(r2)
        & np.all(np.isfinite(x_mat), axis=1)
    )

    y = y[row_mask]
    x_mat = x_mat[row_mask]
    r1 = r1[row_mask]
    r2 = r2[row_mask]
    treated = treated[row_mask]

    # local fitting window restriction
    if fit_window1 is not None or fit_window2 is not None:
        win_mask = np.ones(len(y), dtype=bool)
        if fit_window1 is not None:
            win_mask &= np.abs(r1) <= float(fit_window1)
        if fit_window2 is not None:
            win_mask &= np.abs(r2) <= float(fit_window2)

        before_win = len(y)
        y = y[win_mask]
        x_mat = x_mat[win_mask]
        r1 = r1[win_mask]
        r2 = r2[win_mask]
        treated = treated[win_mask]
        _p(
            f"window kept {len(y):,}/{before_win:,} rows "
            f"(|r1|<={fit_window1}, |r2|<={fit_window2})",
            t0,
        )

    # drop absurd rows
    big_mask = np.max(np.abs(x_mat), axis=1) < 1e6
    yr_mask = (np.abs(y) < 1e12) & (np.abs(r1) < 1e12) & (np.abs(r2) < 1e12)
    keep_mask = big_mask & yr_mask

    y = y[keep_mask]
    x_mat = x_mat[keep_mask]
    r1 = r1[keep_mask]
    r2 = r2[keep_mask]
    treated = treated[keep_mask]

    if len(y) == 0:
        raise ValueError("No rows left after cleaning/windowing.")

    # drop constant columns
    std = np.std(x_mat, axis=0)
    keep_cols = std > 1e-8
    x_mat = x_mat[:, keep_cols]
    if x_mat.shape[1] == 0:
        raise ValueError("All x columns were dropped as constant / invalid.")

    # standardize x after windowing
    x_mean = x_mat.mean(axis=0)
    x_std = x_mat.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_mat = (x_mat - x_mean) / x_std

    n = len(y)
    print(f"  after cleaning/windowing: n={n:,}, p={x_mat.shape[1]}", flush=True)
    print(f"  treated={treated.sum():,}   control={(~treated).sum():,}", flush=True)

    # first stage: residualize running variables on X
    X1 = np.column_stack((np.ones(n), x_mat))

    coef1, *_ = np.linalg.lstsq(X1, r1, rcond=None)
    etaHat1 = r1 - X1 @ coef1
    _p(f"etaHat1  std={etaHat1.std():.3f}", t0)

    coef2, *_ = np.linalg.lstsq(X1, r2, rcond=None)
    etaHat2 = r2 - X1 @ coef2
    _p(f"etaHat2  std={etaHat2.std():.3f}", t0)

    ctrl = ~treated
    iTr = np.where(treated)[0]
    iCon = np.where(ctrl)[0]

    if len(iTr) == 0 or len(iCon) == 0:
        raise ValueError(
            f"Need both treated and control observations after cleaning/windowing; "
            f"got treated={len(iTr)}, control={len(iCon)}"
        )

    etaTr1 = etaHat1[iTr]
    etaTr2 = etaHat2[iTr]
    etaCon1 = etaHat1[iCon]
    etaCon2 = etaHat2[iCon]

    print("\n[2/3] TSplm2D on control ...", flush=True)
    hCon, bCon, mu1Con, bw1Con, mu2Con, bw2Con = TSplm2D(
        y[iCon], x_mat[iCon], etaCon1, etaCon2, kn, bwf, ridge=ridge, label="Con"
    )

    print("\n[2/3] TSplm2D on treated ...", flush=True)
    hTr, bTr, mu1Tr, bw1Tr, mu2Tr, bw2Tr = TSplm2D(
        y[iTr], x_mat[iTr], etaTr1, etaTr2, kn, bwf, ridge=ridge, label="Tr"
    )

    print("\n[3/3] Computing ewA_dual ...", flush=True)

    def basis_at(e1, e2, mu1, bw1, mu2, bw2):
        A = norm.pdf(e1[:, None] - mu1[None, :], scale=bw1)
        B = norm.pdf(e2[:, None] - mu2[None, :], scale=bw2)
        return (A[:, :, None] * B[:, None, :]).reshape(len(e1), kn * kn)

    dmt = basis_at(etaTr1, etaTr2, mu1Tr, bw1Tr, mu2Tr, bw2Tr)
    dmc = basis_at(etaTr1, etaTr2, mu1Con, bw1Con, mu2Con, bw2Con)

    tr_own = basis_at(etaTr1, etaTr2, mu1Tr, bw1Tr, mu2Tr, bw2Tr).sum(1)
    con_own = basis_at(etaCon1, etaCon2, mu1Con, bw1Con, mu2Con, bw2Con).sum(1)
    thr_tr = np.percentile(tr_own, support_pct)
    thr_con = np.percentile(con_own, support_pct)

    insupp = (dmt.sum(1) > thr_tr) & (dmc.sum(1) > thr_con)
    _p(
        f"support: {insupp.sum():,}/{len(iTr):,} treated obs "
        f"(thr_tr={thr_tr:.2e}  thr_con={thr_con:.2e})",
        t0,
    )

    if insupp.any():
        vTr_vals = dmt[insupp] @ hTr
        vCon_vals = dmc[insupp] @ hCon
        alpha_vals = vTr_vals - vCon_vals
        ewA_val = float(np.mean(alpha_vals))
        eta1_insupp = etaTr1[insupp]
        eta2_insupp = etaTr2[insupp]
        margin_insupp = np.minimum(eta1_insupp, eta2_insupp)
    else:
        vTr_vals = np.array([])
        vCon_vals = np.array([])
        alpha_vals = np.array([])
        ewA_val = np.nan
        eta1_insupp = np.array([])
        eta2_insupp = np.array([])
        margin_insupp = np.array([])

    _p(f"ewA_dual = {ewA_val:.6f}", t0)
    _p(f"net gain = {ewA_val - cost:.6f}  (ewA - cost={cost})", t0)

    info = dict(
        ewA_dual=ewA_val,
        net_gain=ewA_val - cost if not np.isnan(ewA_val) else np.nan,
        cost=cost,
        fit_window1=fit_window1,
        fit_window2=fit_window2,
        ridge=ridge,
        n_after_clean=n,
        n_treated=len(iTr),
        n_control=len(iCon),
        n_insupp=int(insupp.sum()),
        kept_x_cols=keep_cols,
        alpha_vals=alpha_vals,
        vTr_vals=vTr_vals,
        vCon_vals=vCon_vals,
        insupp=insupp,
        etaHat1=etaHat1,
        etaHat2=etaHat2,
        treated=treated,
        etaTr1=etaTr1,
        etaTr2=etaTr2,
        etaCon1=etaCon1,
        etaCon2=etaCon2,
        eta1_insupp=eta1_insupp,
        eta2_insupp=eta2_insupp,
        margin_insupp=margin_insupp,
        hTr=hTr,
        hCon=hCon,
        mu1Tr=mu1Tr,
        bw1Tr=bw1Tr,
        mu2Tr=mu2Tr,
        bw2Tr=bw2Tr,
        mu1Con=mu1Con,
        bw1Con=bw1Con,
        mu2Con=mu2Con,
        bw2Con=bw2Con,
        bCon=bCon,
        bTr=bTr,
    )
    return ewA_val, info


# ─────────────────────────────────────────────────────────────────────────────
# prepare_data  — build arrays from dftest with local windows
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data(
    dftest,
    X_col,
    y_col,
    score1_col,
    score2_col,
    c1,
    c2,
    fit_window1=10,
    fit_window2=100000,
):
    """
    Prepare (y, x_mat, r1, r2, treated) from dftest.

    By default only keeps rows satisfying:
        |score1 - c1| <= 10
        |score2 - c2| <= 100000
    """
    t0 = time.time()
    print("\n[0/3] Preparing data ...", flush=True)

    d = dftest.copy()
    for col in [y_col, score1_col, score2_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    n_raw = len(d)
    d = d.dropna(subset=[y_col, score1_col, score2_col]).reset_index(drop=True)
    _p(f"dropped {n_raw-len(d):,} NA rows  →  n={len(d):,}", t0)

    d["_r1"] = d[score1_col] - c1
    d["_r2"] = d[score2_col] - c2

    if fit_window1 is not None or fit_window2 is not None:
        win_mask = np.ones(len(d), dtype=bool)
        if fit_window1 is not None:
            win_mask &= d["_r1"].abs().to_numpy() <= float(fit_window1)
        if fit_window2 is not None:
            win_mask &= d["_r2"].abs().to_numpy() <= float(fit_window2)

        before = len(d)
        d = d.loc[win_mask].reset_index(drop=True)
        _p(
            f"window kept {len(d):,}/{before:,} rows "
            f"(|r1|<={fit_window1}, |r2|<={fit_window2})",
            t0,
        )

    r1 = d["_r1"].to_numpy(dtype=float)
    r2 = d["_r2"].to_numpy(dtype=float)
    treated = (r1 > 0) & (r2 > 0)
    y = d[y_col].to_numpy(dtype=float)

    _p(
        f"treated (AND): {treated.sum():,}  control: {(~treated).sum():,}  "
        f"({100*treated.mean():.1f}%)",
        t0,
    )

    avail = [c for c in X_col if c in d.columns]
    cat_cols = d[avail].select_dtypes(include=["category", "object"]).columns.tolist()
    xdf = d[avail].copy()

    for c in avail:
        if c not in cat_cols:
            xdf[c] = pd.to_numeric(xdf[c], errors="coerce")

    if cat_cols:
        xdf = pd.get_dummies(xdf, columns=cat_cols, drop_first=True, dummy_na=False)

    for c in xdf.columns:
        if xdf[c].isna().any():
            if xdf[c].dtype.kind in "fi":
                xdf[c] = xdf[c].fillna(xdf[c].mean())
            else:
                xdf[c] = xdf[c].fillna(0)

    x_mat = xdf.to_numpy(dtype=float)
    _p(f"X matrix: {x_mat.shape[0]:,} x {x_mat.shape[1]}", t0)

    return y, x_mat, r1, r2, treated


# ─────────────────────────────────────────────────────────────────────────────
# plot  — same diagnostics, now with window metadata if available
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(info, dftest=None, score1_col=None, score2_col=None,
                 c1=80.1, c2=766550, y_col="y", cost=None):
    from matplotlib.patches import Patch
    utility_grid_size = 100
    eta1 = info["etaHat1"]
    eta2 = info["etaHat2"]
    treated = info["treated"]
    alpha = info["alpha_vals"]
    _cost = cost if cost is not None else info.get("cost", 0.0)

    eta1_insupp = info.get("eta1_insupp", np.array([]))
    eta2_insupp = info.get("eta2_insupp", np.array([]))
    margin_insupp = info.get("margin_insupp", np.array([]))
    fit_window1 = info.get("fit_window1", None)
    fit_window2 = info.get("fit_window2", None)

    q1_lo, q1_hi = np.nanpercentile(eta1, [1, 99])
    q2_lo, q2_hi = np.nanpercentile(eta2, [1, 99])

    plot_mask = (
        (eta1 >= q1_lo) & (eta1 <= q1_hi) &
        (eta2 >= q2_lo) & (eta2 <= q2_hi)
    )

    eta1_plot = eta1[plot_mask]
    eta2_plot = eta2[plot_mask]
    treated_plot = treated[plot_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    axes = axes.ravel()

    # (1) residualized running variables
    ax = axes[0]
    c = np.where(treated_plot, "steelblue", "salmon")
    ax.scatter(eta1_plot, eta2_plot, c=c, alpha=0.12, s=4, rasterized=True)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("etaHat1 (score1 residual)")
    ax.set_ylabel("etaHat2 (score2 residual)")
    ttl = "Residualised running variables (trimmed 1%-99%)"
    if fit_window1 is not None or fit_window2 is not None:
        ttl += f"\nfit windows: ({fit_window1}, {fit_window2})"
    ax.set_title(ttl)
    ax.legend(
        handles=[
            Patch(color="steelblue", label="Treated"),
            Patch(color="salmon", label="Control")
        ],
        fontsize=8
    )

    # (2) alpha distribution
    ax = axes[1]
    if len(alpha):
        a_lo, a_hi = np.nanpercentile(alpha, [1, 99])
        alpha_plot = alpha[(alpha >= a_lo) & (alpha <= a_hi)]
        ax.hist(alpha_plot, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(info["ewA_dual"], color="red", lw=1.5,
                   label=f"ewA_dual={info['ewA_dual']:.4f}")
        ax.axvline(_cost, color="orange", lw=1.5, ls="--",
                   label=f"cost={_cost:.4f}")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.legend(fontsize=9)
    ax.set_title("alpha_hat distribution (trimmed 1%-99%)")
    ax.set_xlabel("alpha_hat")

    # (3) mean y by quadrant
    ax = axes[2]
    if dftest is not None and score1_col and score2_col:
        d = dftest.copy()
        d[score1_col] = pd.to_numeric(d[score1_col], errors="coerce")
        d[score2_col] = pd.to_numeric(d[score2_col], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[y_col, score1_col, score2_col])

        if fit_window1 is not None:
            d = d[np.abs(d[score1_col] - c1) <= fit_window1]
        if fit_window2 is not None:
            d = d[np.abs(d[score2_col] - c2) <= fit_window2]

        d["quad"] = ((d[score1_col] > c1).astype(int) * 2 +
                     (d[score2_col] > c2).astype(int))
        labels = {0: "(-,-)", 1: "(-,+)", 2: "(+,-)", 3: "Treated\n(+,+)"}
        colors = ["salmon", "lightsalmon", "lightsalmon", "steelblue"]
        means = d.groupby("quad")[y_col].mean()

        ax.bar([labels.get(q, str(q)) for q in means.index],
               means.values,
               color=colors[:len(means)])
        ax.set_ylabel(f"Mean {y_col}")
    ax.set_title("Mean y by quadrant")





def _moving_average(y, w=5):
    y = np.asarray(y, float)
    if w is None or w <= 1 or len(y) == 0:
        return y.copy()
    out = np.empty_like(y)
    h = w // 2
    for i in range(len(y)):
        lo = max(0, i - h)
        hi = min(len(y), i + h + 1)
        out[i] = np.mean(y[lo:hi])
    return out


def _get_std_indices(info):
    """
    返回标准化后的二维 index:
        z1 = eta1 / bw1_ref
        z2 = eta2 / bw2_ref
        m  = min(z1, z2)

    这里用 treated/control 平均带宽做统一尺度。
    """
    eta1 = np.asarray(info["eta1_insupp"], dtype=float).reshape(-1)
    eta2 = np.asarray(info["eta2_insupp"], dtype=float).reshape(-1)

    bw1_ref = 0.5 * (float(info["bw1Tr"]) + float(info["bw1Con"]))
    bw2_ref = 0.5 * (float(info["bw2Tr"]) + float(info["bw2Con"]))

    if bw1_ref <= 0 or bw2_ref <= 0:
        raise ValueError("Bandwidth must be positive.")

    z1 = eta1 / bw1_ref
    z2 = eta2 / bw2_ref
    m = np.minimum(z1, z2)

    return z1, z2, m, bw1_ref, bw2_ref


def plot_component_curves(
    info,
    x_axis="margin_std",      # "margin_std" / "z1" / "z2"
    bins=60,
    smooth_window=5,
    x_q=(0.01, 0.99),
    min_bin_count=8,
    figsize=(15, 6),
):
    """
    改正版：
    不再用原始 eta1 / eta2 / min(eta1,eta2)
    而是用标准化后的 z1, z2, margin_std=min(z1,z2)
    """
    z1, z2, m, bw1_ref, bw2_ref = _get_std_indices(info)

    vTr = np.asarray(info["vTr_vals"], dtype=float).reshape(-1)
    vCon = np.asarray(info["vCon_vals"], dtype=float).reshape(-1)
    alpha = np.asarray(info["alpha_vals"], dtype=float).reshape(-1)

    n = min(len(z1), len(z2), len(m), len(vTr), len(vCon), len(alpha))
    z1, z2, m = z1[:n], z2[:n], m[:n]
    vTr, vCon, alpha = vTr[:n], vCon[:n], alpha[:n]

    if x_axis == "z1":
        x = z1
        xlab = r"$z_1=\hat{\eta}_1 / \bar{bw}_1$"
    elif x_axis == "z2":
        x = z2
        xlab = r"$z_2=\hat{\eta}_2 / \bar{bw}_2$"
    elif x_axis == "margin_std":
        x = m
        xlab = r"$m=\min(z_1,z_2)$"
    else:
        raise ValueError("x_axis must be one of {'margin_std', 'z1', 'z2'}")

    ok = np.isfinite(x) & np.isfinite(vTr) & np.isfinite(vCon) & np.isfinite(alpha)
    x, vTr, vCon, alpha = x[ok], vTr[ok], vCon[ok], alpha[ok]

    if len(x) == 0:
        raise ValueError("No finite observations to plot.")

    ql, qh = x_q
    lo, hi = np.nanquantile(x, [ql, qh])
    keep = (x >= lo) & (x <= hi)

    x, vTr, vCon, alpha = x[keep], vTr[keep], vCon[keep], alpha[keep]
    if len(x) == 0:
        raise ValueError("No observations left after quantile filtering.")

    edges = np.linspace(lo, hi, bins + 1)
    which = np.digitize(x, edges) - 1
    which = np.clip(which, 0, bins - 1)

    x_mid, tr_mean, con_mean, a_mean = [], [], [], []
    for b in range(bins):
        mb = which == b
        if mb.sum() < min_bin_count:
            continue
        x_mid.append(np.mean(x[mb]))
        tr_mean.append(np.mean(vTr[mb]))
        con_mean.append(np.mean(vCon[mb]))
        a_mean.append(np.mean(alpha[mb]))

    x_mid = np.asarray(x_mid)
    tr_mean = np.asarray(tr_mean)
    con_mean = np.asarray(con_mean)
    a_mean = np.asarray(a_mean)

    if len(x_mid) == 0:
        raise ValueError("No bins have enough observations.")

    order = np.argsort(x_mid)
    x_mid = x_mid[order]
    tr_mean = tr_mean[order]
    con_mean = con_mean[order]
    a_mean = a_mean[order]

    tr_s = _moving_average(tr_mean, smooth_window)
    con_s = _moving_average(con_mean, smooth_window)
    a_s = _moving_average(a_mean, smooth_window)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(x_mid, tr_s, label=r"$vTr(x)$", linewidth=2)
    axes[0].plot(x_mid, con_s, label=r"$vCon(x)$", linewidth=2)
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Before subtraction: treated and control components")
    axes[0].legend()

    axes[1].plot(x_mid, a_s, label=r"$\hat{\alpha}(x)=vTr-vCon$", linewidth=2)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(r"$\hat{\alpha}(x)$")
    axes[1].set_title("Difference curve")
    axes[1].legend()

    fig.suptitle(
        f"Component curves | x_axis={x_axis} | bins={bins} | x_q={x_q} | "
        f"ewA={info['ewA_dual']:.4f}\n"
        f"(bw1_ref={bw1_ref:.4f}, bw2_ref={bw2_ref:.4f})",
        y=1.02
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_alpha_scatter_std(
    info,
    x_axis="margin_std",      # "margin_std" / "z1" / "z2"
    x_q=(0.01, 0.99),
    figsize=(8, 5),
):
    """
    替代原来的 alpha_hat vs threshold margin 图
    """
    z1, z2, m, bw1_ref, bw2_ref = _get_std_indices(info)
    alpha = np.asarray(info["alpha_vals"], dtype=float).reshape(-1)

    n = min(len(z1), len(z2), len(m), len(alpha))
    z1, z2, m, alpha = z1[:n], z2[:n], m[:n], alpha[:n]

    if x_axis == "z1":
        x = z1
        xlab = r"$z_1=\hat{\eta}_1 / \bar{bw}_1$"
    elif x_axis == "z2":
        x = z2
        xlab = r"$z_2=\hat{\eta}_2 / \bar{bw}_2$"
    elif x_axis == "margin_std":
        x = m
        xlab = r"$m=\min(z_1,z_2)$"
    else:
        raise ValueError("x_axis must be one of {'margin_std', 'z1', 'z2'}")

    ok = np.isfinite(x) & np.isfinite(alpha)
    x, alpha = x[ok], alpha[ok]

    ql, qh = x_q
    lo, hi = np.nanquantile(x, [ql, qh])
    keep = (x >= lo) & (x <= hi)
    x, alpha = x[keep], alpha[keep]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, alpha, s=10, alpha=0.25)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel("alpha_hat")
    ax.set_title("alpha_hat vs standardized threshold index")
    plt.tight_layout()
    plt.show()
    return fig


def plot_alpha_heatmap(info, gridsize=40, figsize=(7, 6)):
    """
    真正更合理的双阈值可视化：
    在 (z1, z2) 上画 alpha 的二维热图
    """
    z1, z2, _, _, _ = _get_std_indices(info)
    alpha = np.asarray(info["alpha_vals"], dtype=float).reshape(-1)

    n = min(len(z1), len(z2), len(alpha))
    z1, z2, alpha = z1[:n], z2[:n], alpha[:n]

    ok = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(alpha)
    z1, z2, alpha = z1[ok], z2[ok], alpha[ok]

    x_edges = np.linspace(np.quantile(z1, 0.01), np.quantile(z1, 0.99), gridsize + 1)
    y_edges = np.linspace(np.quantile(z2, 0.01), np.quantile(z2, 0.99), gridsize + 1)

    H = np.full((gridsize, gridsize), np.nan)
    C = np.zeros((gridsize, gridsize), dtype=int)

    ix = np.digitize(z1, x_edges) - 1
    iy = np.digitize(z2, y_edges) - 1

    for i in range(len(alpha)):
        if 0 <= ix[i] < gridsize and 0 <= iy[i] < gridsize:
            if np.isnan(H[iy[i], ix[i]]):
                H[iy[i], ix[i]] = alpha[i]
            else:
                H[iy[i], ix[i]] += alpha[i]
            C[iy[i], ix[i]] += 1

    mask = C > 0
    H[mask] = H[mask] / C[mask]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$z_1=\hat{\eta}_1 / \bar{bw}_1$")
    ax.set_ylabel(r"$z_2=\hat{\eta}_2 / \bar{bw}_2$")
    ax.set_title(r"Heatmap of $\hat{\alpha}(z_1,z_2)$")
    plt.colorbar(im, ax=ax, label="alpha_hat")
    plt.tight_layout()
    plt.show()
    return fig

import numpy as np
import matplotlib.pyplot as plt

def _gauss_basis_1d(x, mu, bw):
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    bw = float(bw)
    u = (x[:, None] - mu[None, :]) / bw
    return np.exp(-0.5 * u * u)

def _moving_average(y, w=5):
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1 or len(y) == 0:
        return y.copy()
    out = np.empty_like(y)
    h = w // 2
    for i in range(len(y)):
        lo = max(0, i - h)
        hi = min(len(y), i + h + 1)
        out[i] = np.mean(y[lo:hi])
    return out

def plot_fullsample_local(
    info,
    x_axis="margin_std",     # "z1" / "z2" / "margin_std"
    bins=60,
    x_q=(0.01, 0.99),
    smooth_window=5,
    min_bin_count=10,
    figsize=(15, 10)
):
    """
    Plot the full-sample figure under the previously defined local window.

    Here, "full sample" means all points corresponding to
    info["etaHat1"] and info["etaHat2"].
    """

    # ===== 1) Full sample under the local window =====
    eta1 = np.asarray(info["etaHat1"], dtype=float).reshape(-1)
    eta2 = np.asarray(info["etaHat2"], dtype=float).reshape(-1)
    treated = np.asarray(info["treated"], dtype=int).reshape(-1)

    n = min(len(eta1), len(eta2), len(treated))
    eta1, eta2, treated = eta1[:n], eta2[:n], treated[:n]

    # ===== 2) Standardized z1 and z2 =====
    bw1_ref = 0.5 * (float(info["bw1Tr"]) + float(info["bw1Con"]))
    bw2_ref = 0.5 * (float(info["bw2Tr"]) + float(info["bw2Con"]))

    z1 = eta1 / bw1_ref
    z2 = eta2 / bw2_ref
    margin_std = np.minimum(z1, z2)

    # ===== 3) Reconstruct full-sample alpha using full-sample etaHat =====
    A_tr = _gauss_basis_1d(eta1, info["mu1Tr"], info["bw1Tr"])   # (n, kn)
    B_tr = _gauss_basis_1d(eta2, info["mu2Tr"], info["bw2Tr"])   # (n, kn)
    Dtr = (A_tr[:, :, None] * B_tr[:, None, :]).reshape(len(eta1), -1)   # (n, kn*kn)

    A_con = _gauss_basis_1d(eta1, info["mu1Con"], info["bw1Con"]) # (n, kn)
    B_con = _gauss_basis_1d(eta2, info["mu2Con"], info["bw2Con"]) # (n, kn)
    Dcon = (A_con[:, :, None] * B_con[:, None, :]).reshape(len(eta1), -1) # (n, kn*kn)

    hTr = np.asarray(info["hTr"], dtype=float).reshape(-1)
    hCon = np.asarray(info["hCon"], dtype=float).reshape(-1)

    vTr_full = Dtr @ hTr
    vCon_full = Dcon @ hCon
    alpha_full = vTr_full - vCon_full

    alpha_full = vTr_full - vCon_full

    # ===== 4) Choose x-axis =====
    if x_axis == "z1":
        x = z1
        xlab = r"$z_1=\hat{\eta}_1/\bar{bw}_1$"
    elif x_axis == "z2":
        x = z2
        xlab = r"$z_2=\hat{\eta}_2/\bar{bw}_2$"
    elif x_axis == "margin_std":
        x = margin_std
        xlab = r"$m=\min(z_1,z_2)$"
    else:
        raise ValueError("x_axis must be one of {'z1', 'z2', 'margin_std'}")

    # ===== 5) Keep only finite values =====
    ok = (
        np.isfinite(z1) & np.isfinite(z2) &
        np.isfinite(alpha_full) & np.isfinite(x)
    )
    z1, z2, treated, x, alpha_full = z1[ok], z2[ok], treated[ok], x[ok], alpha_full[ok]

    # ===== 6) Apply quantile restriction to the alpha curve =====
    ql, qh = x_q
    lo, hi = np.quantile(x, [ql, qh])
    keep = (x >= lo) & (x <= hi)

    x_plot = x[keep]
    alpha_plot = alpha_full[keep]

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(x_plot, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    mids = []
    a_mean = []
    counts = []
    for b in range(bins):
        mb = idx == b
        if mb.sum() >= min_bin_count:
            mids.append(np.mean(x_plot[mb]))
            a_mean.append(np.mean(alpha_plot[mb]))
            counts.append(mb.sum())

    mids = np.asarray(mids)
    a_mean = np.asarray(a_mean)

    if len(mids) > 0:
        order = np.argsort(mids)
        mids = mids[order]
        a_mean = a_mean[order]
        a_s = _moving_average(a_mean, smooth_window)
    else:
        a_s = np.array([])

    # ===== 7) plot =====
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # z1 histogram
    axes[0, 0].hist(z1, bins=50, alpha=0.8)
    axes[0, 0].axvline(0, linestyle="--", linewidth=1)
    axes[0, 0].set_title("Full local sample: z1 distribution")
    axes[0, 0].set_xlabel(r"$z_1=\hat{\eta}_1/\bar{bw}_1$")
    axes[0, 0].set_ylabel("Count")

    # z2 histogram
    axes[0, 1].hist(z2, bins=50, alpha=0.8)
    axes[0, 1].axvline(0, linestyle="--", linewidth=1)
    axes[0, 1].set_title("Full local sample: z2 distribution")
    axes[0, 1].set_xlabel(r"$z_2=\hat{\eta}_2/\bar{bw}_2$")
    axes[0, 1].set_ylabel("Count")

    # scatter z1 vs z2
    mask0 = treated == 0
    mask1 = treated == 1
    axes[1, 0].scatter(z1[mask0], z2[mask0], s=12, alpha=0.30, label="control")
    axes[1, 0].scatter(z1[mask1], z2[mask1], s=12, alpha=0.30, label="treated")
    axes[1, 0].axvline(0, linestyle="--", linewidth=1)
    axes[1, 0].axhline(0, linestyle="--", linewidth=1)
    axes[1, 0].set_title("Full local sample: z1 vs z2")
    axes[1, 0].set_xlabel(r"$z_1=\hat{\eta}_1/\bar{bw}_1$")
    axes[1, 0].set_ylabel(r"$z_2=\hat{\eta}_2/\bar{bw}_2$")
    axes[1, 0].legend()

    # alpha curve on full local sample
    axes[1, 1].scatter(x_plot, alpha_plot, s=10, alpha=0.15)
    if len(mids) > 0:
        axes[1, 1].plot(mids, a_s, linewidth=2)
    axes[1, 1].axhline(0, linestyle="--", linewidth=1)
    axes[1, 1].axvline(0, linestyle="--", linewidth=1)
    axes[1, 1].set_title("Full local sample: alpha curve")
    axes[1, 1].set_xlabel(xlab)
    axes[1, 1].set_ylabel(r"$\hat{\alpha}$")

    fig.suptitle(
        "Full sample under the local window\n"
        f"(n={len(z1)}, x_axis={x_axis}, x_q={x_q}, "
        f"bw1_ref={bw1_ref:.4f}, bw2_ref={bw2_ref:.4f})",
        y=1.02
    )
    plt.tight_layout()
    plt.show()

    return fig