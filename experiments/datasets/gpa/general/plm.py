import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def pGen(n, beta, gamma, rng=None):
    """
    Python translation of the MATLAB pGen function.

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

    # latent heterogeneity a: +/- Exp(1)
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


def TSplm(y, x, q, kn, bwf):
    """
    Python translation of TSplm(y,x,q,kn,bwf) from p4.m

    Partial linear model:
        y = b * x + h(q) + error
    where h(q) is approximated by a Gaussian series with 'kn' knots.

    Returns:
      h : coefficients of the nonparametric component (length kn)
      b : scalar slope on x
      mu, bw : centers and bandwidth of the Gaussian basis
    """
    y = np.asarray(y)
    x = np.asarray(x)
    q = np.asarray(q)
    n = len(y)

    # Range and centers
    ql = np.min(q)
    qr = np.ptp(q)  # range (max-min)
    if qr == 0:
        qr = 1.0  # avoid zero range

    # centers mu and bandwidth bw
    # MATLAB: mu = ql + ((1:kn) - 0.5) * qr/kn
    idx = np.arange(1, kn + 1) - 0.5
    mu = ql + idx * qr / kn
    bw = bwf * qr / kn

    # Helper: Gaussian basis evaluated at q for all centers
    # norm.pdf(x, loc=0, scale=bw) but we pass (q - mu)
    Phi = norm.pdf((q[:, None] - mu[None, :]), scale=bw)

    # First regression: y ~ x + Phi
    H = np.column_stack((x, Phi))
    v = H.var(axis=0, ddof=1)
    # keep columns with sufficient variance (threshold relative to max of non-x columns)
    if len(v) > 1:
        thresh1 = 0.03 * np.max(v[1:])
    else:
        thresh1 = 0.0
    mask1 = v > thresh1
    H_sel = H[:, mask1]
    # Solve OLS for first step
    be1 = np.zeros(H.shape[1])
    if H_sel.shape[1] > 0:
        be1_sel, *_ = np.linalg.lstsq(H_sel, y, rcond=None)
        be1[mask1] = be1_sel

    # Second regression:
    # Hs = [x, normpdf(q + 0.01*x - mu, bw)]
    Phi_shift = norm.pdf((q[:, None] + 0.01 * x[:, None] - mu[None, :]), scale=bw)
    Hs = np.column_stack((x, Phi_shift))
    # H2 = [x, (Hs-H)*be1, H(:,2:end)]
    diff = (Hs - H) @ be1
    H2 = np.column_stack((x, diff, H[:, 1:]))

    v2 = H2.var(axis=0, ddof=1)
    # threshold relative to max of columns from 3rd onward
    if len(v2) > 2:
        thresh2 = 0.03 * np.max(v2[2:])
    else:
        thresh2 = 0.0
    mask2 = v2 > thresh2
    H2_sel = H2[:, mask2]

    be = np.zeros(H2.shape[1])
    if H2_sel.shape[1] > 0:
        be_sel, *_ = np.linalg.lstsq(H2_sel, y, rcond=None)
        be[mask2] = be_sel

    # be(1) is b, be(3:end) are h
    b = be[0]
    h = be[2:]  # length kn (if all kept); some may be effectively unused
    return h, b, mu, bw


def main():
    rng = np.random.default_rng(2025)

    # Parameters
    M = 1000       # number of Monte Carlo experiments (this is large; reduce if needed)
    Gamma = 2.0
    Beta = 2.0
    KN = 22
    BW = 1.0       # not used directly in TSplm; bwf argument is 1 there
    C = 1.0

    # Sample sizes: N(m) = round(linspace(sqrt(1000), sqrt(20000), M).^2)
    N_vals = np.round(
        np.linspace(np.sqrt(1000.0), np.sqrt(20000.0), M) ** 2
    ).astype(int)

    # Arrays to store results
    ew = np.zeros(M)      # matching-based E(W|Q>0)
    ewA = np.zeros(M)     # alpha(eta) plug-in estimate
    ew0 = np.zeros(M)     # oracle sample average of W for Q>0
    utilMaximizer = np.zeros(M)  # estimated optimal thresholds

    # Step 1: big sample to approximate "true" utility curve and optimal threshold
    x_big, y_big, eta_big, nu_big, w_big, q_big = pGen(100000, Beta, Gamma, rng=rng)
    order = np.argsort(q_big)
    q0 = q_big[order]
    y0 = y_big[order]

    # Utility as a function of threshold using big sample:
    # util0(i) = (sum_{j>i}(y_j - C)) / 100000
    diff = y0 - C
    total_sum = diff.sum()
    cums = np.cumsum(diff)  # cumulative from smallest q upward
    util0 = (total_sum - cums) / len(y0)

    umax = np.argmax(util0)
    optTh = q0[umax]

    # Plot oracle utility curve
    plt.figure(3)
    plt.plot(q0, util0)
    plt.xlabel("Threshold")
    plt.ylabel("Utility")
    plt.title(f"Utility_0 vs. threshold, C={C}")
    plt.axvline(optTh, color="red", linestyle="--", label=f"Opt Th={optTh:.3f}")
    plt.legend()

    # Grid for utility maximization in each experiment
    L = 100
    th = np.linspace(-2.0, 2.0, L)

    # Monte Carlo loop
    for m in range(M):
        n = N_vals[m]
        x, y, eta, nu, w, q = pGen(n, Beta, Gamma, rng=rng)

        s = (q > 0)
        idx_all = np.arange(n)
        iTr = idx_all[s]
        iCon = idx_all[~s]

        # oracle E[W|Q>0] in this sample
        nw0 = len(iTr)
        if nw0 > 0:
            ew0[m] = w[iTr].mean()
        else:
            ew0[m] = np.nan

        # First stage: regress q on [1, x] to estimate gamma and recover etaHat
        X_design = np.column_stack((np.ones(n), x))
        coef_q, *_ = np.linalg.lstsq(X_design, q, rcond=None)
        b0_hat, b1_hat = coef_q
        etaHat = q - b0_hat - b1_hat * x

        etaCon = etaHat[iCon]
        etaTr = etaHat[iTr]

        # Partial linear fits on controls and treated
        hCon, bCon, muCon, bwCon = TSplm(y[iCon], x[iCon], etaCon, KN, 1.0)
        hTr, bTr, muTr, bwTr = TSplm(y[iTr], x[iTr], etaTr, KN, 1.0)

        # Construct alpha(eta) estimates for treated sample
        dmt = norm.pdf((etaTr[:, None] - muTr[None, :]), scale=bwTr)
        dmc = norm.pdf((etaTr[:, None] - muCon[None, :]), scale=bwCon)
        insupp = (dmt.sum(axis=1) > 1.0) & (dmc.sum(axis=1) > 1.0)
        if insupp.any():
            ewA[m] = np.mean(dmt[insupp] @ hTr - dmc[insupp] @ hCon)
        else:
            ewA[m] = np.nan

        # Build empirical survival function Gnbar of g(X) ≈ q - etaHat
        gX_hat = q - etaHat
        xsl = gX_hat.min()
        xsr = gX_hat.max()
        # grid points t_j
        j_grid = np.arange(1, n + 1)
        t_j = xsl + (xsr - xsl) * j_grid / n
        Gnbar = (gX_hat[:, None] > t_j[None, :]).mean(axis=0)  # length n

        # utilVector(j) = alpha_hat(etaHat_j)
        utilVector = np.zeros(n)
        for j in range(n):
            eta_j = etaHat[j]
            v1 = norm.pdf((eta_j - muTr), scale=bwTr) @ hTr
            v2 = norm.pdf((eta_j - muCon), scale=bwCon) @ hCon
            utilVector[j] = v1 - v2

        # Approximate U(phi) = E[(alpha(eta) - C) * Gbar(phi - eta)]
        util = np.zeros(L)
        for l_idx, phi in enumerate(th):
            # indices k for each observation
            k = np.round((phi - etaHat - xsl) / (xsr - xsl) * n).astype(int)
            # clip to [1, n]
            k = np.clip(k, 1, n)
            om = Gnbar[k - 1]  # survival values
            util[l_idx] = np.mean((utilVector - C) * om)

        utilMaximizer[m] = th[np.argmax(util)]

        # Matching-based estimator of E[W|Q>0]
        dw = np.zeros(len(iTr))
        c_indicator = np.zeros(len(iTr), dtype=bool)
        for i_idx, i_tr in enumerate(iTr):
            # nearest neighbor in etaCon
            diff_eta = np.abs(etaCon - etaTr[i_idx])
            j_idx = np.argmin(diff_eta)
            am = diff_eta[j_idx]
            j_con = iCon[j_idx]
            # y(iTr) - y(iCon) - bTr*x(iTr) + bCon*x(iCon)
            dw[i_idx] = (
                y[i_tr]
                - y[j_con]
                - bTr * x[i_tr]
                + bCon * x[j_con]
            )
            c_indicator[i_idx] = am < 1.0  # same condition as in MATLAB (can tweak)

        if c_indicator.any():
            ew[m] = (dw[c_indicator].sum()) / c_indicator.sum()
        else:
            ew[m] = np.nan

        if (m + 1) % 100 == 0:
            print(f"Completed replication {m+1}/{M}")

    # Post-processing and plots (similar to MATLAB)

    # Filter out extreme runs
    r = np.abs(ewA - ew0) < 1.0
    # N-weighted oracle mean
    ew0n = np.nansum(N_vals[r] * ew0[r]) / np.sum(N_vals[r])

    plt.figure(1)
    plt.clf()
    plt.title("E(W|Q>0) estimators vs N")

    # Left axis: sqrt(N)*(est - ew0n)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(N_vals[r], np.sqrt(N_vals[r]) * (ew[r] - ew0n),
             'b.', alpha=0.5, label="sqrt(N)*(ew - ew0n)")
    ax1.plot(N_vals[r], np.sqrt(N_vals[r]) * (ewA[r] - ew0n),
             'r*', alpha=0.5, label="sqrt(N)*(ewA - ew0n)")
    ax1.set_ylabel("Normalized E(W|Q>0)")

    ax2.plot(N_vals[r], ew[r], 'g.', alpha=0.5, label="ew")
    ax2.set_ylabel("E(W|Q>0)")
    ax1.set_xlabel("N")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Histograms
    plt.figure(2)
    plt.clf()
    # more aggressive filter as in MATLAB
    r2 = (np.abs(ewA - ew0) < 1.0) & (np.abs(ew - ew0) < 1.0)
    # drop first half in analogy to MATLAB r(1:M/2)=false
    r2[: M // 2] = False

    plt.subplot(1, 2, 1)
    plt.hist(ew[r2], bins=30, alpha=0.5, label="ew")
    plt.hist(ewA[r2], bins=30, alpha=0.5, label="ewA")
    plt.hist(ew0[r2], bins=30, alpha=0.5, label="ew0")
    plt.legend()
    plt.title("Distributions of estimators")

    plt.subplot(1, 2, 2)
    plt.hist(ew[r2] - ewA[r2], bins=30, alpha=0.5, label="ew - ewA")
    plt.hist(ew[r2] - ew0[r2], bins=30, alpha=0.5, label="ew - ew0")
    plt.legend()
    plt.title("Differences")

    # Histogram of estimated optimal thresholds vs Q distribution & oracle util
    plt.figure(4)
    plt.clf()
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.hist(q0, bins=50, density=True, histtype='step', edgecolor='red', label="Q")
    # drop first half for utilMaximizer histogram as in MATLAB
    ax1.hist(utilMaximizer[M // 2:], bins=50, density=True,
             histtype='step', edgecolor='green', label="OptTh est")

    ax2.plot(q0, util0, 'c-', label="utility")

    ax1.set_xlabel("Q / threshold")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("Utility")
    ax1.set_title("Estimated Optimal Threshold vs Q and Utility")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.show()


if __name__ == "__main__":
    main()
