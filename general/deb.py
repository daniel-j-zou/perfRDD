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


def pGen2(n, beta, gamma, x_dist, nu_spec, rng):
    """
    Generalized DGP with configurable x distribution and nu specification.

    x_dist:  "normal", "t3", "exp", "bimodal"
    nu_spec: "current", "eta_sq", "exp_eta", "independent", "linear"
    """
    # --- x (all standardized to unit variance) ---
    if x_dist == "normal":
        x = rng.normal(0, 1, n)
    elif x_dist == "t3":
        x = rng.standard_t(3, n) / np.sqrt(3)
    elif x_dist == "exp":
        x = rng.exponential(1.0, n) - 1.0  # mean 0, var 1
    elif x_dist == "bimodal":
        signs = rng.choice([-1.0, 1.0], size=n)
        x = rng.normal(signs * 1.5, 0.5, n) / np.sqrt(2.5)
    else:
        raise ValueError(f"Unknown x_dist: {x_dist}")

    # --- latent heterogeneity (unchanged) ---
    a_signs = rng.choice([-1.0, 1.0], size=n)
    a = a_signs * rng.exponential(1.0, n)
    eta = rng.normal(a, 1.0, n)

    # --- nu ---
    if nu_spec == "current":
        nu_mean = 3 * np.exp(a) / (1 + np.exp(a))
        nu = rng.normal(nu_mean, 1.0, n)
    elif nu_spec == "eta_sq":
        nu = rng.normal(eta ** 2, 1.0, n)
    elif nu_spec == "exp_eta":
        nu = rng.normal(np.exp(eta / 2), 1.0, n)
    elif nu_spec == "independent":
        nu = rng.normal(0, 1.0, n)
    elif nu_spec == "linear":
        nu = rng.normal(2 * eta, 1.0, n)
    else:
        raise ValueError(f"Unknown nu_spec: {nu_spec}")

    # --- w (unchanged) ---
    w_mean = 2 * np.exp(eta) / (1 + np.exp(eta))
    w = rng.normal(w_mean, 0.1, n)

    # --- treatment and outcome ---
    q = gamma * x + eta
    s = (q > 0).astype(float)
    y = s * w + beta * x + nu

    return x, y, eta, nu, w, q


def _basis_params(q, kn, support=None):
    """Compute B-spline basis parameters.

    support : optional (lo, hi) tuple for fixed knot range.
    """
    degree = 3
    if support is not None:
        lo, hi = support
    else:
        lo = np.min(q)
        hi = np.max(q)
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    return {"t": t, "degree": degree, "ql": lo, "qr_max": hi}


def _eval_basis(pts, info):
    """Evaluate B-spline basis functions at pts."""
    pts = np.asarray(pts, dtype=float)
    pts_c = np.clip(pts, info["ql"], info["qr_max"])
    return BSpline.design_matrix(pts_c, info["t"], info["degree"]).toarray()


# ---- Robinson-style PLM (from splines.py, simplified for B-splines only) ----

def TSplm_robinson(y, x, eta_hat, kn, support):
    """
    Robinson-style partial linear model:
        y = b * x + h(eta_hat) + error
    Two-step estimation with shifted basis for debiasing.

    Returns: (h, b, info)
    """
    y = np.asarray(y)
    x = np.asarray(x)
    eta_hat = np.asarray(eta_hat)

    info = _basis_params(eta_hat, kn, support=support)
    Phi = _eval_basis(eta_hat, info)

    # First regression: y ~ x + Phi
    H = np.column_stack((x, Phi))
    be1, *_ = np.linalg.lstsq(H, y, rcond=None)

    # Second regression with shifted basis
    Phi_shift = _eval_basis(eta_hat + 0.01 * x, info)
    Hs = np.column_stack((x, Phi_shift))
    diff = (Hs - H) @ be1
    H2 = np.column_stack((x, diff, H[:, 1:]))
    be, *_ = np.linalg.lstsq(H2, y, rcond=None)

    b = be[0]
    h = be[2:]
    return h, b, info


# ---- Naive PLM (no generated-regressor correction) ----

def TSplm_naive(y, x, eta_hat, kn, support):
    """
    Naive partial linear model -- no correction for eta being estimated.
        y = b * x + h(eta_hat) + error
    Single OLS of Y on [x, Phi(eta_hat)].

    Returns: (h, b, info)
    """
    y = np.asarray(y)
    x = np.asarray(x)
    eta_hat = np.asarray(eta_hat)

    info = _basis_params(eta_hat, kn, support=support)
    Phi = _eval_basis(eta_hat, info)

    H = np.column_stack((x, Phi))
    be, *_ = np.linalg.lstsq(H, y, rcond=None)

    b = be[0]
    h = be[1:]
    return h, b, info


# ---- Deb-style PLM (SCENTS method) ----

def TSplm_deb(y, x, eta_hat, kn, support):
    """
    Bias-corrected partial linear model using h'(eta) estimation.
        y = b * x + h(eta_hat) + error

    Steps:
      1. Initial OLS of Y on [x, Phi(eta_hat)] -> omega1
      2. Compute h'_hat by differentiating the fitted B-spline
      3. Corrected regression: Y on [x, x_tilde, Phi(eta_hat)]
         where x_tilde = h'_hat(eta_hat) * x absorbs the bias
         from eta_hat != eta.

    Returns: (h, b, info)
    """
    y = np.asarray(y)
    x = np.asarray(x)
    eta_hat = np.asarray(eta_hat)

    info = _basis_params(eta_hat, kn, support=support)
    Phi = _eval_basis(eta_hat, info)

    # Step 1: Initial fit to get spline coefficients
    H1 = np.column_stack((x, Phi))
    be1, *_ = np.linalg.lstsq(H1, y, rcond=None)
    omega1 = be1[1:]

    # Step 2: Compute h'_hat via B-spline derivative
    spl = BSpline(info["t"], omega1, info["degree"])
    spl_deriv = spl.derivative()
    eta_clipped = np.clip(eta_hat, info["ql"], info["qr_max"])
    h_prime = spl_deriv(eta_clipped)
    x_tilde = h_prime * x

    # Step 3: Corrected regression
    # Y = beta*x + delta*x_tilde + Phi*omega + error
    H2 = np.column_stack((x, x_tilde, Phi))
    be2, *_ = np.linalg.lstsq(H2, y, rcond=None)

    b = be2[0]
    h = be2[2:]   # skip beta and delta
    return h, b, info


# ---- Plug-in estimators ----

def _run_plugin(etaTr, etaCon, y, x, iTr, iCon, kn, support, method="robinson"):
    """Run the plug-in estimator E[W|Q>0] = E[h_Tr(eta) - h_Con(eta) | treated].

    method: "robinson", "naive", or "deb"
    Returns: ewA (scalar estimate)
    """
    if method == "deb":
        fit_fn = TSplm_deb
    elif method == "naive":
        fit_fn = TSplm_naive
    else:
        fit_fn = TSplm_robinson

    hCon, bCon, infoCon = fit_fn(y[iCon], x[iCon], etaCon, kn, support)
    hTr, bTr, infoTr = fit_fn(y[iTr], x[iTr], etaTr, kn, support)

    # Evaluate alpha(eta) = h_Tr(eta) - h_Con(eta) on treated, clipped to support
    etaTr_eval = np.clip(etaTr, support[0], support[1])
    dmt = _eval_basis(etaTr_eval, infoTr)
    dmc = _eval_basis(etaTr_eval, infoCon)

    # Support check
    in_tr = (etaTr_eval >= infoTr["ql"]) & (etaTr_eval <= infoTr["qr_max"])
    in_con = (etaTr_eval >= infoCon["ql"]) & (etaTr_eval <= infoCon["qr_max"])
    insupp = in_tr & in_con

    if insupp.any():
        return np.mean(dmt[insupp] @ hTr - dmc[insupp] @ hCon)
    else:
        return np.nan


def main():
    rng = np.random.default_rng(2025)

    N = 50000
    R = 500
    Beta = 2.0
    Gamma = 2.0
    ETA_SUPPORT = (-2.8, 2.8)
    kn_s = max(4, int(round(N ** (1.0 / 3.0))))

    x_dists = ["normal", "t3", "exp", "bimodal"]
    nu_specs = ["current", "eta_sq", "exp_eta", "independent", "linear"]

    print(f"N = {N},  R = {R},  gamma = {Gamma},  kn = {kn_s}\n")
    print(f"{'x_dist':>10s} {'nu_spec':>12s} {'n_ok':>5s} {'Cor_Tr':>7s} "
          f"{'Bias_R':>8s} {'Bias_N':>8s} "
          f"{'Std_R':>8s} {'Std_N':>8s} {'VarRatio':>9s}")
    print("-" * 85)

    results = []

    for x_dist in x_dists:
        for nu_spec in nu_specs:
            ew0 = np.zeros(R)
            ewA_rob = np.zeros(R)
            ewA_naive = np.zeros(R)
            cors = np.zeros(R)

            for r in range(R):
                x, y, eta, nu, w, q = pGen2(
                    N, Beta, Gamma, x_dist, nu_spec, rng
                )

                s = (q > 0)
                idx_all = np.arange(N)
                iTr = idx_all[s]
                iCon = idx_all[~s]

                ew0[r] = w[iTr].mean()

                # First stage
                X_design = np.column_stack((np.ones(N), x))
                coef_q, *_ = np.linalg.lstsq(X_design, q, rcond=None)
                b0_hat, b1_hat = coef_q
                etaHat = q - b0_hat - b1_hat * x

                cors[r] = np.corrcoef(x[iTr], etaHat[iTr])[0, 1]

                etaCon = etaHat[iCon]
                etaTr = etaHat[iTr]

                ewA_rob[r] = _run_plugin(
                    etaTr, etaCon, y, x, iTr, iCon, kn_s, ETA_SUPPORT,
                    method="robinson"
                )
                ewA_naive[r] = _run_plugin(
                    etaTr, etaCon, y, x, iTr, iCon, kn_s, ETA_SUPPORT,
                    method="naive"
                )

            ok = (np.abs(ewA_rob - ew0) < 1.0) & (np.abs(ewA_naive - ew0) < 1.0)
            dr = ewA_rob[ok] - ew0[ok]
            dn = ewA_naive[ok] - ew0[ok]

            n_ok = ok.sum()
            bias_r = np.mean(dr)
            bias_n = np.mean(dn)
            std_r = np.std(dr)
            std_n = np.std(dn)
            var_ratio = std_r ** 2 / std_n ** 2 if std_n > 0 else np.nan
            cor_mean = np.mean(cors[ok])

            print(f"{x_dist:>10s} {nu_spec:>12s} {n_ok:>5d} {cor_mean:>7.3f} "
                  f"{bias_r:>8.4f} {bias_n:>8.4f} "
                  f"{std_r:>8.4f} {std_n:>8.4f} {var_ratio:>9.4f}")

            results.append({
                "x_dist": x_dist, "nu_spec": nu_spec,
                "n_ok": n_ok, "cor": cor_mean,
                "bias_r": bias_r, "bias_n": bias_n,
                "std_r": std_r, "std_n": std_n,
                "var_ratio": var_ratio,
            })

        print("-" * 85)

    # Summary sorted by distance from 1
    print("\n=== Summary: VarRatio sorted by distance from 1.0 ===")
    results.sort(key=lambda r: abs(r["var_ratio"] - 1.0), reverse=True)
    print(f"{'x_dist':>10s} {'nu_spec':>12s} {'Cor_Tr':>7s} "
          f"{'Std_R':>8s} {'Std_N':>8s} {'VarRatio':>9s}")
    for r in results:
        print(f"{r['x_dist']:>10s} {r['nu_spec']:>12s} {r['cor']:>7.3f} "
              f"{r['std_r']:>8.4f} {r['std_n']:>8.4f} {r['var_ratio']:>9.4f}")


if __name__ == "__main__":
    main()
