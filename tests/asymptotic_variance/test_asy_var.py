"""
test_asy_var.py

Test the theoretical asymptotic variance of the perfRDD threshold estimator.

Theory (prefRDD.tex):
    sqrt(n) * (phi_hat - phi_star)  -->  N(0, sigma_dec^2 / U''(phi_star)^2)

where
    sigma_dec^2 = sigma_0^2 + sigma_alpha^2 + sigma_g^2 + sigma_EIV^2

Components:
    sigma_0^2     = E[(alpha(eta)-c)^2 g(phi*-eta)^2]      (utility fluctuation)
    sigma_alpha^2 = Var(R_Y * r_alpha)                      (alpha estimation)
    sigma_g^2     = Var(r_g(T))                             (density estimation)
    sigma_EIV^2   = 0  when X independent of eta

DGP (simplified):
    X   ~ N(0, 1),  eta ~ N(0, 1),  X independent of eta
    W | eta ~ N(alpha(eta), sigma_w^2),  alpha(eta) = 2*sigmoid(eta)
    nu  ~ N(0, sigma_nu^2)  independent
    Q   = gamma*X + eta,  D = 1{Q > 0}
    Y   = D*W + beta*X + nu

With c = 1:  phi_star = 0  by symmetry (alpha(eta) - 1 is odd).

Output (saved to this directory):
    asy_var_convergence.png   -- n*Var(phi_hat) vs theory across N
    asy_var_qq.png            -- QQ plots at each N
    asy_var_components.png    -- bar chart of variance components
    asy_var_table.txt         -- summary table
"""

import os
import numpy as np
from scipy.stats import norm as sp_norm
from scipy.integrate import quad
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

GAMMA = 1.0
BETA = 1.0
SIGMA_ETA = 1.0
SIGMA_W = 0.1
SIGMA_NU = 1.0
C = 1.0
PHI_STAR = 0.0           # by symmetry

N_VALUES = [10000, 20000, 50000, 100000, 200000]
R = 300                   # MC replications per N
PHI_GRID = np.linspace(-3.0, 3.0, 2000)   # fine grid for utility maximization

N_MC = 2_000_000          # large sample for Riesz representer computation


# ═══════════════════════════════════════════════════════════════════════════
# True functions
# ═══════════════════════════════════════════════════════════════════════════

def sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-eta))

def alpha_true(eta):
    """True treatment effect: E[W | eta] = 2*sigmoid(eta)."""
    return 2.0 * sigmoid(eta)

def alpha_minus_c(eta):
    return alpha_true(eta) - C

def g_density(t):
    """Density of T = gamma*X ~ N(0, gamma^2)."""
    return sp_norm.pdf(t, 0, GAMMA)

def g_prime(t):
    """Derivative of g: g'(t) = -(t/gamma^2) * g(t)."""
    return -(t / GAMMA**2) * g_density(t)

def f_eta(eta):
    """Density of eta ~ N(0, sigma_eta^2)."""
    return sp_norm.pdf(eta, 0, SIGMA_ETA)


# ═══════════════════════════════════════════════════════════════════════════
# Theoretical variance components (quadrature)
# ═══════════════════════════════════════════════════════════════════════════

def compute_U_prime(phi):
    """U'(phi) = -E[(alpha(eta)-c) * g(phi-eta)]."""
    integrand = lambda eta: alpha_minus_c(eta) * g_density(phi - eta) * f_eta(eta)
    val, _ = quad(integrand, -10, 10, limit=200)
    return -val

def compute_U_double_prime(phi):
    """U''(phi) = -E[(alpha(eta)-c) * g'(phi-eta)]."""
    integrand = lambda eta: alpha_minus_c(eta) * g_prime(phi - eta) * f_eta(eta)
    val, _ = quad(integrand, -10, 10, limit=200)
    return -val

def compute_sigma_0_sq():
    """sigma_0^2 = E[(alpha(eta)-c)^2 * g(phi*-eta)^2]."""
    integrand = lambda eta: (alpha_minus_c(eta)**2
                             * g_density(PHI_STAR - eta)**2
                             * f_eta(eta))
    val, _ = quad(integrand, -10, 10, limit=200)
    return val

def compute_sigma_g_sq():
    """
    sigma_g^2 = Var_T(r_g(T))  where  T = gamma*X ~ N(0, gamma^2).

    r_g(t) = -integral_{phi*-t}^{inf} (alpha(eta)-c) f_eta(eta) deta.
    """
    def r_g(t):
        val, _ = quad(lambda eta: alpha_minus_c(eta) * f_eta(eta),
                      PHI_STAR - t, 10, limit=200)
        return -val

    # E[r_g(T)^2] where T ~ N(0, gamma^2)
    integrand_sq = lambda t: r_g(t)**2 * sp_norm.pdf(t, 0, GAMMA)
    E_rg_sq, _ = quad(integrand_sq, -8, 8, limit=200)

    # E[r_g(T)] -- should be ~0 since U'(phi*)=0
    integrand_mu = lambda t: r_g(t) * sp_norm.pdf(t, 0, GAMMA)
    E_rg, _ = quad(integrand_mu, -8, 8, limit=200)

    return E_rg_sq - E_rg**2


def compute_sigma_alpha_sq(seed=0):
    """
    sigma_alpha^2 = E[R_Y^2 * r_alpha^2]  via large-sample Riesz representer.

    In the stacked PLM:  H = [X, Phi_base(eta), D*Phi_treat(eta)]
    The functional:  L(omega_treat) = sum_k a_k * omega_treat_k
      where  a_k = E[Phi_treat_k(eta) * g(phi*-eta)]

    The Riesz representer score:  r_i = a' (H'H/n)^{-1} H_i
    restricted to the omega_treat block.

    R_Y = D * R_W + nu,  Var(R_W | eta) = sigma_w^2.
    """
    rng = np.random.default_rng(seed)

    # Draw large sample from DGP (true eta, not eta_hat -- population limit)
    eta = rng.normal(0, SIGMA_ETA, N_MC)
    x = rng.normal(0, 1, N_MC)
    T = GAMMA * x
    D = (T + eta > PHI_STAR).astype(float)  # Q = gamma*X + eta > 0

    R_W = rng.normal(0, SIGMA_W, N_MC)     # W - alpha(eta)
    nu = rng.normal(0, SIGMA_NU, N_MC)
    R_Y = D * R_W + nu                      # regression residual

    # B-spline basis
    K = max(6, int(round(N_MC ** (1.0 / 3.0))))
    support = (np.percentile(eta, 0.5), np.percentile(eta, 99.5))
    lo, hi = support
    degree = 3
    interior = np.linspace(lo, hi, K + 2)[1:-1]
    t_knots = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    eta_c = np.clip(eta, lo, hi)
    Phi = BSpline.design_matrix(eta_c, t_knots, degree).toarray()  # (N_MC, K+degree)
    n_basis = Phi.shape[1]

    # Full stacked design including X (needed because D depends on X,
    # making D*Phi correlated with X — Frisch-Waugh is not valid here)
    DPhi = D[:, None] * Phi                           # (N_MC, n_basis)
    H = np.column_stack((x[:, None], Phi, DPhi))      # (N_MC, 1+2*n_basis)

    # Gram matrix (add small ridge for numerical stability at boundary knots)
    HtH = H.T @ H / N_MC
    ridge = 1e-8 * np.eye(HtH.shape[0])
    HtH_inv = np.linalg.solve(HtH + ridge, np.eye(HtH.shape[0]))

    # Functional gradient: a_k = E[Phi_k(eta) * g(phi*-eta)]
    g_vals = g_density(PHI_STAR - eta)                # (N_MC,)
    a_K = (Phi * g_vals[:, None]).mean(axis=0)        # (n_basis,)

    # Selector for omega_treat block: columns [1+n_basis : 1+2*n_basis]
    e_alpha = np.zeros(1 + 2 * n_basis)
    e_alpha[1 + n_basis:] = a_K

    # Riesz representer scores
    w_vec = HtH_inv @ e_alpha
    r_scores = H @ w_vec                              # (N_MC,)

    # sigma_alpha^2 = E[R_Y^2 * r^2]
    sigma_alpha_sq = np.mean(R_Y**2 * r_scores**2)
    return sigma_alpha_sq


# ═══════════════════════════════════════════════════════════════════════════
# DGP
# ═══════════════════════════════════════════════════════════════════════════

def pGen_simple(n, rng):
    """Simplified DGP with eta independent of X, nu independent of eta."""
    x = rng.normal(0, 1, n)
    eta = rng.normal(0, SIGMA_ETA, n)
    w = rng.normal(alpha_true(eta), SIGMA_W, n)
    nu = rng.normal(0, SIGMA_NU, n)
    q = GAMMA * x + eta
    D = (q > 0).astype(float)
    y = D * w + BETA * x + nu
    return x, y, q, D, eta, w, nu


# ═══════════════════════════════════════════════════════════════════════════
# Estimation (pooled PLM, matching theory)
# ═══════════════════════════════════════════════════════════════════════════

def _bspline_basis(eta, kn, support):
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    eta_c = np.clip(eta, lo, hi)
    Phi = BSpline.design_matrix(eta_c, t, degree).toarray()
    return Phi, t, degree


def estimate_phi_hat(x, y, q, D):
    """
    Full estimation pipeline:
      1. First stage: Q = [1,X] gamma + eta  =>  eta_hat
      2. Pooled PLM:  Y = X*beta + Phi(eta_hat)*omega_base + D*Phi(eta_hat)*omega_treat
      3. alpha_hat(eta) = Phi(eta)*omega_treat
      4. Utility maximization:  phi_hat = argmax U_hat(phi)
    Returns phi_hat or None on failure.
    """
    n = len(y)

    # Step 1: first stage
    Xd = np.column_stack((np.ones(n), x))
    gamma_hat, *_ = np.linalg.lstsq(Xd, q, rcond=None)
    eta_hat = q - Xd @ gamma_hat

    n_tr = int(D.sum())
    if n_tr < 10 or (n - n_tr) < 10:
        return None

    # Step 2: B-spline basis
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    support = (np.percentile(eta_hat, 1), np.percentile(eta_hat, 99))
    Phi, t_knots, degree = _bspline_basis(eta_hat, kn, support)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((x[:, None], Phi, DPhi))

    # Step 3: Ridge-stabilized OLS (tiny ridge to prevent collinearity blowup
    # between Phi and D*Phi blocks sharing the same basis)
    RIDGE_LAM = 1e-5
    P = np.zeros((H.shape[1], H.shape[1]))
    np.fill_diagonal(P[1:, 1:], RIDGE_LAM)  # don't penalize X
    try:
        be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    except np.linalg.LinAlgError:
        return None

    omega_treat = be[1 + n_basis:]      # last n_basis coefficients

    # alpha_hat for all observations
    alpha_hat_all = Phi @ omega_treat

    # Step 4: utility maximization
    gX_sorted = np.sort(q - eta_hat)
    thresh = PHI_GRID[:, None] - eta_hat[None, :]       # (G, n)
    frac_below = np.searchsorted(
        gX_sorted, thresh.ravel()
    ).reshape(len(PHI_GRID), n) / n
    probs = 1.0 - frac_below                             # P(Q > phi | eta_i)
    util = ((alpha_hat_all[None, :] - C) * probs).mean(axis=1)

    # Golden-section refinement around grid max
    idx_max = np.argmax(util)
    lo_idx = max(0, idx_max - 5)
    hi_idx = min(len(PHI_GRID) - 1, idx_max + 5)
    phi_lo = PHI_GRID[lo_idx]
    phi_hi = PHI_GRID[hi_idx]

    gr = (np.sqrt(5) + 1) / 2
    for _ in range(40):
        d = (phi_hi - phi_lo) / gr
        p1 = phi_hi - d
        p2 = phi_lo + d
        u1 = _eval_utility_at(p1, alpha_hat_all, eta_hat, gX_sorted, n)
        u2 = _eval_utility_at(p2, alpha_hat_all, eta_hat, gX_sorted, n)
        if u1 > u2:
            phi_hi = p2
        else:
            phi_lo = p1
    return (phi_lo + phi_hi) / 2


def _eval_utility_at(phi, alpha_hat, eta_hat, gX_sorted, n):
    """Evaluate U_hat(phi) at a single phi."""
    thresh = phi - eta_hat
    frac_below = np.searchsorted(gX_sorted, thresh) / n
    probs = 1.0 - frac_below
    return np.mean((alpha_hat - C) * probs)


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_mc(N_values, R, seed=2025):
    """Run MC experiment, return dict of phi_hat arrays keyed by N."""
    rng_master = np.random.default_rng(seed)
    results = {}
    for N in N_values:
        phi_hats = []
        n_fail = 0
        for r in range(R):
            rng = np.random.default_rng(rng_master.integers(0, 2**63))
            x, y, q, D, *_ = pGen_simple(N, rng)
            phi_hat = estimate_phi_hat(x, y, q, D)
            if phi_hat is None:
                n_fail += 1
                phi_hats.append(np.nan)
            else:
                phi_hats.append(phi_hat)
            if (r + 1) % 100 == 0:
                print(f"  N={N:>6d}: rep {r+1}/{R}", flush=True)
        results[N] = np.array(phi_hats)
        ok = np.isfinite(results[N])
        if ok.sum() > 0:
            bias = np.nanmean(results[N] - PHI_STAR)
            std = np.nanstd(results[N] - PHI_STAR)
            print(f"  N={N:>6d}: {ok.sum()}/{R} ok, "
                  f"bias={bias:+.4f}, std={std:.4f}, "
                  f"n*var={N*std**2:.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(results, asy_var, components, N_values):
    """n*Var(phi_hat) vs theoretical prediction across N."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(r"Asymptotic Variance Test: $n\cdot\mathrm{Var}(\hat\phi)$ "
                 r"vs Theory", fontsize=13)

    empirical_nvar = []
    for N in N_values:
        errs = results[N] - PHI_STAR
        errs = errs[np.isfinite(errs)]
        empirical_nvar.append(N * np.var(errs))

    # Left: n*Var vs N
    ax = axes[0]
    ax.plot(N_values, empirical_nvar, "bo-", linewidth=2, markersize=8,
            label=r"Empirical $n\cdot\mathrm{Var}(\hat\phi)$")
    ax.axhline(asy_var, color="red", linewidth=2, linestyle="--",
               label=f"Theory = {asy_var:.3f}")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$n \cdot \mathrm{Var}(\hat\phi)$")
    ax.set_title("Convergence of scaled variance")
    ax.legend()
    ax.set_xscale("log")

    # Right: ratio
    ax2 = axes[1]
    ratios = [ev / asy_var for ev in empirical_nvar]
    ax2.plot(N_values, ratios, "go-", linewidth=2, markersize=8)
    ax2.axhline(1.0, color="red", linewidth=2, linestyle="--",
                label="Theory = 1.0")
    ax2.set_xlabel("N")
    ax2.set_ylabel(r"Empirical / Theory")
    ax2.set_title("Variance ratio (should converge to 1)")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_ylim(0, max(2.0, max(ratios) * 1.1))

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "asy_var_convergence.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_qq(results, asy_var, N_values):
    """QQ plots of sqrt(n)*(phi_hat - phi*)/sigma_theory at each N."""
    n_plots = len(N_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(r"QQ plots: $\sqrt{n}(\hat\phi - \phi^*)/\sigma$ vs N(0,1)",
                 fontsize=13)

    sigma_theory = np.sqrt(asy_var)
    z_theoretical = np.linspace(-3, 3, 200)

    for ax, N in zip(axes, N_values):
        errs = results[N] - PHI_STAR
        errs = errs[np.isfinite(errs)]
        scaled = np.sqrt(N) * errs / sigma_theory
        scaled_sorted = np.sort(scaled)
        n_ok = len(scaled_sorted)
        theoretical_quantiles = sp_norm.ppf(
            (np.arange(1, n_ok + 1) - 0.5) / n_ok
        )

        ax.plot(theoretical_quantiles, scaled_sorted, "b.", markersize=3)
        ax.plot([-3, 3], [-3, 3], "r-", linewidth=1.5)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(f"N = {N:,}")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "asy_var_qq.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_components(components):
    """Bar chart of variance components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r"$\sigma_0^2$", r"$\sigma_\alpha^2$", r"$\sigma_g^2$"]
    vals = [components["sigma_0_sq"], components["sigma_alpha_sq"],
            components["sigma_g_sq"]]
    colors = ["steelblue", "coral", "seagreen"]

    bars = ax.bar(names, vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.5f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Variance contribution")
    ax.set_title(r"Components of $\sigma_{\mathrm{dec}}^2(\phi^*)$"
                 f"\nTotal = {sum(vals):.5f},  "
                 f"U''(phi*) = {components['U_pp']:.5f},  "
                 f"Asy Var = {components['asy_var']:.4f}")
    fig.tight_layout()
    fname = os.path.join(OUTDIR, "asy_var_components.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def print_table(results, asy_var, components, N_values):
    """Print and save summary table."""
    lines = [
        "Asymptotic Variance Test",
        f"DGP: X~N(0,1), eta~N(0,1), W|eta~N(alpha(eta),{SIGMA_W}^2), "
        f"nu~N(0,{SIGMA_NU}^2)",
        f"alpha(eta) = 2*sigmoid(eta), gamma={GAMMA}, beta={BETA}, c={C}",
        f"phi_star = {PHI_STAR} (by symmetry)",
        "",
        "Theoretical variance components:",
        f"  sigma_0^2     = {components['sigma_0_sq']:.6f}  (utility fluctuation)",
        f"  sigma_alpha^2 = {components['sigma_alpha_sq']:.6f}  (alpha estimation)",
        f"  sigma_g^2     = {components['sigma_g_sq']:.6f}  (density estimation)",
        f"  sigma_EIV^2   = 0  (X indep of eta)",
        f"  sigma_dec^2   = {components['sigma_dec_sq']:.6f}",
        f"  U''(phi*)     = {components['U_pp']:.6f}",
        f"  Asy Var       = {asy_var:.6f}",
        "",
        f"{'N':>8s} {'R_ok':>5s} {'Bias':>9s} {'Std':>9s} "
        f"{'n*Var':>9s} {'Theory':>9s} {'Ratio':>8s}",
        "-" * 62,
    ]
    for N in N_values:
        errs = results[N] - PHI_STAR
        errs = errs[np.isfinite(errs)]
        n_ok = len(errs)
        bias = np.mean(errs)
        std = np.std(errs)
        nvar = N * std**2
        ratio = nvar / asy_var
        lines.append(
            f"{N:>8d} {n_ok:>5d} {bias:>+9.5f} {std:>9.5f} "
            f"{nvar:>9.4f} {asy_var:>9.4f} {ratio:>8.3f}"
        )

    text = "\n".join(lines)
    print(text)
    fname = os.path.join(OUTDIR, "asy_var_table.txt")
    with open(fname, "w") as f:
        f.write(text + "\n")
    print(f"\nTable saved to {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Step 1: Verify phi_star ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Verify phi_star = 0")
    print("=" * 60)
    U_prime_at_0 = compute_U_prime(PHI_STAR)
    print(f"  U'(0)  = {U_prime_at_0:.2e}  (should be ~0)")

    # Also check a small neighborhood to confirm it's a maximum
    for phi in [-0.1, -0.01, 0.0, 0.01, 0.1]:
        print(f"  U'({phi:+.2f}) = {compute_U_prime(phi):+.6f}")

    # ── Step 2: Compute U''(phi*) ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Compute U''(phi*)")
    print("=" * 60)
    U_pp = compute_U_double_prime(PHI_STAR)
    print(f"  U''(0) = {U_pp:.6f}")

    # ── Step 3: Compute sigma_0^2 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Compute sigma_0^2 (utility fluctuation)")
    print("=" * 60)
    s0 = compute_sigma_0_sq()
    print(f"  sigma_0^2 = {s0:.6f}")

    # ── Step 4: Compute sigma_g^2 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Compute sigma_g^2 (density estimation)")
    print("=" * 60)
    sg = compute_sigma_g_sq()
    print(f"  sigma_g^2 = {sg:.6f}")

    # ── Step 5: Compute sigma_alpha^2 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Compute sigma_alpha^2 (alpha estimation via Riesz)")
    print(f"  Using N_MC = {N_MC:,} for Riesz representer")
    print("=" * 60)
    sa = compute_sigma_alpha_sq(seed=0)
    print(f"  sigma_alpha^2 = {sa:.6f}")

    # ── Step 6: Assemble ────────────────────────────────────────────────
    sigma_dec_sq = s0 + sa + sg
    asy_var = sigma_dec_sq / U_pp**2

    components = {
        "sigma_0_sq": s0,
        "sigma_alpha_sq": sa,
        "sigma_g_sq": sg,
        "sigma_dec_sq": sigma_dec_sq,
        "U_pp": U_pp,
        "asy_var": asy_var,
    }

    print("\n" + "=" * 60)
    print("Theoretical asymptotic variance")
    print("=" * 60)
    print(f"  sigma_dec^2   = {sigma_dec_sq:.6f}")
    print(f"  U''(phi*)^2   = {U_pp**2:.6f}")
    print(f"  Asy Var       = {asy_var:.6f}")
    print(f"  (i.e. sqrt(n)*(phi_hat - phi*) ~ N(0, {asy_var:.4f}))")

    # ── Step 7: Monte Carlo simulation ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 7: Monte Carlo (R={R} reps per N)")
    print("=" * 60)
    results = run_mc(N_VALUES, R)

    # ── Step 8: Plots and table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8: Output")
    print("=" * 60)
    plot_convergence(results, asy_var, components, N_VALUES)
    plot_qq(results, asy_var, N_VALUES)
    plot_components(components)
    print_table(results, asy_var, components, N_VALUES)


if __name__ == "__main__":
    main()
