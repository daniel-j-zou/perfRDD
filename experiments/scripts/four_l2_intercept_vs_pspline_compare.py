"""Four-dataset L2-with-intercept vs P-spline-with-intercept comparison.

Both methods now prepend a 1s column to X inside the PLM design matrix
(matching what the Q first stage already does), so β has a free intercept
that can absorb the population-mean of Y. Without this fix, the L2 penalty
on ω_base forces the constant offset to leak into α̂; see
`four_l2_vs_pspline_compare.py` (no-intercept baseline) for that pathology.

Layout (per dataset, 4 rows × 7 cols):
    stats sidebar | L2 α(η) | L2 b̂ + α̂+b̂ | L2 U(φ)
                  | P-spline α(η) | P-spline b̂ + α̂+b̂ | P-spline U(φ)

Both fits use the same cost grid (computed from the L2 fit's support-restricted
average α) so utility curves are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    FitResult,
    _auto_c_values,
    _basis_params,
    _detect_direction,
    _eval_basis,
    _reduce_to_primary_axis,
    _subsample,
    _utility_curve,
)


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "four_l2_intercept_vs_pspline_compare.png"

ALPHA_FILL = 0.30
BINS = 60
RIDGE_TREAT = 0.1
RIDGE_BASE_L2 = 0.1
EPS_RIDGE_BASE = 1e-6
LAM_SCALE = 1.0   # P-spline λ = LAM_SCALE * n_used


DATASETS = [
    (
        "gpa", "GPA — academic probation",
        "Q = first-year GPA − cutoff",
        "Y = next-term GPA",
        "X = HS%, credits, campus, sex, ...",
        "treated when Q < 0",
    ),
    (
        "nhanes", "NHANES — HbA1c diabetic cutoff",
        "Q = HbA1c (%)",
        "Y = systolic BP (mmHg)",
        "X = age, sex, race, BMI, poverty",
        "treated when Q ≥ 6.5",
    ),
    (
        "oulad", "OULAD — first-TMA pass mark",
        "Q = first TMA score",
        "Y = mean subsequent TMA",
        "X = prev attempts, credits, demo...",
        "treated when Q ≥ 40",
    ),
    (
        "lending_club", "Lending Club — DTI trigger",
        "Q = DTI (%)",
        "Y = interest rate (%)",
        "X = loan, income, accts, term, ...",
        "treated when Q ≥ 30",
    ),
]


def second_diff_matrix(m: int) -> np.ndarray:
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _common_setup(Q, X, Y, D, direction):
    n = len(Y)
    n_tr = int(D.sum())
    X_design = np.column_stack((np.ones(n), X))
    gamma, *_ = np.linalg.lstsq(X_design, Q, rcond=None)
    eta = Q - X_design @ gamma

    support = (float(np.percentile(eta, 0.5)),
               float(np.percentile(eta, 99.5)))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta, info)
    n_basis = Phi.shape[1]

    # Intercept column prepended to X for the PLM design.
    X_aug = np.column_stack((np.ones(n), X))
    DPhi = D[:, None] * Phi
    H = np.column_stack((X_aug, Phi, DPhi))
    p = X_aug.shape[1]

    eta_tr = eta[D == 1]
    if len(eta_tr) >= 20:
        eval_lo = max(support[0], float(np.percentile(eta_tr, 5)))
        eval_hi = min(support[1], float(np.percentile(eta_tr, 95)))
    else:
        eval_lo, eval_hi = support

    return {
        "n": n, "n_tr": n_tr, "gamma": gamma, "eta": eta,
        "info": info, "Phi": Phi, "n_basis": n_basis,
        "H": H, "p": p,
        "eta_eval": (eval_lo, eval_hi),
    }


def fit_l2_intercept(Q, X, Y, D, direction, ridge_base=RIDGE_BASE_L2,
                     ridge_treat=RIDGE_TREAT):
    s = _common_setup(Q, X, Y, D, direction)
    H, p, n_basis, n = s["H"], s["p"], s["n_basis"], s["n"]
    total = H.shape[1]

    # L2 ridge on both ω_base and ω_treat blocks; X (with intercept) unpenalised.
    lam_base = ridge_base / np.sqrt(n)
    lam_treat = ridge_treat / np.sqrt(n)
    P = np.zeros((total, total))
    np.fill_diagonal(P[p:p + n_basis, p:p + n_basis], lam_base)
    np.fill_diagonal(P[p + n_basis:, p + n_basis:], lam_treat)

    coefs = np.linalg.solve(H.T @ H + n * P, H.T @ Y)
    beta_aug = coefs[:p]
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    return FitResult(
        gamma=s["gamma"], eta=s["eta"],
        omega_base=omega_base, omega_treat=omega_treat,
        beta=beta_aug, info=s["info"],
        eta_eval=s["eta_eval"], direction=direction,
        n=n, n_treated=s["n_tr"],
    )


def fit_pspline_intercept(Q, X, Y, D, direction, lam_smooth,
                          ridge_treat=RIDGE_TREAT):
    s = _common_setup(Q, X, Y, D, direction)
    H, p, n_basis, n = s["H"], s["p"], s["n_basis"], s["n"]
    total = H.shape[1]

    D2 = second_diff_matrix(n_basis)
    DtD = D2.T @ D2

    P = np.zeros((total, total))
    P[p:p + n_basis, p:p + n_basis] = (
        lam_smooth * DtD
        + EPS_RIDGE_BASE * np.sqrt(n) * np.eye(n_basis)
    )
    P[p + n_basis:, p + n_basis:] = (
        ridge_treat * np.sqrt(n) * np.eye(n_basis)
    )

    coefs = np.linalg.solve(H.T @ H + P, H.T @ Y)
    beta_aug = coefs[:p]
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    return FitResult(
        gamma=s["gamma"], eta=s["eta"],
        omega_base=omega_base, omega_treat=omega_treat,
        beta=beta_aug, info=s["info"],
        eta_eval=s["eta_eval"], direction=direction,
        n=n, n_treated=s["n_tr"],
    )


def overlay_density(ax, fit, D_arr, eta_range):
    eta_lo, eta_hi = eta_range
    bins = np.linspace(eta_lo, eta_hi, BINS)
    ax_d = ax.twinx()
    eta_co = fit.eta[D_arr == 0]
    eta_tr = fit.eta[D_arr == 1]
    in_co = (eta_co >= eta_lo) & (eta_co <= eta_hi)
    in_tr = (eta_tr >= eta_lo) & (eta_tr <= eta_hi)
    ax_d.hist(eta_co[in_co], bins=bins, density=True,
              alpha=ALPHA_FILL, color="steelblue", zorder=1)
    ax_d.hist(eta_tr[in_tr], bins=bins, density=True,
              alpha=ALPHA_FILL, color="firebrick", zorder=1)
    ax_d.set_yticks([])
    ax.set_zorder(ax_d.get_zorder() + 1)
    ax.patch.set_visible(False)


def plot_alpha(ax, fit, D_arr, title):
    eta_lo, eta_hi = fit.eta_eval
    grid = np.linspace(eta_lo, eta_hi, 500)
    alpha = _eval_basis(grid, fit.info) @ fit.omega_treat
    avg_alpha = float(alpha.mean())

    overlay_density(ax, fit, D_arr, (eta_lo, eta_hi))

    ax.plot(grid, alpha, "C0-", lw=1.8, zorder=4)
    ax.axhline(0, color="black", lw=0.4, zorder=3)
    ax.axhline(avg_alpha, color="red", ls="--", lw=0.8, zorder=4)
    ax.set_xlabel(r"$\eta$", fontsize=8)
    ax.set_ylabel(r"$\hat\alpha(\eta)$", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.text(0.03, 0.95, f"avg = {avg_alpha:.3g}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.65, pad=1.5))
    ax.tick_params(axis="both", labelsize=7)
    return avg_alpha


def plot_decomp(ax, fit, D_arr, title):
    eta_lo, eta_hi = fit.eta_eval
    info = fit.info
    grid_eval = np.linspace(eta_lo, eta_hi, 500)
    aplusb = _eval_basis(grid_eval, info) @ (fit.omega_base + fit.omega_treat)
    grid_full = np.linspace(info["lo"], info["hi"], 600)
    b_full = _eval_basis(grid_full, info) @ fit.omega_base

    overlay_density(ax, fit, D_arr, (info["lo"], info["hi"]))

    ax.plot(grid_full, b_full, "C0-", lw=1.5,
            label=r"$\hat b(\eta)$", zorder=4)
    ax.plot(grid_eval, aplusb, "C3-", lw=1.5,
            label=r"$\hat\alpha+\hat b$", zorder=4)
    ax.axvline(eta_lo, color="green", ls=":", alpha=0.6, zorder=3)
    ax.axvline(eta_hi, color="green", ls=":", alpha=0.6, zorder=3)
    ax.set_xlabel(r"$\eta$", fontsize=8)
    ax.set_ylabel("h", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", fontsize=7, framealpha=0.7)
    ax.tick_params(axis="both", labelsize=7)


def plot_utility(ax, fit, Q, X_arr, threshold, c_values, title):
    s = float(Q.std())
    phi_grid = np.linspace(threshold - 3 * s, threshold + 3 * s, 400)
    utils = _utility_curve(fit, Q, X_arr, phi_grid, c_values)

    cmap = plt.get_cmap("viridis")
    cols = cmap(np.linspace(0.1, 0.85, len(utils)))
    phi_stars = {}
    for (c, u), col in zip(utils.items(), cols):
        idx = int(np.argmax(u))
        phi_star = float(phi_grid[idx])
        phi_stars[c] = phi_star
        ax.plot(phi_grid, u, color=col, lw=1.2,
                label=f"c={c:.2g}, φ*={phi_star:.3g}")
        ax.axvline(phi_star, color=col, ls="--", lw=0.5, alpha=0.4)
    ax.axvline(threshold, color="black", ls=":", lw=0.9,
               label=f"current φ={threshold:.3g}")
    ax.axhline(0, color="black", lw=0.3)
    ax.set_xlabel(r"$\phi$", fontsize=8)
    ax.set_ylabel(r"$\hat U(\phi)$", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", fontsize=6, framealpha=0.65)
    ax.tick_params(axis="both", labelsize=7)
    return phi_stars


def main() -> None:
    sum_lin = json.loads((RUNS / "perfrdd" / "summary.json").read_text())
    sum_qnl = json.loads((RUNS / "perfrdd_q_nonlinear" / "summary.json").read_text())

    nrows = len(DATASETS)
    fig = plt.figure(figsize=(24, 4.0 * nrows))
    gs = fig.add_gridspec(
        nrows=nrows, ncols=7,
        width_ratios=[1.5, 1.0, 1.2, 1.2, 1.0, 1.2, 1.2],
        hspace=0.55, wspace=0.32,
        left=0.025, right=0.99, top=0.93, bottom=0.04,
    )

    fig.suptitle(
        "Four-dataset comparison with INTERCEPT in PLM design: "
        "L2 ridge vs P-spline penalty on " r"$\omega_{\mathrm{base}}$",
        fontsize=15, fontweight="bold", y=0.975,
    )
    fig.text(0.31, 0.952,
             r"L2 ridge  +  intercept in $X$   ($\rho_b=\rho_t=0.1/\sqrt{n}$)",
             ha="center", fontsize=12, fontweight="bold", color="#222")
    fig.text(0.72, 0.952,
             r"P-spline on $\omega_{\mathrm{base}}$  +  intercept in $X$   "
             r"($\lambda = n$)",
             ha="center", fontsize=12, fontweight="bold", color="#222")

    for i, (name, title, q_def, y_def, x_def, treat_def) in enumerate(DATASETS):
        sample = load(name)
        Q, X, threshold = _reduce_to_primary_axis(sample)
        Y, D = sample.Y, sample.D
        direction = _detect_direction(D, Q)
        keep = (np.isfinite(Q) & np.isfinite(Y)
                & np.isfinite(X).all(axis=1))
        Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
        if len(Y) > DEFAULT_MAX_N:
            (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
        n_used = len(Y)
        n_tr = int(D.sum())

        fit_l = fit_l2_intercept(Q, X, Y, D, direction)
        fit_p = fit_pspline_intercept(Q, X, Y, D, direction,
                                      lam_smooth=LAM_SCALE * n_used)

        # Shared cost grid from the L2 (intercept) fit.
        Phi_eta_l = _eval_basis(fit_l.eta, fit_l.info)
        alpha_all_l = Phi_eta_l @ fit_l.omega_treat
        in_supp_l = ((fit_l.eta >= fit_l.eta_eval[0])
                     & (fit_l.eta <= fit_l.eta_eval[1]))
        avg_alpha_supp_l = float(
            np.mean(np.where(in_supp_l, alpha_all_l, 0.0))
        )
        c_values = _auto_c_values(avg_alpha_supp_l)

        ax_txt = fig.add_subplot(gs[i, 0])
        ax_txt.axis("off")
        ax_txt.text(0.02, 0.98, title, transform=ax_txt.transAxes,
                    fontsize=11, fontweight="bold", va="top")
        ax_txt.text(0.02, 0.86,
                    f"{q_def}\n{y_def}\n{x_def}\n{treat_def}",
                    transform=ax_txt.transAxes,
                    fontsize=8.2, family="monospace", va="top")
        ax_txt.text(0.02, 0.52,
                    f"n = {n_used:,}\n"
                    f"treated = {n_tr:,}\n"
                    f"direction = {direction}\n"
                    f"current φ = {threshold:.3g}\n"
                    f"R²(linear)      = {sum_lin[name]['first_stage_R2']:.3f}\n"
                    f"R²(q_nonlinear) = {sum_qnl[name]['first_stage_R2']:.3f}\n"
                    f"P-spline λ = {LAM_SCALE * n_used:,.0f}\n"
                    f"intercept β₀ (L2)       = {fit_l.beta[0]:.3g}\n"
                    f"intercept β₀ (P-spline) = {fit_p.beta[0]:.3g}",
                    transform=ax_txt.transAxes,
                    fontsize=8.2, family="monospace", va="top")

        avg_l = plot_alpha(
            fig.add_subplot(gs[i, 1]), fit_l, D,
            f"{name}: α̂(η)  [L2 + intercept]",
        )
        plot_decomp(
            fig.add_subplot(gs[i, 2]), fit_l, D,
            f"{name}: b̂ and α̂+b̂  [L2 + intercept]",
        )
        phi_l = plot_utility(
            fig.add_subplot(gs[i, 3]), fit_l, Q, X, threshold, c_values,
            f"{name}: U(φ)  [L2 + intercept]",
        )

        avg_p = plot_alpha(
            fig.add_subplot(gs[i, 4]), fit_p, D,
            f"{name}: α̂(η)  [P-spline + intercept]",
        )
        plot_decomp(
            fig.add_subplot(gs[i, 5]), fit_p, D,
            f"{name}: b̂ and α̂+b̂  [P-spline + intercept]",
        )
        phi_p = plot_utility(
            fig.add_subplot(gs[i, 6]), fit_p, Q, X, threshold, c_values,
            f"{name}: U(φ)  [P-spline + intercept]",
        )

        ax_txt.text(0.02, 0.14,
                    f"avg α (L2+int)       = {avg_l:.3g}\n"
                    f"avg α (P-spline+int) = {avg_p:.3g}\n"
                    f"φ* shifts (c=0): {phi_l[c_values[0]]:.3g} → "
                    f"{phi_p[c_values[0]]:.3g}",
                    transform=ax_txt.transAxes,
                    fontsize=8.2, family="monospace", va="top")

        print(f"[{name}] n={n_used:,} n_treated={n_tr:,}  "
              f"β₀(L2)={fit_l.beta[0]:.3g}  β₀(P)={fit_p.beta[0]:.3g}  "
              f"avg α (L2)={avg_l:.3g}  avg α (P)={avg_p:.3g}",
              flush=True)

    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
