"""Same composite as four_alpha_with_eta.py but relabels the right panel curves
as b̂(η) and α̂(η) + b̂(η) — i.e., the additive decomposition of the treated
component into its baseline and treatment-effect pieces.

Curves are mathematically identical to four_alpha_with_eta.py; only the legend
and panel title change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    _auto_c_values,
    _detect_direction,
    _eval_basis,
    _fit_pooled_plm,
    _plot_utility,
    _reduce_to_primary_axis,
    _subsample,
    _utility_curve,
    DEFAULT_MAX_N,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_DIR = RUNS / "four_datasets_linear_alpha_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_FILL = 0.32
BINS = 70

DATASETS = [
    (
        "gpa",
        "GPA — academic probation",
        "Q = first-year GPA − probation cutoff",
        "Y = next-term GPA",
        "X = HS-grade %, year-1 credits, 2 campus dummies,\n"
        "    male, N.American-born, age at entry, English",
        "treated when Q < 0  (placed on probation)",
    ),
    (
        "nhanes",
        "NHANES — HbA1c diabetic cutoff",
        "Q = HbA1c (%)",
        "Y = systolic blood pressure (mmHg)",
        "X = age, sex, race/ethnicity, BMI, poverty index",
        "treated when Q ≥ 6.5  (ADA diabetic threshold)",
    ),
    (
        "oulad",
        "OULAD — first-TMA pass mark",
        "Q = first TMA score",
        "Y = mean score on subsequent TMAs",
        "X = prev attempts, studied credits, gender,\n"
        "    education, IMD band, age band, disability",
        "treated when Q ≥ 40  (UK pass mark)",
    ),
    (
        "lending_club",
        "Lending Club — DTI trigger",
        "Q = debt-to-income ratio (DTI %)",
        "Y = originated interest rate (%)",
        "X = loan amount, income, delinquencies, open accts,\n"
        "    pub records, total accts, inquiries, term,\n"
        "    home ownership, purpose, verification",
        "treated when Q ≥ 30  (LC underwriting trigger)",
    ),
]


def fit_linear(name: str):
    sample = load(name)
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
    fit = _fit_pooled_plm(Q, X, Y, D, direction)
    return fit, D, Q, X, threshold


def plot_alpha_b_with_eta(fit, D, name: str, out_path: Path) -> float:
    eta_lo, eta_hi = fit.eta_eval
    eta_tr = fit.eta[D == 1]
    eta_co = fit.eta[D == 0]

    grid_eval = np.linspace(eta_lo, eta_hi, 500)
    Phi_eval = _eval_basis(grid_eval, fit.info)
    alpha_curve = Phi_eval @ fit.omega_treat
    alpha_plus_b = Phi_eval @ (fit.omega_base + fit.omega_treat)

    grid_full = np.linspace(fit.info["lo"], fit.info["hi"], 500)
    b_full = _eval_basis(grid_full, fit.info) @ fit.omega_base

    avg_alpha = float(alpha_curve.mean())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- left: alpha(eta) + density underlay ---
    bins_eval = np.linspace(eta_lo, eta_hi, BINS)
    ax0d = ax[0].twinx()
    in_co = (eta_co >= eta_lo) & (eta_co <= eta_hi)
    in_tr = (eta_tr >= eta_lo) & (eta_tr <= eta_hi)
    ax0d.hist(eta_co[in_co], bins=bins_eval, density=True,
              alpha=ALPHA_FILL, color="steelblue", label="control η", zorder=1)
    ax0d.hist(eta_tr[in_tr], bins=bins_eval, density=True,
              alpha=ALPHA_FILL, color="firebrick", label="treated η", zorder=1)
    ax0d.set_ylabel("η density", color="grey", fontsize=9)
    ax0d.tick_params(axis="y", labelsize=8, colors="grey")

    ax[0].plot(grid_eval, alpha_curve, "C0-", lw=2.2,
               label=r"$\hat\alpha(\eta)$", zorder=4)
    ax[0].axhline(0, color="black", lw=0.5, zorder=3)
    ax[0].axhline(avg_alpha, color="red", ls="--", lw=1,
                  label=f"avg = {avg_alpha:.3g}", zorder=4)
    ax[0].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$\alpha(\eta)$")
    ax[0].set_title(f"{name}: treatment effect α(η)")
    ax[0].set_zorder(ax0d.get_zorder() + 1)
    ax[0].patch.set_visible(False)
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax0d.get_legend_handles_labels()
    ax[0].legend(h0 + h1, l0 + l1, loc="best", fontsize=8)

    # --- right: b̂ and α̂ + b̂ + density underlay ---
    bins_full = np.linspace(fit.info["lo"], fit.info["hi"], BINS)
    ax1d = ax[1].twinx()
    ax1d.hist(eta_co, bins=bins_full, density=True,
              alpha=ALPHA_FILL, color="steelblue", zorder=1)
    ax1d.hist(eta_tr, bins=bins_full, density=True,
              alpha=ALPHA_FILL, color="firebrick", zorder=1)
    ax1d.set_ylabel("η density", color="grey", fontsize=9)
    ax1d.tick_params(axis="y", labelsize=8, colors="grey")

    ax[1].plot(grid_full, b_full, "C0-", lw=1.8,
               label=r"$\hat b(\eta)$", zorder=4)
    ax[1].plot(grid_eval, alpha_plus_b, "C3-", lw=1.8,
               label=r"$\hat\alpha(\eta) + \hat b(\eta)$", zorder=4)
    ax[1].axvline(eta_lo, color="green", ls=":", alpha=0.6, zorder=3)
    ax[1].axvline(eta_hi, color="green", ls=":", alpha=0.6,
                  label="eval region", zorder=3)
    ax[1].set_xlabel(r"$\eta$")
    ax[1].set_ylabel("h")
    ax[1].set_title(r"$\hat b(\eta)$  and  $\hat\alpha(\eta) + \hat b(\eta)$")
    ax[1].set_zorder(ax1d.get_zorder() + 1)
    ax[1].patch.set_visible(False)
    ax[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return avg_alpha


def plot_utility_for_fit(fit, D, Q, X, threshold: float, name: str,
                         out_path: Path) -> dict:
    Phi_eta = _eval_basis(fit.eta, fit.info)
    alpha_all = Phi_eta @ fit.omega_treat
    in_supp = (fit.eta >= fit.eta_eval[0]) & (fit.eta <= fit.eta_eval[1])
    avg_alpha_supp = float(np.mean(np.where(in_supp, alpha_all, 0.0)))
    c_values = _auto_c_values(avg_alpha_supp)

    s = float(Q.std())
    phi_grid = np.linspace(threshold - 3 * s, threshold + 3 * s, 400)
    utils = _utility_curve(fit, Q, X, phi_grid, c_values)
    return _plot_utility(phi_grid, utils, threshold, out_path, name)


def main() -> None:
    sum_lin = json.loads((RUNS / "perfrdd" / "summary.json").read_text())
    sum_qnl = json.loads((RUNS / "perfrdd_q_nonlinear" / "summary.json").read_text())

    fits = {}
    for name, *_ in DATASETS:
        print(f"[{name}] fitting linear PLM…", flush=True)
        fit, D, Q, X, threshold = fit_linear(name)
        a_path = OUT_DIR / f"{name}_alpha.png"
        u_path = OUT_DIR / f"{name}_utility.png"
        avg = plot_alpha_b_with_eta(fit, D, name, a_path)
        plot_utility_for_fit(fit, D, Q, X, threshold, name, u_path)
        fits[name] = {"avg_alpha": avg}
        print(f"[{name}] α img -> {a_path.name}  U img -> {u_path.name}  "
              f"avg α = {avg:.3g}", flush=True)

    n = len(DATASETS)
    fig = plt.figure(figsize=(22, 4.0 * n))
    gs = fig.add_gridspec(
        nrows=n, ncols=3,
        width_ratios=[1.5, 4.5, 3.0],
        hspace=0.18, wspace=0.04,
        left=0.01, right=0.99, top=0.96, bottom=0.02,
    )
    fig.suptitle(
        r"Performative RDD — Linear setting, 4 datasets: $\hat\alpha(\eta)$, "
        r"$\hat b(\eta)$ vs $\hat\alpha(\eta) + \hat b(\eta)$, and $U(\phi)$",
        fontsize=14, fontweight="bold", y=0.985,
    )

    for i, (name, title, q_def, y_def, x_def, treat_def) in enumerate(DATASETS):
        s_lin = sum_lin[name]
        s_qnl = sum_qnl[name]

        ax_txt = fig.add_subplot(gs[i, 0])
        ax_txt.axis("off")
        ax_txt.text(0.02, 0.98, title, transform=ax_txt.transAxes,
                    fontsize=12, fontweight="bold", va="top")

        defs = f"{q_def}\n{y_def}\n{x_def}\n{treat_def}"
        ax_txt.text(0.02, 0.86, defs, transform=ax_txt.transAxes,
                    fontsize=8.8, family="monospace", va="top")

        body = (
            f"n = {s_lin['n_used']:,}    treated = {s_lin['n_treated']:,}\n"
            f"direction = {s_lin['direction']}    "
            f"current φ = {s_lin['threshold_actual']:.3g}\n"
            f"R²(linear)      = {s_lin['first_stage_R2']:.3f}\n"
            f"R²(q_nonlinear) = {s_qnl['first_stage_R2']:.3f}\n"
            f"avg α(η)        = {s_lin['avg_alpha']:.3g}"
        )
        ax_txt.text(0.02, 0.40, body, transform=ax_txt.transAxes,
                    fontsize=9.5, family="monospace", va="top")

        ax_a = fig.add_subplot(gs[i, 1])
        ax_a.imshow(mpimg.imread(OUT_DIR / f"{name}_alpha.png"))
        ax_a.axis("off")

        ax_u = fig.add_subplot(gs[i, 2])
        ax_u.imshow(mpimg.imread(OUT_DIR / f"{name}_utility.png"))
        ax_u.axis("off")

    comp = RUNS / "four_datasets_linear_alpha_b_compare.png"
    fig.savefig(comp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {comp}")


if __name__ == "__main__":
    main()
