"""Screen a candidate dataset for the PerfRDD performative-threshold method.

Given a registered dataset (or an ``RDDSample``), this runs the pooled
partially-linear estimator and emits exactly three review artifacts plus a
machine-readable verdict:

* ``alpha.png``   — the estimated treatment-effect curve ``alpha_hat(eta)``;
* ``b.png``       — the estimated baseline curve ``b_hat(eta)``;
* ``utility.png`` — the estimated welfare ``U(phi)`` with its maximizer marked;
* ``description.md`` / ``summary.json`` — a short dataset description and the
  screening verdict.

The screening question is deliberately narrow. A dataset is *interesting* for
this project when the fitted treatment effect ``alpha_hat(eta)`` is
**non-constant and changes sign** across the latent-ability support, because a
sign-definite ``alpha`` forces a boundary policy at zero cost. The harness
therefore reports, front and centre, whether ``alpha_hat`` crosses zero on the
overlap support and whether the welfare maximizer is interior.

This is a fast first-pass screen built on the pooled PLM (``_fit_pooled_plm``);
a candidate that passes should then be re-run through the inference-grade
hard-trim estimator (``perfrdd_hard_trim``).

Run as a module::

    python -m experiments.scripts.screen_candidate gpa taxi oulad
    python -m experiments.scripts.screen_candidate --eps 0.1 lending_club
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.registry import load
from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    _detect_direction,
    _eval_basis,
    _fit_pooled_plm,
    _reduce_to_primary_axis,
    _subsample,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "screen_candidate"

# The welfare surface is inherently flat near the optimum, so a small sample lets
# noise manufacture a spurious interior optimum (e.g. lending_default flips from
# "interior" at 40k to "boundary" at >=200k). Screen on enough data to tell.
SCREEN_MAX_N = 250_000


@dataclass
class ScreenResult:
    name: str
    n: int
    first_stage_R2: float
    eta_support: Tuple[float, float]
    overlap_window: Tuple[float, float]
    hard_retention: float
    alpha_min: float
    alpha_max: float
    neg_alpha_mass: float
    alpha_crosses_zero: bool
    phi_star_zero_cost: float
    phi_star_interior_zero_cost: bool
    boundary_gain: float
    boundary_gain_rel: float
    confirmed_full_n: bool | None
    boundary_gain_full: float
    n_confirm: int
    interior_cost_range: Tuple[float, float] | None
    cost_shown: float
    phi_star_shown: float
    phi_grid_edges: Tuple[float, float]
    threshold: float
    verdict: str
    out_dir: str


def _prep(sample: RDDSample, max_n: int | None):
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = np.asarray(sample.Y, dtype=float)
    D = np.asarray(sample.D, dtype=float)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    return Q, X, Y, D, float(threshold)


def _plot_alpha(fit, D, grid, alpha, out_path: Path, name: str,
               window: Tuple[float, float]) -> None:
    lo, hi = fit.eta_eval
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvspan(window[0], window[1], color="gold", alpha=0.15,
               label="overlap window [l₀,u₀]", zorder=0)
    dens = ax.twinx()
    eta_tr, eta_co = fit.eta[D == 1], fit.eta[D == 0]
    bins = np.linspace(lo, hi, 60)
    for e, col, lab in ((eta_co, "steelblue", "control η"),
                        (eta_tr, "firebrick", "treated η")):
        m = (e >= lo) & (e <= hi)
        dens.hist(e[m], bins=bins, density=True, alpha=0.28, color=col, label=lab)
    dens.set_ylabel("η density", color="grey", fontsize=9)
    dens.tick_params(axis="y", labelsize=8, colors="grey")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.plot(grid, alpha, color="C3", lw=2.2, zorder=5)
    ax.set_zorder(dens.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title(rf"{name}: treatment effect $\hat\alpha(\eta)$")
    ax.set_xlabel(r"latent ability $\eta$")
    ax.set_ylabel(r"$\hat\alpha(\eta)$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_b(fit, grid, b_curve, out_path: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0.0, color="black", lw=0.5)
    ax.plot(grid, b_curve, color="C0", lw=2.2)
    ax.set_title(rf"{name}: baseline $\hat b(\eta)$")
    ax.set_xlabel(r"latent ability $\eta$")
    ax.set_ylabel(r"$\hat b(\eta)$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_utility(phi_grid, utilities, c_main, threshold, out_path: Path, name: str) -> Tuple[float, bool]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    main = utilities[c_main]
    best = int(np.argmax(main))
    phi_star = float(phi_grid[best])
    interior = best not in (0, len(phi_grid) - 1)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, len(utilities)))
    for (c, u), col in zip(sorted(utilities.items()), colors):
        lw = 2.4 if c == c_main else 1.2
        ax.plot(phi_grid, u, color=col, lw=lw, label=f"c={c:.3g}")
    ax.plot(phi_star, main[best], "*", color="C3", ms=16,
            label=f"argmax φ*={phi_star:.3g} ({'interior' if interior else 'BOUNDARY'})")
    ax.axvline(threshold, color="black", ls=":", lw=1.0, label=f"current cutoff={threshold:.3g}")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title(rf"{name}: estimated welfare $\hat U(\phi)$")
    ax.set_xlabel(r"policy threshold $\phi$")
    ax.set_ylabel(r"$\hat U(\phi)$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return phi_star, interior


def _utility_profiles(fit, Q, phi_grid, window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two cost-free welfare profiles A(phi), B(phi).

    The welfare at cost c is ``U_c(phi) = A(phi) - c * B(phi)`` where
    ``A(phi) = mean_i alpha_b(eta_i) P(treated|eta_i,phi)`` and
    ``B(phi) = mean_i P(treated|eta_i,phi)``. Neither depends on c, so we compute
    them once (one pass over phi) and derive every cost analytically. ``alpha`` is
    zeroed outside ``window`` (the overlap / hard-trim target support).
    """
    eta = fit.eta
    Phi = _eval_basis(eta, fit.info)
    alpha_all = Phi @ fit.omega_treat
    in_supp = (eta >= window[0]) & (eta <= window[1])
    alpha_b = np.where(in_supp, alpha_all, 0.0)
    win = in_supp.astype(float)
    gX_sorted = np.sort(Q - eta)
    n = len(gX_sorted)
    A = np.empty(len(phi_grid))
    B = np.empty(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        cdf = np.searchsorted(gX_sorted, phi - eta) / n
        probs = cdf if fit.direction == "below" else (1.0 - cdf)
        A[j] = float(np.mean(alpha_b * probs))     # in-window benefit
        B[j] = float(np.mean(win * probs))          # in-window treated mass (cost base)
    return A, B


def _interior_cost_range(
    A, B, phi_grid, alpha_min, alpha_max, n_grid: int = 61
) -> Tuple[float, float] | None:
    """Which *non-negative* treatment costs c give an *interior* welfare max.

    A cost is a cost: only ``c >= 0`` is explainable (a negative "cost" is a
    benefit and would let any positive alpha manufacture an interior). A crossing
    of ``alpha(eta) - c`` needs ``c < alpha_max``; we scan ``[0, alpha_max]``
    (padded), vectorized over c, and return the [min, max] c whose maximizer is
    interior AND strictly beats both boundaries.
    """
    hi = max(0.0, alpha_max)
    pad = 0.05 * (hi + 1e-9)
    c_grid = np.linspace(0.0, hi + pad, n_grid)
    U = A[:, None] - c_grid[None, :] * B[:, None]        # (n_phi, n_c)
    argmax = U.argmax(axis=0)
    gain = U[argmax, np.arange(U.shape[1])] - np.maximum(U[0], U[-1])
    interior = (argmax != 0) & (argmax != len(phi_grid) - 1) & (gain > 0)
    cs = c_grid[interior]
    return (float(cs.min()), float(cs.max())) if cs.size else None


def screen(
    dataset: str | RDDSample,
    *,
    out_root: Path = RUNS,
    eps: float = 0.1,
    max_n: int | None = SCREEN_MAX_N,
    description: str | None = None,
    phi_grid: np.ndarray | None = None,
) -> ScreenResult:
    """Fit the method on one dataset and write the three figures + description."""
    sample = load(dataset) if isinstance(dataset, str) else dataset
    name = sample.name
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    Qf, Xf, Yf, Df, threshold = _prep(sample, None)   # full cleaned data
    if max_n is not None and len(Yf) > max_n:
        (Q, X, Y, D), _, _ = _subsample([Qf, Xf, Yf, Df], max_n)
    else:
        Q, X, Y, D = Qf, Xf, Yf, Df
    direction = _detect_direction(D, Q)
    fit = _fit_pooled_plm(Q, X, Y, D, direction)

    lo, hi = fit.eta_eval
    grid = np.linspace(lo, hi, 500)
    Phi = _eval_basis(grid, fit.info)
    alpha_curve = Phi @ fit.omega_treat
    b_curve = Phi @ fit.omega_base

    first_stage_R2 = float(1.0 - np.var(fit.eta) / np.var(Q))

    # Overlap window [l0, u0] from the eps/1-eps quantiles of the observed index T.
    # This is where the (hard-trim) estimand lives; the spline extrapolates in the
    # low-density tails, so alpha's sign change and the welfare optimum must be
    # judged on this window, not on the full fitted support.
    T = Q - fit.eta
    l0 = threshold - float(np.quantile(T, 1.0 - eps))
    u0 = threshold - float(np.quantile(T, eps))
    l0, u0 = (max(min(l0, u0), lo), min(max(l0, u0), hi))
    hard_retention = float(np.mean((fit.eta >= l0) & (fit.eta <= u0)))

    # alpha's sign change is assessed ONLY on the overlap window.
    win = (grid >= l0) & (grid <= u0)
    alpha_win = alpha_curve[win] if win.any() else alpha_curve
    alpha_min, alpha_max = float(alpha_win.min()), float(alpha_win.max())
    # A sign change is only real if it carries data mass: evaluate alpha at the
    # actual in-window observations and measure the fraction with alpha < 0. A dip
    # into negatives in a near-empty tail (spline extrapolation) is not a crossing.
    eta_in = fit.eta[(fit.eta >= l0) & (fit.eta <= u0)]
    alpha_obs = _eval_basis(eta_in, fit.info) @ fit.omega_treat if eta_in.size else np.array([0.0])
    neg_alpha_mass = float(np.mean(alpha_obs < 0.0))
    MIN_MASS = 0.10  # each sign must carry >=10% of in-window data to count as a crossing
    crosses = bool(MIN_MASS <= neg_alpha_mass <= 1.0 - MIN_MASS)

    if phi_grid is None:
        span = 3.0 * float(np.std(Q))
        phi_grid = np.linspace(threshold - span, threshold + span, 401)

    # One pass gives the cost-free profiles; every cost is then U = A - c*B.
    # alpha is zeroed outside the overlap window (the hard-trim target support).
    A, B = _utility_profiles(fit, Q, phi_grid, window=(l0, u0))
    interior_range = _interior_cost_range(A, B, phi_grid, alpha_min, alpha_max)
    best0 = int(np.argmax(A))
    phi_star = float(phi_grid[best0])
    # Welfare gain of the interior optimum over the better boundary policy. The
    # optimum is inherently flat, so this is small even when real; at large n its
    # SIGN is what matters. A gain <= 0 means "treat all / none" is at least as
    # good -> not an interesting interior.
    boundary_gain = float(A[best0] - max(A[0], A[-1]))
    scale = float(np.max(np.abs(A))) + 1e-12
    boundary_gain_rel = boundary_gain / scale
    GAIN_TOL = 1e-4  # relative; filters exact numerical ties, not real flat gains
    interior0 = (best0 not in (0, len(phi_grid) - 1)) and (boundary_gain_rel > GAIN_TOL)

    # The utility figure shows the *interesting* case: zero cost if alpha already
    # gives an interior, otherwise a representative interior-inducing cost.
    if interior0:
        cost_shown = 0.0
    elif interior_range is not None:
        cost_shown = 0.5 * (interior_range[0] + interior_range[1])
    else:
        cost_shown = 0.0
    show = {0.0: A}
    if abs(cost_shown) > 1e-12:
        show[cost_shown] = A - cost_shown * B
    # Does the shown cost split the data mass (alpha - c changing sign on the data)?
    neg_mass_shown = float(np.mean(alpha_obs < cost_shown)) if eta_in.size else 0.0
    cost_splits_mass = bool(MIN_MASS <= neg_mass_shown <= 1.0 - MIN_MASS)

    _plot_alpha(fit, D, grid, alpha_curve, out_dir / "alpha.png", name, (l0, u0))
    _plot_b(fit, grid, b_curve, out_dir / "b.png", name)
    phi_star_shown, interior_shown = _plot_utility(
        phi_grid, show, cost_shown, threshold, out_dir / "utility.png", name
    )

    if crosses and interior0:
        verdict = ("INTERESTING — alpha changes sign on the data "
                   f"({neg_alpha_mass:.0%} of in-window mass negative); interior beats the "
                   f"boundary at zero cost (gain {boundary_gain_rel:+.1%} of welfare scale)")
    elif interior_range is not None and interior_shown and cost_splits_mass:
        verdict = (
            f"INTERESTING with an explainable cost — interior optimum at treatment "
            f"cost c≈{cost_shown:.3g} (outcome units; {neg_mass_shown:.0%} of in-window mass "
            f"below it); interior for c in [{interior_range[0]:.3g}, {interior_range[1]:.3g}]; "
            f"c must be economically justifiable"
        )
    else:
        verdict = (f"boundary — interior does not beat 'treat all/none' "
                   f"(gain {boundary_gain_rel:+.1%}; {neg_alpha_mass:.0%} of in-window mass negative)")

    # Confirmation gate: the welfare surface is flat, so a working-sample interior can
    # be noise (lending-ROI passed at 250k, boundary at 884k). Whenever the working
    # verdict is INTERESTING and we screened a subsample, re-check the gain — at the
    # SAME cost — on the FULL data, and downgrade if it does not survive.
    confirmed_full_n: bool | None = None
    boundary_gain_full = float("nan")
    n_confirm = len(Y)
    if verdict.startswith("INTERESTING") and len(Yf) > len(Y):
        fitf = _fit_pooled_plm(Qf, Xf, Yf, Df, _detect_direction(Df, Qf))
        lof, hif = fitf.eta_eval
        Tf = Qf - fitf.eta
        l0f = threshold - float(np.quantile(Tf, 1.0 - eps))
        u0f = threshold - float(np.quantile(Tf, eps))
        l0f, u0f = (max(min(l0f, u0f), lof), min(max(l0f, u0f), hif))
        spanf = 3.0 * float(np.std(Qf))
        pgf = np.linspace(threshold - spanf, threshold + spanf, 401)
        Af, Bf = _utility_profiles(fitf, Qf, pgf, window=(l0f, u0f))
        Uf = Af - cost_shown * Bf
        bestf = int(np.argmax(Uf))
        gainf = float(Uf[bestf] - max(Uf[0], Uf[-1]))
        boundary_gain_full = gainf / (float(np.max(np.abs(Uf))) + 1e-12)
        n_confirm = len(Yf)
        confirmed_full_n = bool(bestf not in (0, len(pgf) - 1) and boundary_gain_full > 1e-4)
        if not confirmed_full_n:
            verdict = (f"boundary on full n={len(Yf)} — the {len(Y)}-row interior did NOT "
                       f"survive confirmation (full-n gain {boundary_gain_full:+.1%})")

    result = ScreenResult(
        name=name, n=len(Y), first_stage_R2=first_stage_R2,
        eta_support=(lo, hi), overlap_window=(l0, u0),
        hard_retention=hard_retention, alpha_min=alpha_min, alpha_max=alpha_max,
        neg_alpha_mass=neg_alpha_mass,
        alpha_crosses_zero=crosses, phi_star_zero_cost=phi_star,
        phi_star_interior_zero_cost=interior0,
        boundary_gain=boundary_gain, boundary_gain_rel=boundary_gain_rel,
        confirmed_full_n=confirmed_full_n, boundary_gain_full=boundary_gain_full,
        n_confirm=n_confirm,
        interior_cost_range=(tuple(interior_range) if interior_range else None),
        cost_shown=float(cost_shown), phi_star_shown=float(phi_star_shown),
        phi_grid_edges=(float(phi_grid[0]), float(phi_grid[-1])),
        threshold=threshold, verdict=verdict, out_dir=str(out_dir),
    )
    _write_description(result, sample, description)
    (out_dir / "summary.json").write_text(json.dumps(result.__dict__, indent=2))
    return result


def _write_description(r: ScreenResult, sample: RDDSample, description: str | None) -> None:
    desc = description or (
        f"Running variable and covariates from the `{r.name}` adapter; "
        f"features: {', '.join(sample.feature_names)}."
    )
    lines = [
        f"# Screening report — {r.name}",
        "",
        f"**Verdict: {r.verdict}**",
        "",
        desc,
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| n (screened) | {r.n:,} |",
        f"| full-n confirmation | "
        f"{'not needed' if r.confirmed_full_n is None else ('CONFIRMED at n=%d (gain %+.2f%%)' % (r.n_confirm, r.boundary_gain_full*100) if r.confirmed_full_n else 'FAILED at n=%d (gain %+.2f%%)' % (r.n_confirm, r.boundary_gain_full*100))} |",
        f"| first-stage R² (Q on X) | {r.first_stage_R2:.3f} |",
        f"| η support | [{r.eta_support[0]:.3g}, {r.eta_support[1]:.3g}] |",
        f"| overlap window [l₀,u₀] (ε={0.1}) | [{r.overlap_window[0]:.3g}, {r.overlap_window[1]:.3g}] |",
        f"| hard-trim retention | {r.hard_retention:.3f} |",
        f"| α̂(η) range (in-window) | [{r.alpha_min:.3g}, {r.alpha_max:.3g}] |",
        f"| in-window data mass with α̂<0 | {r.neg_alpha_mass:.1%} |",
        f"| **α̂ crosses zero on the data** | **{r.alpha_crosses_zero}** |",
        f"| zero-cost φ* | {r.phi_star_zero_cost:.3g} (current cutoff {r.threshold:.3g}) |",
        f"| **interior beats boundary (zero cost)** | **{r.phi_star_interior_zero_cost}** |",
        f"| welfare gain of interior over boundary | {r.boundary_gain_rel:+.2%} of scale |",
        f"| interior-inducing cost range (outcome units) | "
        f"{('[%.3g, %.3g]' % r.interior_cost_range) if r.interior_cost_range else 'none'} |",
        f"| utility figure shown at c | {r.cost_shown:.3g} → φ*={r.phi_star_shown:.3g} |",
        "",
        "Figures: `alpha.png`, `b.png`, `utility.png`.",
        "",
        "_Screen uses the pooled PLM. A passing candidate should be re-run through "
        "`perfrdd_hard_trim` for inference-grade estimates._",
    ]
    Path(r.out_dir, "description.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", help="registered dataset name(s)")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--max-n", type=int, default=SCREEN_MAX_N)
    args = ap.parse_args(argv)
    for name in args.datasets:
        try:
            r = screen(name, eps=args.eps, max_n=args.max_n)
        except Exception as exc:  # keep screening the rest of the batch
            print(f"[{name}] FAILED: {exc}")
            continue
        flag = "★" if r.verdict.startswith("INTERESTING") else " "
        conf = "" if r.confirmed_full_n is None else f" conf={r.confirmed_full_n}@{r.n_confirm}"
        print(f"[{name}] {flag} n={r.n} R2={r.first_stage_R2:.2f} "
              f"neg_mass={r.neg_alpha_mass:.0%} gain={r.boundary_gain_rel:+.1%}{conf} "
              f"phi*={r.phi_star_zero_cost:.3g}  {r.verdict.split(' — ')[0]}")


if __name__ == "__main__":
    main()
