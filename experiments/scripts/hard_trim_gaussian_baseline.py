"""Gaussian baseline for the hard-support-trimmed PerfRDD target.

This simulation is deliberately simple enough that the population policy
targets are known accurately.  It compares an oracle-index hard estimator, a
feasible split-sample hard estimator, the same feasible estimator with the
symmetric smooth support gate, and an untrimmed benchmark.

The feasible estimators use a deterministic nuisance support and disjoint
folds for the first stage, lower boundary, upper boundary, outcome nuisance,
T-distribution nuisance, and utility evaluation.  This mirrors the structure
of the hard-trimming theorem more closely than the application harness does.

Run from the repository root, for example:

    python -m experiments.scripts.hard_trim_gaussian_baseline --reps 200

Outputs are written to ``experiments/runs/hard_trim_gaussian_baseline``.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.perfrdd_smooth_trim import _smooth_trim_weights


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "hard_trim_gaussian_baseline"

P = 3
GAMMA = np.ones(P) / np.sqrt(P)
BETA = np.array([0.3, -0.2, 0.1])
PHI_0 = 0.0
EPS = 0.10
COST = 2.25
SIGMA_Y = 0.5
POLICY_BOUNDS = (-1.5, 1.5)
NUISANCE_SUPPORT = (-1.75, 1.75)
UNTRIMMED_SUPPORT = (-4.0, 4.0)
KNOT_EXPONENT = 11.0 / 60.0
DELTA_EXPONENT = 1.0 / 3.0

L0 = -float(norm.ppf(1.0 - EPS))
U0 = -float(norm.ppf(EPS))


@dataclass(frozen=True)
class GeneratedData:
    X: np.ndarray
    eta: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    D: np.ndarray
    Y: np.ndarray


@dataclass(frozen=True)
class SplineFit:
    info: Dict[str, Any]
    omega_treat: np.ndarray
    n_fit: int
    n_treated: int
    n_control: int


def alpha(eta: np.ndarray) -> np.ndarray:
    return 2.0 + np.asarray(eta)


def baseline(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta)
    return 0.5 * eta ** 2


def generate_data(n: int, seed: int) -> GeneratedData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, P))
    eta = rng.standard_normal(n)
    T = X @ GAMMA
    Q = T + eta
    D = (Q > PHI_0).astype(float)
    Y = D * alpha(eta) + baseline(eta) + X @ BETA
    Y = Y + rng.normal(0.0, SIGMA_Y, n)
    return GeneratedData(X=X, eta=eta, T=T, Q=Q, D=D, Y=Y)


def population_utility(phi: float, trimmed: bool = True) -> float:
    lo, hi = (L0, U0) if trimmed else (-10.0, 10.0)
    integrand = lambda eta: (eta - 0.25) * norm.sf(phi - eta) * norm.pdf(eta)
    return float(quad(integrand, lo, hi, epsabs=1e-12, limit=200)[0])


def population_score(phi: float, trimmed: bool = True) -> float:
    lo, hi = (L0, U0) if trimmed else (-10.0, 10.0)
    integrand = lambda eta: -(eta - 0.25) * norm.pdf(phi - eta) * norm.pdf(eta)
    return float(quad(integrand, lo, hi, epsabs=1e-12, limit=200)[0])


def population_curvature(phi: float, trimmed: bool = True) -> float:
    lo, hi = (L0, U0) if trimmed else (-10.0, 10.0)
    integrand = (
        lambda eta: (eta - 0.25) * (phi - eta)
        * norm.pdf(phi - eta) * norm.pdf(eta)
    )
    return float(quad(integrand, lo, hi, epsabs=1e-12, limit=200)[0])


def population_truth() -> Dict[str, float]:
    hard = float(brentq(lambda x: population_score(x, True), *POLICY_BOUNDS))
    untrimmed = float(
        brentq(lambda x: population_score(x, False), *POLICY_BOUNDS)
    )
    return {
        "hard_phi_star": hard,
        "untrimmed_phi_star": untrimmed,
        "hard_curvature": population_curvature(hard, True),
        "hard_utility": population_utility(hard, True),
        "hard_retention": float(norm.cdf(U0) - norm.cdf(L0)),
        "hard_treated_mass": float(
            quad(lambda e: norm.cdf(e) * norm.pdf(e), L0, U0)[0]
        ),
    }


def make_folds(n: int, seed: int) -> Dict[str, np.ndarray]:
    """Create disjoint folds with fixed proportions that cover all rows."""
    rng = np.random.default_rng(seed + 10_000_019)
    order = rng.permutation(n)
    cuts = np.floor(np.array([0.15, 0.25, 0.35, 0.65, 0.75]) * n).astype(int)
    chunks = np.split(order, cuts)
    names = ("first_stage", "boundary_l", "boundary_u", "outcome", "density", "utility")
    return dict(zip(names, chunks))


def _fit_gamma(data: GeneratedData, idx: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(idx)), data.X[idx]))
    coef, *_ = np.linalg.lstsq(design, data.Q[idx], rcond=None)
    return coef


def _predict_T(X: np.ndarray, gamma_hat: np.ndarray) -> np.ndarray:
    return gamma_hat[0] + X @ gamma_hat[1:]


def _fit_spline_plm(
    data: GeneratedData,
    idx: np.ndarray,
    eta_values: np.ndarray,
    support: tuple[float, float],
    ridge_scale: float = 0.0,
) -> SplineFit:
    """Fit the pooled PLM on a deterministic support.

    The B-spline base block contains the constant function, so a separate
    intercept is intentionally omitted.  This avoids an exact collinearity.
    When ``ridge_scale`` is positive, the spline blocks receive the same
    ``ridge_scale / sqrt(n_fit)`` penalty rate as the application estimator;
    the linear coefficients on X remain unpenalized.
    """
    if not np.isfinite(ridge_scale) or ridge_scale < 0.0:
        raise ValueError("ridge_scale must be finite and nonnegative")
    eta_idx = eta_values[idx]
    keep = (eta_idx >= support[0]) & (eta_idx <= support[1])
    fit_idx = idx[keep]
    eta_fit = eta_values[fit_idx]
    D_fit = data.D[fit_idx]
    n_treated = int(D_fit.sum())
    n_control = int(len(fit_idx) - n_treated)
    if len(fit_idx) < 50 or min(n_treated, n_control) < 10:
        raise ValueError(
            f"nuisance fold too small: n={len(fit_idx)}, "
            f"treated={n_treated}, control={n_control}"
        )

    n_eff = min(n_treated, n_control)
    n_interior_knots = max(3, int(round(n_eff ** KNOT_EXPONENT)))
    info = _basis_params(n_interior_knots, support)
    Phi = _eval_basis(eta_fit, info)
    design = np.column_stack((data.X[fit_idx], Phi, D_fit[:, None] * Phi))
    if ridge_scale == 0.0:
        coef, *_ = np.linalg.lstsq(design, data.Y[fit_idx], rcond=None)
    else:
        penalty = np.zeros(design.shape[1])
        penalty[P:] = ridge_scale / np.sqrt(len(fit_idx))
        lhs = design.T @ design + len(fit_idx) * np.diag(penalty)
        rhs = design.T @ data.Y[fit_idx]
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    n_basis = Phi.shape[1]
    omega_treat = coef[P + n_basis: P + 2 * n_basis]
    return SplineFit(
        info=info,
        omega_treat=omega_treat,
        n_fit=len(fit_idx),
        n_treated=n_treated,
        n_control=n_control,
    )


def _estimate_T_normal(T_values: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(T_values))
    sd = float(np.std(T_values, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-8:
        raise ValueError("estimated T standard deviation is degenerate")
    return mu, sd


def _maximize_utility(
    fit: SplineFit,
    eta_eval: np.ndarray,
    weights: np.ndarray,
    T_mu: float,
    T_sd: float,
) -> tuple[float, float, bool]:
    active = weights > 0.0
    if int(active.sum()) < 20 or float(weights[active].sum()) <= 0.0:
        raise ValueError("utility fold has insufficient positive trim weight")
    eta = eta_eval[active]
    weight = weights[active]
    effect = _eval_basis(eta, fit.info) @ fit.omega_treat
    value = weight * (effect - COST)
    denominator = float(weight.sum())

    def objective(phi: float) -> float:
        probability = norm.sf((phi - eta - T_mu) / T_sd)
        return float(np.sum(value * probability) / denominator)

    result = minimize_scalar(
        lambda x: -objective(float(x)),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    phi_hat = float(result.x)
    tol = 2e-4
    boundary = bool(
        phi_hat <= POLICY_BOUNDS[0] + tol or phi_hat >= POLICY_BOUNDS[1] - tol
    )
    return phi_hat, objective(phi_hat), boundary


def _estimate_boundaries(data: GeneratedData, folds: Dict[str, np.ndarray]) -> tuple[float, float]:
    idx_l = folds["boundary_l"]
    gamma_l = _fit_gamma(data, idx_l)
    T_l = _predict_T(data.X[idx_l], gamma_l)
    l_hat = PHI_0 - float(np.quantile(T_l, 1.0 - EPS))

    idx_u = folds["boundary_u"]
    gamma_u = _fit_gamma(data, idx_u)
    T_u = _predict_T(data.X[idx_u], gamma_u)
    u_hat = PHI_0 - float(np.quantile(T_u, EPS))
    if not l_hat < u_hat:
        raise ValueError(f"estimated overlap window is invalid: [{l_hat}, {u_hat}]")
    return l_hat, u_hat


def run_replication(n: int, seed: int) -> Dict[str, Any]:
    data = generate_data(n, seed)
    folds = make_folds(n, seed)
    utility_idx = folds["utility"]

    # Oracle-index hard estimator: eta and trim endpoints are known, while
    # alpha and the T distribution are estimated on independent folds.
    fit_oracle = _fit_spline_plm(
        data, folds["outcome"], data.eta, NUISANCE_SUPPORT
    )
    T_mu_oracle, T_sd_oracle = _estimate_T_normal(data.T[folds["density"]])
    eta_oracle_eval = data.eta[utility_idx]
    hard_oracle_weights = (
        (eta_oracle_eval >= L0) & (eta_oracle_eval <= U0)
    ).astype(float)
    oracle_phi, oracle_u, oracle_boundary = _maximize_utility(
        fit_oracle, eta_oracle_eval, hard_oracle_weights, T_mu_oracle, T_sd_oracle
    )

    # Feasible estimators share an independently estimated first stage and
    # nuisance fit.  Only the final hard indicator versus smooth gate differs.
    gamma_hat = _fit_gamma(data, folds["first_stage"])
    eta_hat = data.Q - _predict_T(data.X, gamma_hat)
    l_hat, u_hat = _estimate_boundaries(data, folds)
    fit_feasible = _fit_spline_plm(
        data, folds["outcome"], eta_hat, NUISANCE_SUPPORT
    )
    T_density_hat = _predict_T(data.X[folds["density"]], gamma_hat)
    T_mu, T_sd = _estimate_T_normal(T_density_hat)
    eta_feasible_eval = eta_hat[utility_idx]
    hard_weights = (
        (eta_feasible_eval >= l_hat) & (eta_feasible_eval <= u_hat)
    ).astype(float)
    hard_phi, hard_u, hard_boundary = _maximize_utility(
        fit_feasible, eta_feasible_eval, hard_weights, T_mu, T_sd
    )

    delta = float((u_hat - l_hat) * n ** (-DELTA_EXPONENT))
    smooth_weights = _smooth_trim_weights(
        eta_feasible_eval, l_hat, u_hat, delta
    )
    smooth_phi, smooth_u, smooth_boundary = _maximize_utility(
        fit_feasible, eta_feasible_eval, smooth_weights, T_mu, T_sd
    )

    # Untrimmed benchmark.  A wide deterministic support covers all but about
    # 6e-5 of a standard-normal eta distribution; basis evaluation clips the
    # vanishingly rare observations outside it.
    fit_untrimmed = _fit_spline_plm(
        data, folds["outcome"], eta_hat, UNTRIMMED_SUPPORT
    )
    untrimmed_phi, untrimmed_u, untrimmed_boundary = _maximize_utility(
        fit_untrimmed,
        eta_feasible_eval,
        np.ones_like(eta_feasible_eval),
        T_mu,
        T_sd,
    )

    hard_active = hard_weights > 0.0
    smooth_active = smooth_weights > 0.0
    return {
        "n": int(n),
        "seed": int(seed),
        "oracle_hard_phi": oracle_phi,
        "feasible_hard_phi": hard_phi,
        "feasible_smooth_phi": smooth_phi,
        "untrimmed_phi": untrimmed_phi,
        "oracle_hard_utility": oracle_u,
        "feasible_hard_utility": hard_u,
        "feasible_smooth_utility": smooth_u,
        "untrimmed_utility": untrimmed_u,
        "oracle_hard_boundary": oracle_boundary,
        "feasible_hard_boundary": hard_boundary,
        "feasible_smooth_boundary": smooth_boundary,
        "untrimmed_boundary": untrimmed_boundary,
        "l_hat": l_hat,
        "u_hat": u_hat,
        "delta": delta,
        "hard_retention": float(np.mean(hard_active)),
        "smooth_positive_retention": float(np.mean(smooth_active)),
        "smooth_effective_retention": float(np.mean(smooth_weights)),
        "hard_treated": int(np.sum(data.D[utility_idx][hard_active])),
        "hard_control": int(np.sum(1.0 - data.D[utility_idx][hard_active])),
        "outcome_fit_n": int(fit_feasible.n_fit),
        "outcome_fit_treated": int(fit_feasible.n_treated),
        "outcome_fit_control": int(fit_feasible.n_control),
        "first_stage_gamma_error": float(
            np.linalg.norm(gamma_hat[1:] - GAMMA)
        ),
        "first_stage_intercept": float(gamma_hat[0]),
    }


def _worker(task: tuple[int, int]) -> Dict[str, Any]:
    return run_replication(*task)


def _summarize(rows: Sequence[Dict[str, Any]], truth: Dict[str, float]) -> Dict[str, Any]:
    estimators = {
        "oracle_hard": ("oracle_hard_phi", truth["hard_phi_star"]),
        "feasible_hard": ("feasible_hard_phi", truth["hard_phi_star"]),
        "feasible_smooth": ("feasible_smooth_phi", truth["hard_phi_star"]),
        "untrimmed": ("untrimmed_phi", truth["untrimmed_phi_star"]),
    }
    out: Dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        subset = [row for row in rows if int(row["n"]) == n]
        item: Dict[str, Any] = {
            "n": n,
            "replications": len(subset),
            "mean_hard_retention": float(np.mean([r["hard_retention"] for r in subset])),
            "mean_smooth_effective_retention": float(
                np.mean([r["smooth_effective_retention"] for r in subset])
            ),
            "mean_l_hat": float(np.mean([r["l_hat"] for r in subset])),
            "mean_u_hat": float(np.mean([r["u_hat"] for r in subset])),
            "estimators": {},
        }
        smooth_minus_hard = np.asarray(
            [
                r["feasible_smooth_phi"] - r["feasible_hard_phi"]
                for r in subset
            ],
            dtype=float,
        )
        item["smooth_vs_hard"] = {
            "mean_difference": float(np.mean(smooth_minus_hard)),
            "mean_absolute_difference": float(np.mean(np.abs(smooth_minus_hard))),
            "max_absolute_difference": float(np.max(np.abs(smooth_minus_hard))),
        }
        for name, (field, target) in estimators.items():
            values = np.asarray([r[field] for r in subset], dtype=float)
            error = values - target
            is_trimmed = name != "untrimmed"
            target_utility = population_utility(target, is_trimmed)
            regrets = np.asarray(
                [
                    target_utility - population_utility(float(value), is_trimmed)
                    for value in values
                ],
                dtype=float,
            )
            boundary_field = field.replace("_phi", "_boundary")
            item["estimators"][name] = {
                "target": float(target),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "bias": float(np.mean(error)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "sd": float(np.std(values, ddof=1)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "mean_utility_regret": float(np.mean(regrets)),
                "median_utility_regret": float(np.median(regrets)),
                "boundary_rate": float(
                    np.mean([bool(r[boundary_field]) for r in subset])
                ),
            }
        out[str(n)] = item
    return out


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary: Dict[str, Any], path: Path) -> None:
    ns = np.asarray(sorted(int(k) for k in summary), dtype=int)
    names = ("oracle_hard", "feasible_hard", "feasible_smooth", "untrimmed")
    labels = {
        "oracle_hard": "oracle-index hard",
        "feasible_hard": "feasible hard",
        "feasible_smooth": "feasible smooth",
        "untrimmed": "untrimmed",
    }
    colors = dict(zip(names, ("C0", "C1", "C2", "C3")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in names:
        means = np.asarray([summary[str(n)]["estimators"][name]["mean"] for n in ns])
        lo = np.asarray([summary[str(n)]["estimators"][name]["q025"] for n in ns])
        hi = np.asarray([summary[str(n)]["estimators"][name]["q975"] for n in ns])
        rmse = np.asarray([summary[str(n)]["estimators"][name]["rmse"] for n in ns])
        target = summary[str(ns[0])]["estimators"][name]["target"]
        axes[0].plot(ns, means, "o-", color=colors[name], label=labels[name])
        axes[0].fill_between(ns, lo, hi, color=colors[name], alpha=0.12)
        axes[0].axhline(target, color=colors[name], ls=":", lw=0.9)
        axes[1].plot(ns, rmse, "o-", color=colors[name], label=labels[name])

    hard_ret = np.asarray([summary[str(n)]["mean_hard_retention"] for n in ns])
    smooth_ret = np.asarray([
        summary[str(n)]["mean_smooth_effective_retention"] for n in ns
    ])
    axes[2].plot(ns, hard_ret, "o-", label="hard retained mass")
    axes[2].plot(ns, smooth_ret, "s-", label="smooth effective mass")
    axes[2].axhline(0.8, color="black", ls=":", lw=1.0, label="population 0.8")

    axes[0].set_title("Estimated policy threshold")
    axes[0].set_ylabel(r"$\hat\phi$")
    axes[1].set_title("RMSE against each estimand")
    axes[1].set_ylabel("RMSE")
    axes[2].set_title("Evaluation-fold retained mass")
    axes[2].set_ylabel("fraction")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    ns: Iterable[int], reps: int, workers: int, out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(int(n), int(seed)) for n in ns for seed in range(reps)]
    if workers == 1:
        rows = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, tasks, chunksize=4))
    rows.sort(key=lambda row: (int(row["n"]), int(row["seed"])))

    truth = population_truth()
    summary = _summarize(rows, truth)
    payload = {
        "description": "Gaussian baseline for exact hard-support trimming",
        "dgp": {
            "p": P,
            "gamma": GAMMA.tolist(),
            "beta": BETA.tolist(),
            "phi_0": PHI_0,
            "epsilon": EPS,
            "cost": COST,
            "sigma_y": SIGMA_Y,
            "alpha": "2 + eta",
            "baseline": "0.5 * eta^2",
            "policy_bounds": list(POLICY_BOUNDS),
            "nuisance_support": list(NUISANCE_SUPPORT),
            "knot_exponent": KNOT_EXPONENT,
            "delta_exponent": DELTA_EXPONENT,
        },
        "truth": truth,
        "reps": reps,
        "summary": summary,
    }
    _write_csv(rows, out_dir / "replications.csv")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plot_summary(summary, out_dir / "summary.png")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.reps <= 0:
        parser.error("--reps must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if any(n < 500 for n in args.n):
        parser.error("all sample sizes must be at least 500")

    payload = run_experiment(args.n, args.reps, args.workers, args.out)
    print(json.dumps({"truth": payload["truth"], "summary": payload["summary"]}, indent=2))
    print(f"[wrote] {args.out / 'replications.csv'}")
    print(f"[wrote] {args.out / 'summary.json'}")
    print(f"[wrote] {args.out / 'summary.png'}")


if __name__ == "__main__":
    main()
