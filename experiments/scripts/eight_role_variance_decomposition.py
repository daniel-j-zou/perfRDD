"""Analytic role-by-role variance decomposition for the eight-block baseline.

The Gaussian hard-trim simulation (``hard_trim_crossfit_regularization.py``)
assigns eight roles: three first-stage fits (for the outcome, density, and
evaluation blocks), two boundary blocks (each with its own first-stage fit
and empirical quantile), the outcome regression, the spline density, and the
evaluation average.  Write the utility-score expansion at the optimum as

    U_hat'(phi*) = sum_j P_{block j}[psi_j] + o_p(n^{-1/2}),

with one influence function psi_j per role.  With role fraction 1/8 each,

    fixed split:   N Var(phi_hat) -> 8 sum_j Var(psi_j) / H^2,
    role rotation: N Var(phi_hat) -> Var(sum_j psi_j) / H^2,

where H = U''(phi*).  This script evaluates every psi_j in closed form under
the DGP (X ~ N(0, I_3), eta ~ N(0, 1), T = gamma'X ~ N(0, 1), alpha(eta) =
2 + eta, b(eta) = eta^2 / 2, c = 2.25, eps = 0.10), integrates their 8 x 8
covariance matrix by quadrature, and reports where the rotated variance
departs from the diagonal sum.

Influence functions (F(e) = (e - 0.25) g(phi - e), I0 = 1{l0 <= e <= u0},
z = q_{0.9}, all loadings defined below):

    gamma_alpha : -(A_a0 + A_aT T) eta          outcome generated regressor
    gamma_g     : -(A_g0 + A_gT T) eta          density generated index
    gamma_U     : +a_U eta                      evaluation generated residual
    boundary_l  : -b_l {(0.9 - 1{T <= z}) / g(z) + (1 + z T) eta}
    boundary_u  : +b_u {(0.1 - 1{T <= -z}) / g(z) + (1 - z T) eta}
    outcome     : -r_alpha(D, eta, T) eps_Y     Riesz score on the interval J
    density     : -r_g(T),  r_g(t) = I0(phi - t) (phi - t - 0.25) f_eta(phi - t)
    utility     : -I0(eta) F(eta)

Run:
    python -m experiments.scripts.eight_role_variance_decomposition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.stats import norm

from experiments.scripts.hard_trim_gaussian_baseline import (
    COST,
    L0,
    NUISANCE_SUPPORT,
    SIGMA_Y,
    U0,
    population_truth,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "fixed_rotated_full_spline_20260915" / "eight_role_variance.json"
DEFAULT_SUMMARY = ROOT / "runs" / "fixed_rotated_full_spline_20260915" / "summary.json"
ROLES = ("gamma_alpha", "gamma_g", "gamma_U", "boundary_l", "boundary_u",
         "outcome", "density", "utility")
Z = float(norm.ppf(0.9))


def _nodes(breaks, order: int = 60):
    """Composite Gauss-Legendre nodes/weights against the N(0,1) density."""
    x, w = leggauss(order)
    points, weights = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        points.append(t)
        weights.append(0.5 * (b - a) * w * norm.pdf(t))
    return np.concatenate(points), np.concatenate(weights)


def _e_eta(function, lo: float, hi: float) -> float:
    return float(quad(lambda e: function(e) * norm.pdf(e), lo, hi,
                      epsabs=1e-13, limit=400)[0])


def loadings(phi: float) -> Dict[str, float]:
    """Scalars entering the eight influence functions."""
    amc = lambda e: e + 2.0 - COST                       # alpha(eta) - c
    F = lambda e: amc(e) * norm.pdf(phi - e)
    a_U = _e_eta(lambda e: e * F(e), L0, U0)             # = -U''(phi*)
    b_l = F(L0) * norm.pdf(L0)
    b_u = F(U0) * norm.pdf(U0)

    # Outcome Riesz representer on J: r = (D - e(eta)) A(eta) + k T.
    lo, hi = NUISANCE_SUPPORT
    e_ = norm.cdf                                         # P(D = 1 | eta)
    s_ = norm.pdf                                         # E[T D | eta]
    pv = lambda e: e_(e) * (1.0 - e_(e))
    load = lambda e: norm.pdf(phi - e) if L0 <= e <= U0 else 0.0
    P_J = float(norm.cdf(hi) - norm.cdf(lo))
    B = _e_eta(lambda e: s_(e) ** 2 / pv(e), lo, hi)
    C = (_e_eta(lambda e: s_(e) * load(e) / pv(e), L0, U0))
    k = C / (B - P_J)
    A = lambda e: (load(e) - k * s_(e)) / pv(e)
    split = lambda f: sum(_e_eta(f, a, b) for a, b in ((lo, L0), (L0, U0), (U0, hi)))
    riesz_second_moment = split(lambda e: pv(e) * A(e) ** 2 + 2 * k * A(e) * s_(e) + k ** 2)
    # Generated-regressor loading E[r xi (1, T)], xi = D alpha' + b' = D + eta.
    A_a0 = split(lambda e: A(e) * pv(e) + k * s_(e))
    A_aT = split(lambda e: A(e) * s_(e) * (1.0 + e - e_(e))
                 + k * (norm.cdf(e) - e * norm.pdf(e) + e))

    # Density representer r_g and its generated-index loading E[(1, T) r_g'(T)]
    # in weak form: A_g0 = E[T r_g(T)], A_gT = E[(T^2 - 1) r_g(T)].
    r_g = lambda t: amc(phi - t) * norm.pdf(phi - t)      # on [phi - U0, phi - L0]
    A_g0 = _e_eta(lambda t: t * r_g(t), phi - U0, phi - L0)
    A_gT = _e_eta(lambda t: (t * t - 1.0) * r_g(t), phi - U0, phi - L0)
    return {
        "a_U": a_U, "b_l": b_l, "b_u": b_u, "riesz_k": k,
        "riesz_second_moment": riesz_second_moment,
        "A_alpha_0": A_a0, "A_alpha_T": A_aT, "A_g_0": A_g0, "A_g_T": A_gT,
    }


def decomposition() -> Dict[str, Any]:
    truth = population_truth()
    phi, H = truth["hard_phi_star"], truth["hard_curvature"]
    q = loadings(phi)
    amc = lambda e: e + 2.0 - COST

    # Tensor quadrature over (T, eta); break at every discontinuity.
    t, wt = _nodes(sorted({-9.0, -Z, phi - U0, Z, phi - L0, 9.0}))
    e, we = _nodes(sorted({-9.0, L0, U0, 9.0}))
    T, E = np.meshgrid(t, e, indexing="ij")
    W = np.outer(wt, we)
    I0 = ((E >= L0) & (E <= U0)).astype(float)
    r_g = np.where((T >= phi - U0) & (T <= phi - L0),
                   amc(phi - T) * norm.pdf(phi - T), 0.0)
    gz = norm.pdf(Z)
    psi = {
        "gamma_alpha": -(q["A_alpha_0"] + q["A_alpha_T"] * T) * E,
        "gamma_g": -(q["A_g_0"] + q["A_g_T"] * T) * E,
        "gamma_U": q["a_U"] * E,
        "boundary_l": -q["b_l"] * ((0.9 - (T <= Z)) / gz + (1.0 + Z * T) * E),
        "boundary_u": q["b_u"] * ((0.1 - (T <= -Z)) / gz + (1.0 - Z * T) * E),
        "density": -r_g,
        "utility": -I0 * amc(E) * norm.pdf(phi - E),
    }
    means = {name: float(np.sum(W * value)) for name, value in psi.items()}
    cov = np.zeros((len(ROLES), len(ROLES)))
    for i, a in enumerate(ROLES):
        for j, b in enumerate(ROLES):
            if "outcome" in (a, b):
                # The outcome score multiplies the independent outcome error.
                cov[i, j] = (SIGMA_Y ** 2 * q["riesz_second_moment"]
                             if a == b else 0.0)
            else:
                cov[i, j] = float(np.sum(W * psi[a] * psi[b])) - means[a] * means[b]
    diag_sum = float(np.trace(cov))
    total = float(cov.sum())
    first_stage = [ROLES.index(r) for r in
                   ("gamma_alpha", "gamma_g", "gamma_U", "boundary_l", "boundary_u")]
    groups = {
        "first_stage_and_boundaries_only": float(
            cov[np.ix_(first_stage, first_stage)].sum()
            - np.trace(cov[np.ix_(first_stage, first_stage)])),
    }
    pairs = {
        f"{ROLES[i]}|{ROLES[j]}": float(2.0 * cov[i, j])
        for i in range(len(ROLES)) for j in range(i + 1, len(ROLES))
        if abs(cov[i, j]) > 1e-12
    }
    return {
        "truth": truth,
        "loadings": q,
        "influence_means": means,
        "roles": list(ROLES),
        "score_covariance": cov.tolist(),
        "score_variances": {r: float(cov[i, i]) for i, r in enumerate(ROLES)},
        "twice_covariances": pairs,
        "diagonal_sum_S": diag_sum,
        "off_diagonal_sum_C": total - diag_sum,
        "off_diagonal_groups": groups,
        "threshold_variance": {
            "fixed_eight_block": 8.0 * diag_sum / H ** 2,
            "rotated_or_full_sample": total / H ** 2,
            "ratio": 8.0 * diag_sum / total,
            "role_contributions_fixed": {
                r: 8.0 * float(cov[i, i]) / H ** 2 for i, r in enumerate(ROLES)},
        },
    }


def compare_with_simulation(result: Dict[str, Any], summary_path: Path) -> Dict[str, Any]:
    """Pooled N * sample-variance of each estimator in the saved Monte Carlo."""
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text())
    blocks = summary.get("summary", summary)
    out: Dict[str, Any] = {}
    for label in ("decoupled_8block", "rotated_8block", "full_sample"):
        values = []
        for key, block in blocks.items():
            if not isinstance(block, dict) or "estimators" not in block:
                continue
            est = block["estimators"][label]
            values.append((int(block["n"]), int(block["n"]) * est["sd"] ** 2,
                           int(block["n"]) * est["rmse"] ** 2))
        if values:
            values.sort()
            out[label] = {
                "n": [v[0] for v in values],
                "n_times_variance": [v[1] for v in values],
                "n_times_mse": [v[2] for v in values],
                "pooled_n_times_variance": float(np.mean([v[1] for v in values])),
                "pooled_n_times_mse": float(np.mean([v[2] for v in values])),
            }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    result = decomposition()
    result["simulation"] = compare_with_simulation(result, args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    H = result["truth"]["hard_curvature"]
    print(f"phi* = {result['truth']['hard_phi_star']:.6f}   H = U''(phi*) = {H:.6f}")
    print("loadings:", {k: round(v, 6) for k, v in result["loadings"].items()})
    print("\nrole            Var(psi)     8 Var / H^2")
    for r in ROLES:
        v = result["score_variances"][r]
        print(f"{r:14s} {v:11.6f} {8 * v / H ** 2:12.3f}")
    print(f"\nS = sum Var(psi_j)        = {result['diagonal_sum_S']:.6f}")
    print(f"C = 2 sum_{{j<k}} Cov        = {result['off_diagonal_sum_C']:.6f}")
    for name, value in sorted(result["twice_covariances"].items(), key=lambda kv: kv[1]):
        print(f"    2 Cov({name}) = {value:+.6f}")
    tv = result["threshold_variance"]
    print(f"\nfixed eight-block  N Var = 8 S / H^2     = {tv['fixed_eight_block']:.3f}")
    print(f"rotated / full     N Var = (S + C) / H^2 = {tv['rotated_or_full_sample']:.3f}")
    print(f"ratio = 8 S / (S + C) = {tv['ratio']:.3f}")
    for label, item in result["simulation"].items():
        print(f"simulation {label:17s} pooled N*Var = {item['pooled_n_times_variance']:.2f}"
              f"  N*MSE = {item['pooled_n_times_mse']:.2f}")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
