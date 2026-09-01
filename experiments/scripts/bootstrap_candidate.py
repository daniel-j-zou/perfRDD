"""Bootstrap the interior welfare optimum of a screened candidate dataset.

Companion to ``screen_candidate``. Draws a working analysis sample, computes the
point-estimate optimum phi*, and resamples it B times to report the sampling
distribution of phi*, the fraction of resamples whose optimum is interior, and
the fraction whose alpha still changes sign on the overlap window (a robustness
check that the interesting result is not an artifact of one sample).

    python -m experiments.scripts.bootstrap_candidate lending_default --B 300
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

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
from experiments.scripts.screen_candidate import _utility_profiles

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "bootstrap_candidate"


def _optimum(Q, X, Y, threshold, eps, phi_grid) -> Tuple[float, float, bool]:
    """Return (phi*, neg-alpha mass on overlap window, interior flag)."""
    D = (Q > threshold).astype(float)
    fit = _fit_pooled_plm(Q, X, Y, D, _detect_direction(D, Q))
    lo, hi = fit.eta_eval
    T = Q - fit.eta
    l0 = threshold - float(np.quantile(T, 1.0 - eps))
    u0 = threshold - float(np.quantile(T, eps))
    l0, u0 = (max(min(l0, u0), lo), min(max(l0, u0), hi))
    eta_in = fit.eta[(fit.eta >= l0) & (fit.eta <= u0)]
    a_obs = _eval_basis(eta_in, fit.info) @ fit.omega_treat if eta_in.size else np.array([0.0])
    neg = float(np.mean(a_obs < 0.0))
    A, _ = _utility_profiles(fit, Q, phi_grid, window=(l0, u0))
    j = int(np.argmax(A))
    return float(phi_grid[j]), neg, (j not in (0, len(phi_grid) - 1))


@dataclass
class BootResult:
    name: str
    n_work: int
    threshold: float
    B: int
    phi_star_point: float
    neg_mass_point: float
    interior_point: bool
    phi_star_mean: float
    phi_star_sd: float
    phi_star_ci95: Tuple[float, float]
    frac_interior: float
    frac_sign_changing: float
    neg_mass_mean: float


def bootstrap(dataset: str | RDDSample, *, B: int = 300, eps: float = 0.1,
              n_work: int = DEFAULT_MAX_N, seed: int = 11) -> BootResult:
    sample = load(dataset) if isinstance(dataset, str) else dataset
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = np.asarray(sample.Y, dtype=float)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y = Q[keep], X[keep], Y[keep]
    n_work = min(len(Y), n_work)
    (Qa, Xa, Ya), n_used, _ = _subsample([Q, X, Y], n_work)
    phi_grid = np.linspace(threshold - 3 * np.std(Qa), threshold + 3 * np.std(Qa), 401)

    p0, neg0, int0 = _optimum(Qa, Xa, Ya, threshold, eps, phi_grid)
    rng = np.random.default_rng(seed)
    phis, negs, ints = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n_used, n_used)
        p, ng, it = _optimum(Qa[idx], Xa[idx], Ya[idx], threshold, eps, phi_grid)
        phis.append(p); negs.append(ng); ints.append(it)
    phis, negs, ints = map(np.asarray, (phis, negs, ints))
    ci = tuple(float(x) for x in np.percentile(phis, [2.5, 97.5]))
    res = BootResult(
        name=sample.name, n_work=n_used, threshold=float(threshold), B=B,
        phi_star_point=p0, neg_mass_point=neg0, interior_point=int0,
        phi_star_mean=float(phis.mean()), phi_star_sd=float(phis.std()),
        phi_star_ci95=ci, frac_interior=float(ints.mean()),
        frac_sign_changing=float(np.mean((negs >= 0.1) & (negs <= 0.9))),
        neg_mass_mean=float(negs.mean()),
    )
    out = RUNS / sample.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap.json").write_text(json.dumps(asdict(res), indent=2))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n-work", type=int, default=DEFAULT_MAX_N)
    a = ap.parse_args(argv)
    t = time.time()
    r = bootstrap(a.dataset, B=a.B, eps=a.eps, n_work=a.n_work)
    print(f"[{r.name}] n_work={r.n_work} phi*={r.phi_star_point:.3f} "
          f"(cutoff {r.threshold:g}) 95% CI=[{r.phi_star_ci95[0]:.3f},{r.phi_star_ci95[1]:.3f}] "
          f"interior={r.frac_interior:.0%} sign-changing={r.frac_sign_changing:.0%} "
          f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
