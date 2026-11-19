"""How much dependence between X and eta the estimator tolerates (author, 2026-11-19: "Lets
definitely do 1").

The taxi-matched design of ``ds_review_ablations.py`` (hetero_ca / hetero_cb) has
eta = m(X1) + s(X) zeta with m(X1) = A (e^{-B X1} - e^{B^2/2}) + A B e^{B^2/2} X1 (the part
of a decreasing exponential orthogonal to 1, X1, X2) and s(X) = exp{k (0.4 X1 + 0.2 X2)}, at
A = 1.056159, B = 0.588515, k = 0.335396, which matches the three taxi diagnostics. This
script scales the dependence by a factor f: A -> f A and k -> f k (B fixed), so f = 0 is no
dependence (eta = zeta ~ N(0, 1), independent of X) and f = 1 is taxi strength; and it runs
the two parts alone at f = 1 (mean only: k = 0; spread only: A = 0). Each setting is run with
the effect and baseline acting on zeta (the standardized residual of the correct score,
design c-a) and on eta (design c-b).

For each setting: the diagnostics of eta (share of Var eta and of Var eta^2 explained by X,
spread factor of eta across X1 deciles, the linear predictor's R^2), the target, and, over
``reps`` data sets at n (no bootstrap; the bias does not shrink with n), the mean and
standard deviation of the rotated average and of full-sample reuse, the bias in standard
deviations of the rotated average at n and at the taxi n = 633,616 (scaled by sqrt n), and
the share of the target's gain over phi0 that the mean forgoes. Data sets share draws
across settings (seed 20261019 + 104729 n + rep).

Run (from code/):
    python -m experiments.scripts.ds_dependence_path --n 64000 --reps 200 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.scripts import ds_review_ablations as A
from experiments.scripts.ds_fullsample_reuse import estimate_full

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_dependence_path"
FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
N_TAXI = 633_616


def settings():
    out = []
    for f in FACTORS:
        out.append((f"f{f:.2f}", f, f * A.C_A, f * A.C_K))
    out.append(("mean_only", 1.0, A.C_A, 0.0))
    out.append(("spread_only", 1.0, 0.0, A.C_K))
    return out


def make_cfg(name, a, k, latent):
    return A._het(f"dep_{name}_{latent}", f"dependence path {name}, effect of {latent}", k, a,
                  A.C_B, latent=latent)


def diagnostics(a, k) -> dict:
    xg = A.XGrid(a, A.C_B, k, m=40001, lim=9.0)
    E2 = np.sum(xg.w * (xg.m ** 2 + xg.s ** 2))
    c2 = xg.m ** 2 + xg.s ** 2
    e4 = xg.m ** 4 + 6 * xg.m ** 2 * xg.s ** 2 + 3 * xg.s ** 4
    rng = np.random.default_rng(3)
    x1 = rng.standard_normal(2_000_000)
    x2 = rng.integers(0, 2, len(x1)) - 0.5
    eta = A.m_tilde(x1, a, A.C_B) + A.het_scale(x1, x2, k) * rng.standard_normal(len(x1))
    dec = np.searchsorted(np.quantile(x1, np.linspace(0.1, 0.9, 9)), x1)
    sd = np.array([eta[dec == d].std() for d in range(10)])
    return {"r2_eta": float(np.sum(xg.w * xg.m ** 2) / E2),
            "r2_eta2": float((np.sum(xg.w * c2 ** 2) - E2 ** 2) / (np.sum(xg.w * e4) - E2 ** 2)),
            "spread_factor": float(sd.max() / sd.min()),
            "first_stage_r2": float(1.0 - eta.var() / (x1 + 0.5 * x2 + eta).var())}


def one(task):
    cfg, n, rep = task
    seed = A.data_seed(A.SEED, n, rep)
    s, _ = A.generate(cfg, n, seed)
    est, _ = A.estimate(s, cfg, np.array_split(np.random.default_rng([seed, 1]).permutation(n), 4))
    return rep, est["avg"], estimate_full(s, cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64000)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"path_n{args.n}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.perf_counter()
    for name, f, a, k in settings():
        diag = diagnostics(a, k)
        for latent in ("zeta", "eta"):
            key = f"{name}_{latent}"
            if key in res:
                continue
            cfg = make_cfg(name, a, k, latent)
            U, info = A.het_utility(cfg)
            phi = A._argmax_numeric(U, cfg.policy)
            gain = U(phi) - U(cfg.phi0)
            with ProcessPoolExecutor(args.workers) as ex:
                rows = list(ex.map(one, [(cfg, args.n, r) for r in range(args.reps)]))
            avg = np.array([r[1] for r in rows])
            full = np.array([r[2] for r in rows])
            sd = float(avg.std(ddof=1))
            res[key] = {
                "setting": name, "factor": f, "A": a, "k": k, "latent": latent, **diag,
                "target": phi, "gain_over_phi0": gain, "window": info["window"],
                "mean_avg": float(avg.mean()), "sd_avg": sd,
                "mean_full": float(full.mean()), "sd_full": float(full.std(ddof=1)),
                "bias_sd_n": float((avg.mean() - phi) / sd),
                "bias_sd_taxi_n": float((avg.mean() - phi) / sd * np.sqrt(N_TAXI / args.n)),
                "gain_lost_avg": float((U(phi) - U(avg.mean())) / gain),
                "gain_lost_full": float((U(phi) - U(full.mean())) / gain),
                "reps": args.reps, "n": args.n}
            path.write_text(json.dumps(res, indent=1) + "\n")
            r = res[key]
            print(f"{key:>18}: R2(eta) {r['r2_eta']:.3f} R2(eta2) {r['r2_eta2']:.3f} spread "
                  f"{r['spread_factor']:.2f} | target {phi:.4f} avg {r['mean_avg']:.4f} "
                  f"(sd {sd:.4f}) full {r['mean_full']:.4f} | bias/sd {r['bias_sd_n']:+.2f} "
                  f"(taxi n {r['bias_sd_taxi_n']:+.1f}) | gain lost {100 * r['gain_lost_avg']:.1f}% "
                  f"(full {100 * r['gain_lost_full']:.1f}%) | {time.perf_counter() - t0:.0f}s",
                  flush=True)
    print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
