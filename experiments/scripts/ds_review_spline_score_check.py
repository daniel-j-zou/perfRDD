"""Exploratory check (2026-11-18): design ``hetero_ca`` of ``ds_review_ablations.py`` with a
cubic B-spline in X1 (knots at the octiles, two boundary columns dropped, centered) added
to the covariates of the score, the outcome fit and the D X block, so that the linear
predictor can absorb the nonlinear mean m(X1). J and the density interval are then set
from the full-sample pilot residuals, as in the taxi application. Point estimates only,
no bootstrap: rotated average over ``reps`` data sets (the seeds of the coverage run).

Result at n = 64,000 with 24 data sets: mean 1.500 with the linear score (gain lost 825%),
0.841 with the spline (gain lost 10%), against the target 0.770.

Run (from code/):
    python -m experiments.scripts.ds_review_spline_score_check 64000 24
"""
import dataclasses, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import BSpline
from experiments.scripts import ds_review_ablations as A
from experiments.scripts.ds_local_efficiency_sim import Sample

def augment(s, knots):
    x1 = s.X[:, 0]
    t = np.concatenate([[knots[0]] * 4, knots[1:-1], [knots[-1]] * 4])
    B = BSpline.design_matrix(np.clip(x1, knots[0], knots[-1]), t, 3).toarray()[:, 1:-1]
    B = B - B.mean(0)
    return Sample(X=np.column_stack([s.X, B]), Q=s.Q, D=s.D, Y=s.Y)

def one(args):
    key, n, rep, aug = args
    cfg = A.CFGS[key]
    seed = A.data_seed(A.SEED, n, rep)
    s, _ = A.generate(cfg, n, seed)
    if aug:
        knots = np.quantile(s.X[:, 0], np.linspace(0, 1, 8))
        s = augment(s, knots)
        Z = np.column_stack([np.ones(n), s.X]); g = np.linalg.lstsq(Z, s.Q, rcond=None)[0]
        e = s.Q - Z @ g; T = Z @ g
        cfg = dataclasses.replace(cfg, J=tuple(np.quantile(e, [0.0025, 0.9975])),
                                  T_support=(float(np.quantile(T, 0.0005)) - 0.5, float(np.quantile(T, 0.9995)) + 0.5))
    est, _ = A.estimate(s, cfg, np.array_split(np.random.default_rng([seed, 1]).permutation(n), 4))
    return est["avg"]

if __name__ == "__main__":
    n, reps = int(sys.argv[1]), int(sys.argv[2])
    with ProcessPoolExecutor(6) as ex:
        for aug in (False, True):
            v = np.array(list(ex.map(one, [("hetero_ca", n, r, aug) for r in range(reps)])))
            print(f"hetero_ca n={n} spline-in-X1={aug}: mean {v.mean():.4f} sd {v.std(ddof=1):.4f} (target {A.population('hetero_ca')['phi_star']:.4f}); gain lost at mean {100*A.gain_lost_share('hetero_ca', v.mean()):.1f}%")
