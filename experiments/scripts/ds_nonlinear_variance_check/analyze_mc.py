"""Read-only analysis of the existing nonlinear-design MC rows (point estimates)."""
import json, sys
import numpy as np
from scipy import stats
D = "/Users/zoudj/Documents/Codex/PerfRDD/code/experiments/runs/ds_four_role_coverage"
PHI = 0.15214328763980622
V = 4.729270448121349
rng = np.random.default_rng(0)
out = {}
for n in (800, 1600, 3200, 6400, 12800):
    rows = [json.loads(l) for l in open(f"{D}/nonlinear_n{n}.jsonl") if l.strip()]
    rows = sorted(rows, key=lambda r: r["rep"])
    cell = {"reps": len(rows)}
    for key in ("avg", "pooled", "fixed", "oracle_avg", "linear_avg"):
        x = np.array([r["est"][key] for r in rows])
        nv = n * x.var(ddof=1)
        # bootstrap s.e. of n Var over data sets
        bs = [n * rng.choice(x, len(x)).var(ddof=1) for _ in range(2000)]
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        mad = stats.median_abs_deviation(x, scale="normal")
        cell[key] = {"nVar": nv, "nVar_bse": float(np.std(bs)), "nVar_iqr": n * (iqr / 1.349) ** 2,
                     "nVar_mad": n * mad ** 2, "exkurt": float(stats.kurtosis(x)),
                     "skew": float(stats.skew(x)), "sqrt_n_bias": float(np.sqrt(n) * (x.mean() - PHI)),
                     "nVar_trim1pct": float(n * stats.trimboth(np.sort(x), 0.005).var(ddof=1))}
    out[n] = cell
    print(n, len(rows))
    for key in ("avg", "pooled", "fixed", "oracle_avg"):
        c = cell[key]
        print(f"  {key:11s} nVar={c['nVar']:7.3f} (bse {c['nVar_bse']:.3f})  iqr-based={c['nVar_iqr']:.3f} mad-based={c['nVar_mad']:.3f}"
              f" exkurt={c['exkurt']:.2f} skew={c['skew']:.2f} sqrt_n_bias={c['sqrt_n_bias']:+.3f}")
json.dump(out, open("analyze_mc.json", "w"), indent=1)
