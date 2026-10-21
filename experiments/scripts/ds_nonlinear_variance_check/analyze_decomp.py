"""Summarize decomp_n*.jsonl: n Var of the rotated estimator, its linear proxy, and role parts."""
import json, sys
import numpy as np
sys.argv += []
import indep_pieces as P
PHI, H, V = P.PHI, P.H, P.V

def load(n):
    rows = [json.loads(l) for l in open(f"decomp_n{n}.jsonl") if l.strip()]
    return sorted(rows, key=lambda r: r["rep"])

def bse_nvar(x, n, B=1000, rng=np.random.default_rng(1)):
    idx = rng.integers(0, len(x), (B, len(x)))
    return float(np.std(n * x[idx].var(axis=1, ddof=1)))

def summarize(n, rows):
    R = len(rows)
    g = lambda k: np.array([r[k] for r in rows])
    avg, pooled, rot, lin = g("avg"), g("pooled"), g("rot"), g("lin")
    Zh, Hh = g("Zhat"), g("Hhat")
    lin_avg = lin.mean(1)
    score_phi = PHI + Zh / H           # threshold implied by the score at phi*, exact curvature
    out = {"n": n, "R": R}
    out["nVar_avg"] = n * avg.var(ddof=1); out["nVar_avg_bse"] = bse_nvar(avg, n)
    out["nVar_pooled"] = n * pooled.var(ddof=1)
    out["nVar_rot_mean"] = float(np.mean([n * rot[:, k].var(ddof=1) for k in range(4)]))
    C = np.corrcoef(rot.T); out["mean_cross_rot_corr"] = float((C.sum() - 4) / 12)
    out["nVar_lin_avg"] = n * lin_avg.var(ddof=1)
    out["nVar_lin_rot_mean"] = float(np.mean([n * lin[:, k].var(ddof=1) for k in range(4)]))
    Cl = np.corrcoef(lin.T); out["mean_cross_rot_corr_lin"] = float((Cl.sum() - 4) / 12)
    rem = avg - lin_avg
    out["nVar_rem"] = n * rem.var(ddof=1)
    out["2nCov_lin_rem"] = 2 * n * np.cov(lin_avg, rem)[0, 1]
    out["sqrt_n_bias_avg"] = float(np.sqrt(n) * (avg.mean() - PHI))
    # score vs curvature conversion
    sp = score_phi.mean(1)
    out["nVar_scorephi_avg"] = n * sp.var(ddof=1)
    out["nVar_scorephi_rot_mean"] = float(np.mean([n * score_phi[:, k].var(ddof=1) for k in range(4)]))
    out["mean_Hhat"] = float(Hh.mean()); out["sd_Hhat"] = float(Hh.std())
    # role parts (Z scale), rotation-averaged then scaled to threshold: n Var(part/H)
    parts = {}
    for p in ("eval", "out", "dens", "end"):
        a, l = g(p), g(p + "_lin")
        aa, la = a.mean(1), l.mean(1)
        parts[p] = {"nVar_actual_avg": n * aa.var(ddof=1) / H ** 2,
                    "nVar_lin_avg": n * la.var(ddof=1) / H ** 2,
                    "nVar_diff_avg": n * (aa - la).var(ddof=1) / H ** 2,
                    "corr_actual_lin_avg": float(np.corrcoef(aa, la)[0, 1]),
                    "nVar_actual_rot": float(np.mean([n * a[:, k].var(ddof=1) for k in range(4)])) / H ** 2,
                    "nVar_lin_rot": float(np.mean([n * l[:, k].var(ddof=1) for k in range(4)])) / H ** 2,
                    "mean_cross_rot_corr_actual": float((np.corrcoef(a.T).sum() - 4) / 12)}
    out["parts"] = parts
    tot_a = sum(g(p) for p in ("eval", "out", "dens", "end")).mean(1)
    out["nVar_sum_parts_avg"] = n * tot_a.var(ddof=1) / H ** 2
    out["nVar_Zhat_avg_over_H2"] = n * Zh.mean(1).var(ddof=1) / H ** 2
    return out

if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        rows = load(n)
        s = summarize(n, rows)
        print(json.dumps(s, indent=1))
