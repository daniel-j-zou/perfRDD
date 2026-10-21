"""Paired (control-variate) comparison of the estimator with its first-order linear proxy
on the same data sets.  lin_avg = phi* - H^{-1} P_all(psi_U+psi_o+psi_T+psi_gamma) has
n Var = sigma^2/H^2 = V exactly; so  V + n[Var(est) - Var(lin)]  estimates n Var(est)
with a much smaller Monte Carlo error than n Var(est) itself."""
import json, sys
import numpy as np
import indep_pieces as P
PHI, H, V = P.PHI, P.H, P.V

def stats(n, est, lin, B=2000, seed=3):
    rng = np.random.default_rng(seed)
    R = len(est)
    def f(idx):
        e, l = est[idx], lin[idx]
        return n * e.var(ddof=1), n * l.var(ddof=1), n * (e.var(ddof=1) - l.var(ddof=1))
    base = f(np.arange(R))
    bs = np.array([f(rng.integers(0, R, R)) for _ in range(B)])
    return {"nVar_est": base[0], "se_est": bs[:, 0].std(), "nVar_lin": base[1], "se_lin": bs[:, 1].std(),
            "diff": base[2], "se_diff": bs[:, 2].std(), "corr": float(np.corrcoef(est, lin)[0, 1])}

for n in map(int, sys.argv[1:]):
    rows = sorted([json.loads(l) for l in open(f"decomp_n{n}.jsonl") if l.strip()], key=lambda r: r["rep"])
    g = lambda k: np.array([r[k] for r in rows])
    avg, pooled, rot, lin, Zh = g("avg"), g("pooled"), g("rot"), g("lin"), g("Zhat")
    lin_avg = lin.mean(1)
    sp = (PHI + Zh / H)
    print(f"n={n}, R={len(rows)}, V={V:.4f}, 4V={4*V:.3f};  sqrt(n) bias avg={np.sqrt(n)*(avg.mean()-PHI):+.3f} (se {np.sqrt(n)*avg.std()/np.sqrt(len(avg)):.3f}), pooled={np.sqrt(n)*(pooled.mean()-PHI):+.3f}")
    for name, est, l, ref in (("avg", avg, lin_avg, V), ("pooled", pooled, lin_avg, V),
                               ("score@phi* (avg of rot)", sp.mean(1), lin_avg, V),
                               ("rot0 (fixed)", rot[:, 0], lin[:, 0], 4 * V)):
        s = stats(n, est, l)
        print(f"  {name:24s} nVar={s['nVar_est']:.3f} ({s['se_est']:.3f})  proxy nVar={s['nVar_lin']:.3f} ({s['se_lin']:.3f})"
              f"  diff={s['diff']:+.3f} ({s['se_diff']:.3f})  corr={s['corr']:.4f}  CV estimate={ref + s['diff']:.3f} -> ratio {(ref + s['diff'])/ref:.3f}")
    # all four rotations pooled for the fixed-assignment check
    d = [stats(n, rot[:, k], lin[:, k], B=500)["diff"] for k in range(4)]
    print(f"  fixed, mean over k of diff = {np.mean(d):+.3f}  -> CV {4*V+np.mean(d):.3f}")
    C = np.corrcoef(rot.T); Cl = np.corrcoef(lin.T)
    print(f"  mean cross-rotation corr: est {(C.sum()-4)/12:+.4f}, proxy {(Cl.sum()-4)/12:+.4f}")
