import json, sys, numpy as np
import indep_pieces as P
PHI, H, V = P.PHI, P.H, P.V
for n in map(int, sys.argv[1:]):
    rows = sorted([json.loads(l) for l in open(f"decomp_n{n}.jsonl")], key=lambda r: r["rep"])
    g = lambda k: np.array([r[k] for r in rows])
    rot, Zh, Hh, lin = g("rot"), g("Zhat"), g("Hhat"), g("lin")
    b = rot - PHI
    a = Zh / H
    c = Zh / Hh
    nv = lambda x: n * x.var(ddof=1, axis=0).mean()
    print(f"n={n}: per-rotation nVar: rot {nv(b):.2f}  Z/H {nv(a):.2f}  Z/Hhat(phi_k) {nv(c):.2f}  lin {nv(lin):.2f};"
          f" mean Hhat {Hh.mean():+.4f} (H {H:+.4f}), n*Var(Hhat) {n*Hh.var(axis=0,ddof=1).mean():.3f},"
          f" corr(Z^2,Hhat) {np.mean([np.corrcoef(Zh[:,k]**2, Hh[:,k])[0,1] for k in range(4)]):+.3f},"
          f" corr(Z,Hhat) {np.mean([np.corrcoef(Zh[:,k], Hh[:,k])[0,1] for k in range(4)]):+.3f}")
    # average estimator analogs
    print(f"     avg: nVar(mean rot) {n*b.mean(1).var(ddof=1):.3f}  nVar(mean Z/Hhat) {n*c.mean(1).var(ddof=1):.3f}  nVar(mean Z/H) {n*a.mean(1).var(ddof=1):.3f}")
