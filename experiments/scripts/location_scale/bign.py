"""Large-n runs of both estimators; records the U-hat curves (additive limit / pseudo-target)."""
import common as C
import numpy as np, json, time, sys

n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000
seeds = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
tag = sys.argv[3] if len(sys.argv) > 3 else "bign"
grid = np.linspace(C.PHI_LO, C.PHI_HI, 121)
out = []
for sd in seeds:
    rng = np.random.default_rng([99, n, sd])
    d = C.simulate(n, rng)
    X, Q, D, Y = d["X"], d["Q"], d["D"], d["Y"]
    folds = rng.permutation(n) % 4
    rec = dict(seed=sd, n=n)
    for scale, nm in [(True, "ls"), (False, "add")]:
        t0 = time.time()
        ests, curves, diag = [], [], []
        for r in range(4):
            fg, fo, fi, fu = [(folds == ((r + k) % 4)) for k in range(4)]
            idx = C.fit_index(X[fg], Q[fg], scale=scale)
            To, so = C.eval_index(X[fo], idx)
            ah, b2, arms = C.fit_outcome((Q[fo] - To) / so, X[fo], D[fo], Y[fo])
            Tj, sj = C.eval_index(X[fi], idx)
            lh, uh = np.quantile((C.PHI0 - Tj) / sj, [C.EPS, 1 - C.EPS])
            Tu, su = C.eval_index(X[fu], idx)
            xiu = (Q[fu] - Tu) / su
            e = xiu[(xiu >= lh) & (xiu <= uh)]
            Uh = C.make_Uhat(e, ah(e), Tj, sj, X[fi] @ b2)
            curves.append((Uh(grid) / fu.sum()).tolist())
            ests.append(C.argmax_step(Uh))
            diag.append(dict(gamma=idx[0].tolist(), delta=None if idx[1] is None else idx[1].tolist(),
                             beta2=b2.tolist(), l=lh, u=uh))
        rec[nm] = dict(phi=float(np.mean(ests)), rot=ests, curves=curves, diag=diag,
                       secs=time.time() - t0)
        print(sd, nm, rec[nm]["phi"], np.round(ests, 4), f"{time.time()-t0:.1f}s", flush=True)
    out.append(rec)
json.dump(dict(grid=grid.tolist(), runs=out), open(f"{tag}_n{n}.json", "w"), default=float)
print("done")
