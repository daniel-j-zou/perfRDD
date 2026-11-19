"""Sample analogue of vartheta_J (Lemma ds-gram) for the taxi main specification R2-E1.

Review-6 item o13 (2026-11-19): the value reported in Appendix 5 of differing_slopes.tex.
Reads the processed taxi data; writes runs/taxi_theta_J/theta_J_taxi.json.
Run from the code/ folder: python -m experiments.scripts.taxi_theta_J
"""
import json, sys, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_section6 as S6
t0 = time.time()
out = pathlib.Path(__file__).resolve().parents[1] / "runs" / "taxi_theta_J"
out.mkdir(parents=True, exist_ok=True)
cfg = S6.SAMPLES["daytime"]
df = M.load_frame(cfg["frame"])
data, design = S6.build(df, "R2-E1")
n = len(data.Q)
Zstd = data.Z                                   # columns standardized, as in the estimator
A = M._xt(Zstd)
g = np.linalg.lstsq(A, data.Q, rcond=None)[0]
T = A @ g; eta = data.Q - T
J = design.J; phi0 = design.phi0
print("n", n, "p", A.shape[1], "J", J, "phi0", phi0, "window", design.window, flush=True)
# raw-unit covariates, centered (same columns before standardizing)
d = df["Trip_Distance"].to_numpy(float)
sd = None
def versions():
    yield "standardized_columns", A
    # whitened: Var(X) = I
    C = np.cov(Zstd, rowvar=False)
    w, V = np.linalg.eigh(C)
    print("eig Var(Zstd): min %.3e max %.3e" % (w[0], w[-1]), flush=True)
    W = V @ np.diag(w ** -0.5) @ V.T
    yield "whitened", M._xt(Zstd @ W)
res = {"n": n, "J_code": list(J), "J_paper": [0.5 * J[0], 0.5 * J[1]], "phi0_code": phi0}
es = np.linspace(J[0], J[1], 801)
thr = phi0 - es                                  # decreasing in e
order = np.argsort(T); Ts = T[order]
for name, X in versions():
    Xs = X[order]
    tot = Xs.T @ Xs / n
    # bins by threshold (ascending thresholds)
    th = thr[::-1]
    cut = np.searchsorted(Ts, th, side="right")   # number with T <= th
    M0 = []
    acc = np.zeros_like(tot); prev = 0
    for c in cut:
        if c > prev:
            blk = Xs[prev:c]; acc = acc + blk.T @ blk / n; prev = c
        M0.append(acc.copy())
    lam0 = np.array([np.linalg.eigvalsh(m)[0] for m in M0])[::-1]          # indexed by es
    lam1 = np.array([np.linalg.eigvalsh(tot - m)[0] for m in M0])[::-1]
    share1 = (n - cut[::-1]) / n
    k0, k1 = int(lam0.argmin()), int(lam1.argmin())
    res[name] = {"vartheta_J": float(min(lam0.min(), lam1.min())),
                 "min_M0": float(lam0.min()), "argmin_e_M0_paper": float(0.5 * es[k0]),
                 "min_M1": float(lam1.min()), "argmin_e_M1_paper": float(0.5 * es[k1]),
                 "lam_min_total": float(np.linalg.eigvalsh(tot)[0]),
                 "treated_side_share_at_lower_J": float(share1[0]),
                 "treated_side_share_at_upper_J": float(share1[-1]),
                 "M1_at": {f"{0.5*es[i]:.3f}": float(lam1[i]) for i in range(0, 801, 100)},
                 "M0_at": {f"{0.5*es[i]:.3f}": float(lam0[i]) for i in range(0, 801, 100)}}
    print(name, json.dumps(res[name], indent=1), flush=True)
# window-only version (central 99% of eta)
res["seconds"] = time.time() - t0
(out / "theta_J_taxi.json").write_text(json.dumps(res, indent=1))
print("done", res["seconds"])
