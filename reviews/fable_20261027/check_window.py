import numpy as np
from experiments.scripts import taxi_four_role as M
df = M.load_frame("full")
for lam, score, name in [(0.5, "distlog", "R2-E1"), (1.0, "distlog", "S2-E1")]:
    data = M.make_data(df, lam, "distance", score=score)
    Xt = M._xt(data.Z)
    g = np.linalg.lstsq(Xt, data.Q, rcond=None)[0]
    T = Xt @ g; eta = data.Q - T
    phi0 = float(M.boxcox(M.FARE0, lam))
    caps = np.quantile(eta, [0.005, 0.995]); J = (np.quantile(eta, 0.0025), np.quantile(eta, 0.9975))
    print(f"== {name}: n={len(eta)} treated share={data.D.mean():.4f} phi0={phi0:.4f} sd_eta={eta.std():.4f}")
    for eps in (0.05, 0.075, 0.10):
        l0 = phi0 - np.quantile(T, 1 - eps); u0 = phi0 - np.quantile(T, eps)
        print(f"  eps={eps}: l0={l0:.4f} u0={u0:.4f}  caps=[{caps[0]:.4f},{caps[1]:.4f}]  J=[{J[0]:.4f},{J[1]:.4f}]"
              f"  l0<J_lo:{l0<J[0]}  l0<cap_lo:{l0<caps[0]}  u0>cap_hi:{u0>caps[1]}  u0>J_hi:{u0>J[1]}")
    # overlap (empirical treatment prob) at the caps and at the residual quantiles
    for q in (0.005, 0.025, 0.5, 0.975, 0.995):
        e = np.quantile(eta, q)
        print(f"  P(T > phi0 - eta) at eta=q_{q}: {np.mean(T > phi0 - e):.4f}")
    print(f"  share of eta in [l0,u0] at eps=.05: {np.mean((eta>=phi0-np.quantile(T,.95))&(eta<=phi0-np.quantile(T,.05))):.4f}")
