import sys, json
import numpy as np
sys.path.insert(0, "/Users/zoudj/Documents/Codex/PerfRDD/code")
import mc_decomp as M
import indep_pieces as P
from experiments.scripts.ds_local_efficiency_sim import generate, _xt
n = int(sys.argv[1]); R = int(sys.argv[2]); off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
means, within, parts_means = [], [], []
for rep in range(off, off + R):
    seed = M.SEED0 + 104_729 * n + rep
    s = generate(M.DESIGN, n, seed)
    eta = s.Q - s.X[:, 0]
    m0 = P.bfun(eta) + s.X @ np.array(M.DESIGN.beta1) + s.D * (P.alpha(eta) + s.X @ P.BETA2)
    RY = s.Y - m0
    I0 = (eta >= P.L0) & (eta <= P.U0)
    psiU = -np.where(I0, P.Fs(eta), 0.0)
    psiO = -M.r_out(s.D, eta, s.X) * RY
    A, w2 = P.psiT_parts(s.X[:, 0]); psiT = A - w2 * s.X[:, 1]
    psiG = (_xt(s.X) @ M.ABAR) * eta
    S = np.column_stack((psiU, psiO, psiT, psiG))
    tot = S.sum(1)
    means.append(tot.mean()); within.append(tot.var()); parts_means.append(S.mean(0))
means = np.array(means); parts_means = np.array(parts_means)
print(f"n={n} R={R} off={off}: n*Var_across(mean)={n*means.var(ddof=1):.5f}  mean within var={np.mean(within):.5f}  sigma2={P.sigma2:.5f}")
print("  n*Var_across per role:", np.round(n * parts_means.var(0, ddof=1), 5), " theory:", np.round([P.varU, P.varO, P.varT, P.varG], 5))
print("  implied nVar(lin_avg) =", n * means.var(ddof=1) / P.H**2, " s.e. approx", P.V * np.sqrt(2 / (R - 1)))
