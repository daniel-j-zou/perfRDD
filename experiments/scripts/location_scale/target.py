"""Task (a): population target phi*, window, identity check, clip-limit, non-normal targets."""
import common as C
import numpy as np
import json, time, sys

t00 = time.time()
out = {}
l0, u0 = C.window_pop()
out["window"] = [l0, u0]
xq = C.XQuad()
res = {}
for dist in ["normal", "t5", "logistic"]:
    fz = C.xi_frozen(dist)
    xc = C.XiCum(dist, C.alpha_fn)
    ps, Us, ph, U = C.argmax_pop(xq, xc, l0, u0)
    h = 1e-3
    curv = (C.U_pop(ps + h, xq, xc, l0, u0) - 2 * Us + C.U_pop(ps - h, xq, xc, l0, u0)) / h ** 2
    share = float(np.sum(xq.w * fz.sf((C.PHI0 - xq.T) / xq.s)))
    pwin = float(fz.cdf(u0) - fz.cdf(l0))
    res[dist] = dict(phi_star=ps, U_star=Us, U2=curv, treated_share=share, P_xi_in_window=pwin)
    print(dist, res[dist], flush=True)
out["targets"] = res

# ------------------------------------------------ identity check (normal)
xc = C.XiCum("normal", C.alpha_fn)
grid = np.linspace(C.PHI_LO, C.PHI_HI, 25)
U_semi = np.array([C.U_pop(p, xq, xc, l0, u0) for p in grid])
phi_star = res["normal"]["phi_star"]

# (i) direct joint MC: U(phi) = mean[(alpha+X'b2) 1(Q>phi) 1(window)]
nmc = 2_000_000
dir_U, dir_arg, dir_argq = [], [], []
fine = np.linspace(C.PHI_LO, C.PHI_HI, 4801)
NDIR = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for sd in range(NDIR):
    rng = np.random.default_rng([2026, sd])
    d = C.simulate(nmc, rng)
    xi, X, Q = d["xi"], d["X"], d["Q"]
    w = (C.alpha_fn(xi) + X @ C.BETA2) * ((xi >= l0) & (xi <= u0))
    o = np.argsort(Q)
    Qs, ws = Q[o], w[o]
    SW = np.r_[np.cumsum(ws[::-1])[::-1], 0.0]
    Uf = lambda p: SW[np.searchsorted(Qs, p, side="right")] / nmc
    dir_U.append(Uf(grid))
    vf = Uf(fine)
    dir_arg.append(fine[int(np.argmax(vf))])
    m = np.abs(fine - fine[int(np.argmax(vf))]) <= 0.4      # quadratic smoothing near max
    cf = np.polyfit(fine[m], vf[m], 2)
    dir_argq.append(-cf[1] / (2 * cf[0]))
dir_U = np.array(dir_U)
out["direct_mc"] = dict(n=nmc, seeds=NDIR, argmax_raw=dir_arg, argmax_quad=dir_argq,
                        max_abs_U_diff=float(np.max(np.abs(dir_U.mean(0) - U_semi))),
                        U_mc_se=float(np.max(dir_U.std(0, ddof=1) / np.sqrt(NDIR))))
print("direct", out["direct_mc"], flush=True)

# (ii) iterated form with independent draws of xi and X
nx = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
it_U, it_arg = [], []
NIT = int(sys.argv[3]) if len(sys.argv) > 3 else 2
for sd in range(NIT):
    rng = np.random.default_rng([2027, sd])
    xi = rng.standard_normal(nx)
    X1 = rng.standard_normal(nx)
    X2 = rng.integers(0, 2, nx) - 0.5
    T, s = C.T_true(X1, X2), C.s_true(X1, X2)
    keep = (xi >= l0) & (xi <= u0)
    e = xi[keep]
    Uh = C.make_Uhat(e, C.alpha_fn(e), T, s, C.BETA2[0] * X1 + C.BETA2[1] * X2)
    scale = keep.sum() / nx          # make_Uhat divides by #kept; convert to 1/n_xi
    it_U.append(Uh(grid) * 1.0 / len(e) * scale)
    it_arg.append(C.argmax_step(Uh))
it_U = np.array(it_U)
out["iterated_mc"] = dict(n_xi=nx, n_X=nx, seeds=NIT, argmax=it_arg,
                          max_abs_U_diff=float(np.max(np.abs(it_U.mean(0) - U_semi))))
print("iterated", out["iterated_mc"], flush=True)
out["U_table"] = dict(phi=grid.tolist(), semi=U_semi.tolist(), direct=dir_U.mean(0).tolist(),
                      iterated=it_U.mean(0).tolist())

# ------------------------------------------------ limit of LS estimator with clipped alpha-hat
# population J bounds per arm: control density f(e) P(R>=e), treated f(e) P(R<e)
Rv = (C.PHI0 - xq.T) / xq.s
o = np.argsort(Rv)
Rs, Wc = Rv[o], np.cumsum(xq.w[o])
PRlt = lambda e: np.interp(e, Rs, Wc, left=0.0, right=1.0)
eg = np.linspace(-12, 12, 480001)
f = C.xi_frozen("normal").pdf(eg)
d1 = f * PRlt(eg); d0 = f - d1
c1 = np.cumsum(d1); c1 /= c1[-1]
c0 = np.cumsum(d0); c0 /= c0[-1]
lo1, hi1 = np.interp([C.TRIM_J, 1 - C.TRIM_J], c1, eg)
lo0, hi0 = np.interp([C.TRIM_J, 1 - C.TRIM_J], c0, eg)
print("J0", lo0, hi0, "J1", lo1, hi1)
alim = lambda e: (C.b_fn(np.clip(e, lo1, hi1)) + C.alpha_fn(np.clip(e, lo1, hi1))
                  - C.b_fn(np.clip(e, lo0, hi0)))
xcl = C.XiCum("normal", alim)
pl, _, _, _ = C.argmax_pop(xq, xcl, l0, u0)
fz = C.xi_frozen("normal")
out["clip_limit"] = dict(J0=[lo0, hi0], J1=[lo1, hi1], phi_lim=pl, shift=pl - phi_star,
                         P_window_above_hi0=float(fz.cdf(u0) - fz.cdf(hi0)),
                         P_window_below_lo1=float(max(0, fz.cdf(lo1) - fz.cdf(l0))))
print("clip", out["clip_limit"], flush=True)
out["runtime_s"] = time.time() - t00
json.dump(out, open("target_results.json", "w"), indent=1, default=float)
print("done", time.time() - t00)
