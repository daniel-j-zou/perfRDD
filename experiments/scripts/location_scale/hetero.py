"""Heteroskedastic eta at taxi strength: where the paper's (additive) estimator converges.

Design: common.py with the scale multiplied by k,
    s(X) = exp(k (0.4 X1 + 0.2 X2)),   Q = T + s(X) zeta,   zeta ~ N(0,1) indep. of X,
so eta = Q - T = s(X) zeta has E(eta | X) = 0 but a spread that grows with X1.
k = 1 is the original design of common.py. k is calibrated so that the sd of eta
across X1 deciles varies by a given factor (max/min decile sd), the taxi diagnostic.

Targets (all maximize E[(W - c) 1(Q > phi) I] over Phi = [-0.2, 2.2], W = alpha(zeta) + X'beta2):
  phi_ls  : I = 1(l0 <= zeta <= u0), [l0, u0] = eps / 1-eps quantiles of (phi0 - T)/s
            (the location-scale target of the 2026-10-21 log entry; the "true supported optimum");
  phi_eta : I = 1(l <= eta <= u),    [l, u]  = phi0 - Q_{1-eps}(T), phi0 - Q_eps(T)
            (the paper's window on eta taken literally, with the true effect);
  lim_add : large-n limit of the additive (paper's) estimator = argmax of
            int_l^u {a~(e) Gbar(phi - e) + b~' H_X(phi - e)} f_eta(e) de,
            (a~, b~) the population projection of the additive spline fit (by quadrature);
  lim_ls  : large-n limit of the location-scale estimator (phi_ls plus the shift from
            constant extrapolation of alpha-hat outside J; as in target.py).
Design b (`extra`, population only): the same s(X), but b and alpha act on eta itself,
  W = alpha(eta) + X'beta2, so the paper's outcome model holds and only X indep. eta fails;
  the target uses the paper's window on eta, and the additive limit is computed as above.

usage (run from experiments/runs/location_scale_hetero):
  python hetero.py calib 1.48 2.30                 -> calib.json
  python hetero.py targets k[,k...]                -> targets_k<k>.json
  python hetero.py extra k[,k...]                  -> extra_k<k>.json (regret; design b)
  python hetero.py mc out.jsonl k:n:reps[:start[:design]] ...  (2 worker processes;
                                                   design b runs the additive estimator only)
  python hetero.py bign k n seed                   -> bign_k<k>_n<n>_s<seed>.json
  python hetero.py summary out.jsonl [...]
"""
import common as C
import numpy as np, json, sys, time, glob
from scipy import stats, optimize
from multiprocessing import Pool

D_BASE = np.array([0.8, 0.4])          # common.DELTA: s = exp(0.5 (d1 X1 + d2 X2))
N_TAXI = 633_616


def set_k(k):
    C.DELTA = float(k) * D_BASE


def ktag(k):
    return f"{float(k):.6f}"


# ------------------------------------------------------------------ calibration
def decile_sd(k):
    """sd of eta within X1 deciles, relative to the smallest (exact: E(eta^2|X) = s^2)."""
    a1 = 0.5 * D_BASE[0] * k
    q = stats.norm.ppf(np.linspace(0, 1, 11))
    m = np.diff(stats.norm.cdf(q - 2 * a1))          # prop. to E(e^{2 a1 X1} | decile)
    sd = np.sqrt(m)
    return sd / sd.min()


def r2_eta2(k):
    """R^2 of eta^2 on any rich function of X: Var(s^2) / Var(eta^2), zeta normal."""
    a1, a2 = 0.5 * D_BASE * k
    Es2 = np.exp(2 * a1 ** 2) * np.cosh(a2)
    Es4 = np.exp(8 * a1 ** 2) * np.cosh(2 * a2)
    return (Es4 - Es2 ** 2) / (3 * Es4 - Es2 ** 2)


def calib(ratios):
    out = {}
    for r in ratios:
        k = optimize.brentq(lambda k: decile_sd(k).max() - r, 1e-6, 3.0, xtol=1e-12)
        out[f"{r}"] = dict(k=k, coef_X1=0.4 * k, coef_X2=0.2 * k, ratio=decile_sd(k).max(),
                           ratio_wo_bottom=float(decile_sd(k)[-1] / decile_sd(k)[1]),
                           r2_eta2=r2_eta2(k), decile_sd=decile_sd(k).tolist())
    out["1.0(original)"] = dict(k=1.0, ratio=decile_sd(1.0).max(),
                                ratio_wo_bottom=float(decile_sd(1.0)[-1] / decile_sd(1.0)[1]),
                                r2_eta2=r2_eta2(1.0))
    # empirical check of the diagnostics with OLS residuals (n = 1e6)
    for key in list(out):
        k = out[key]["k"]
        set_k(k)
        rng = np.random.default_rng([7, int(k * 1e6)])
        d = C.simulate(1_000_000, rng)
        X = d["X"]
        Z = np.column_stack([np.ones(len(X)), X])
        eh = d["Q"] - Z @ np.linalg.lstsq(Z, d["Q"], rcond=None)[0]
        dec = np.searchsorted(np.quantile(X[:, 0], np.linspace(0, 1, 11)[1:-1]), X[:, 0])
        sds = np.array([eh[dec == j].std() for j in range(10)])
        # rich: 10 X1 deciles x X2
        cell = dec * 2 + (X[:, 1] > 0)
        cm = np.bincount(cell, eh ** 2) / np.bincount(cell)
        e2 = eh ** 2
        out[key]["emp_ratio"] = float(sds.max() / sds.min())
        out[key]["emp_r2_eta2_cells"] = float(1 - np.mean((e2 - cm[cell]) ** 2) / e2.var())
        out[key]["emp_r2_eta2_linear"] = float(
            1 - np.mean((e2 - Z @ np.linalg.lstsq(Z, e2, rcond=None)[0]) ** 2) / e2.var())
        print(key, out[key], flush=True)
    json.dump(out, open("calib.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ population objects
def R_cdf(r):
    """P((phi0 - T)/s <= r) for the current C.DELTA (robust brackets for small k)."""
    tot = 0.0
    a = 0.5 * C.DELTA[0]
    for x2 in (-0.5, 0.5):
        g = lambda x1: (C.PHI0 - C.T_true(x1, x2)) / C.s_true(x1, x2) - r
        c = (C.PHI0 - C.GAMMA[0] - C.GAMMA[2] * x2) / C.GAMMA[1]
        x1m = c + 1.0 / a if a > 0 else np.inf
        top = min(x1m, 40.0)
        if g(top) >= 0:
            continue
        lo = optimize.brentq(g, -60.0, top, xtol=1e-13) if g(-60.0) > 0 else -np.inf
        hi = np.inf
        if r < 0 and x1m < 40.0 and g(40.0) > 0:
            hi = optimize.brentq(g, x1m, 40.0, xtol=1e-13)
        tot += 0.5 * (stats.norm.cdf(hi) - stats.norm.cdf(lo))
    return tot


def window_zeta(eps=C.EPS):
    l0 = optimize.brentq(lambda r: R_cdf(r) - eps, -10.0, 60.0, xtol=1e-12)
    u0 = optimize.brentq(lambda r: R_cdf(r) - (1 - eps), -10.0, 200.0, xtol=1e-12)
    return l0, u0


def G_T(t):
    return 0.5 * (stats.norm.cdf(t + 0.5 * C.GAMMA[2]) + stats.norm.cdf(t - 0.5 * C.GAMMA[2]))


def window_eta(eps=C.EPS):
    q = lambda p: optimize.brentq(lambda t: G_T(t) - p, -20, 20, xtol=1e-13)
    return C.PHI0 - q(1 - eps), C.PHI0 - q(eps)


def H_X(t):
    """E[X 1(T > t)] for T = X1 + 0.5 X2, X2 = +-1/2 (columns: X1, X2)."""
    h1 = sum(0.5 * stats.norm.pdf(t - C.GAMMA[2] * x2) for x2 in (-0.5, 0.5))
    h2 = sum(0.5 * x2 * stats.norm.sf(t - C.GAMMA[2] * x2) for x2 in (-0.5, 0.5))
    return h1, h2


def argmax_fn(U, grid=241):
    ph = np.linspace(C.PHI_LO, C.PHI_HI, grid)
    v = np.array([U(p) for p in ph])
    k = int(np.argmax(v))
    a, b = ph[max(k - 1, 0)], ph[min(k + 1, grid - 1)]
    res = optimize.minimize_scalar(lambda p: -U(p), bounds=(a, b), method="bounded",
                                   options=dict(xatol=1e-9))
    return float(res.x), float(-res.fun), ph, v


def U_eta_true(phi, xq, xc, l, u):
    """E[(alpha(zeta) + X'b2) 1(Q > phi) 1(l <= eta <= u)], eta = s zeta."""
    hz = u / xq.s
    lo = np.minimum(np.maximum((phi - xq.T) / xq.s, l / xq.s), hz)
    xb = C.BETA2[0] * xq.X1 + C.BETA2[1] * xq.X2
    return float(np.sum(xq.w * ((xc.Ai(hz) - xc.Ai(lo)) + xb * (xc.Fi(hz) - xc.Fi(lo)))))


def f_eta(e, m=160):
    """Marginal density of eta = s(X) zeta by Gauss-Hermite in X1 and the two X2 values."""
    x, w = np.polynomial.hermite_e.hermegauss(m)
    w = w / w.sum()
    out = np.zeros_like(e)
    for x2 in (-0.5, 0.5):
        s = C.s_true(x, x2)
        out += 0.5 * np.sum(w[None, :] * stats.norm.pdf(e[:, None] / s[None, :]) / s[None, :], 1)
    return out


def _wquant(x, w, p):
    o = np.argsort(x)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(p, cw, x[o])


def additive_projection(k, K=20, m1=100, mz=4001, zlim=8.0, design="a"):
    """Population limit of fit_outcome on (eta, X, D): weighted least squares of
    E(Y | X, zeta) on arm-wise cubic splines in eta (K interior knots at arm quantiles,
    0.25% arm tails trimmed) + X + D X, by quadrature over X1 (Gauss-Hermite), X2, zeta."""
    from scipy.interpolate import BSpline
    set_k(k)
    x, w = np.polynomial.hermite_e.hermegauss(m1)
    w = w / w.sum()
    z = np.linspace(-zlim, zlim, mz)
    wz = stats.norm.pdf(z) * (z[1] - z[0]); wz[[0, -1]] /= 2
    X1 = np.tile(np.repeat(x, mz), 2)
    X2 = np.repeat([-0.5, 0.5], m1 * mz)
    Z = np.tile(z, 2 * m1)
    W = 0.5 * np.tile(np.repeat(w, mz), 2) * np.tile(wz, 2 * m1)
    eta = C.s_true(X1, X2) * Z
    D = (C.T_true(X1, X2) + eta > C.PHI0).astype(float)
    Xm = np.column_stack([X1, X2])
    arg = Z if design == "a" else eta          # design b: b and alpha act on eta itself
    m = C.b_fn(arg) + D * C.alpha_fn(arg) + Xm @ C.BETA1 + D * (Xm @ C.BETA2)
    arms, rows, blocks = {}, [], []
    for d in (0, 1):
        ia = np.flatnonzero(D == d)
        lo, hi = _wquant(eta[ia], W[ia], [C.TRIM_J, 1 - C.TRIM_J])
        ia = ia[(eta[ia] >= lo) & (eta[ia] <= hi)]
        inner = _wquant(eta[ia], W[ia], np.linspace(0, 1, K + 2)[1:-1])
        t = np.r_[[lo] * 4, inner, [hi] * 4]
        arms[d] = (t, lo, hi)
        rows.append(ia)
        blocks.append(BSpline.design_matrix(np.clip(eta[ia], lo, hi), t, 3).toarray())
    p0, p1 = blocks[0].shape[1], blocks[1].shape[1]
    idx = np.r_[rows[0], rows[1]]
    A = np.zeros((len(idx), p0 + p1 + 4))
    A[:len(rows[0]), :p0] = blocks[0]
    A[len(rows[0]):, p0:p0 + p1] = blocks[1]
    A[:, p0 + p1:p0 + p1 + 2] = Xm[idx]
    A[:, p0 + p1 + 2:] = D[idx, None] * Xm[idx]
    sw = np.sqrt(W[idx])
    coef = np.linalg.lstsq(A * sw[:, None], m[idx] * sw, rcond=None)[0]
    c0, c1, b2 = coef[:p0], coef[p0:p0 + p1], coef[p0 + p1 + 2:]

    def ah(e):
        t1, lo1, hi1 = arms[1]
        t0, lo0, hi0 = arms[0]
        return (BSpline.design_matrix(np.clip(e, lo1, hi1), t1, 3).toarray() @ c1
                - BSpline.design_matrix(np.clip(e, lo0, hi0), t0, 3).toarray() @ c0)
    return ah, b2, arms


def additive_limit(k, K=20, design="a", proj=None):
    """Large-n limit of the additive estimator: population projection, then the
    product-measure criterion over the eta window."""
    ah, b2, arms = proj if proj is not None else additive_projection(k, K=K, design=design)
    l, u = window_eta()
    e = np.linspace(l, u, 20001)
    fe, ae = f_eta(e), ah(e)
    h = e[1] - e[0]
    wts = np.full(len(e), h); wts[[0, -1]] = h / 2

    def U(phi):
        t = phi - e
        h1, h2 = H_X(t)
        return float(np.sum(wts * fe * (ae * (1 - G_T(t)) + b2[0] * h1 + b2[1] * h2)))

    ps, Us, ph, v = argmax_fn(U)
    # local maxima of the criterion on the grid (to flag near-ties / boundary maxima)
    loc = [float(ph[i]) for i in range(len(ph))
           if (i == 0 or v[i] >= v[i - 1]) and (i == len(ph) - 1 or v[i] >= v[i + 1])]
    return dict(phi=ps, U=Us, beta2_tilde=b2.tolist(), J0=[arms[0][1], arms[0][2]],
                J1=[arms[1][1], arms[1][2]], local_max_grid=loc,
                U_at_bounds=[float(v[0]), float(v[-1])], grid=ph.tolist(), curve=v.tolist(),
                alpha_tilde_grid=dict(e=e[::500].tolist(), a=ae[::500].tolist()))


def ls_clip_limit(xq, l0, u0):
    """Location-scale estimator limit: alpha-hat clipped to the population J of each arm."""
    Rv = (C.PHI0 - xq.T) / xq.s
    o = np.argsort(Rv)
    Rs, Wc = Rv[o], np.cumsum(xq.w[o])
    PRlt = lambda e: np.interp(e, Rs, Wc, left=0.0, right=1.0)
    eg = np.linspace(-12, 12, 480001)
    f = stats.norm.pdf(eg)
    d1 = f * PRlt(eg); d0 = f - d1
    c1 = np.cumsum(d1); c1 /= c1[-1]
    c0 = np.cumsum(d0); c0 /= c0[-1]
    lo1, hi1 = np.interp([C.TRIM_J, 1 - C.TRIM_J], c1, eg)
    lo0, hi0 = np.interp([C.TRIM_J, 1 - C.TRIM_J], c0, eg)
    alim = lambda e: (C.b_fn(np.clip(e, lo1, hi1)) + C.alpha_fn(np.clip(e, lo1, hi1))
                      - C.b_fn(np.clip(e, lo0, hi0)))
    xcl = C.XiCum("normal", alim)
    pl, _, _, _ = C.argmax_pop(xq, xcl, l0, u0)
    return pl


def targets(k):
    t0 = time.time()
    set_k(k)
    xq = C.XQuad()
    xc = C.XiCum("normal", C.alpha_fn)
    l0, u0 = window_zeta()
    phi_ls, U_ls, _, _ = C.argmax_pop(xq, xc, l0, u0)
    l, u = window_eta()
    phi_eta, U_eta, ph, v = argmax_fn(lambda p: U_eta_true(p, xq, xc, l, u))
    h = 1e-3
    curv_ls = (C.U_pop(phi_ls + h, xq, xc, l0, u0) - 2 * U_ls
               + C.U_pop(phi_ls - h, xq, xc, l0, u0)) / h ** 2
    out = dict(k=k, coef=[0.4 * k, 0.2 * k], ratio=float(decile_sd(k).max()), r2_eta2=r2_eta2(k),
               window_zeta=[l0, u0], window_eta=[l, u], phi_ls=phi_ls, U2_ls=curv_ls,
               phi_eta=phi_eta, lim_ls=ls_clip_limit(xq, l0, u0))
    print(ktag(k), {a: b for a, b in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
    adds = [additive_limit(k, K=KK) for KK in (20, 12)]
    out["lim_add"] = adds[0]["phi"]
    out["lim_add_seeds"] = [a["phi"] for a in adds]        # K = 20, 12 interior knots
    out["add_detail"] = adds[0]
    out["runtime_s"] = time.time() - t0
    print(ktag(k), "lim_add", out["lim_add_seeds"], "local max", adds[0]["local_max_grid"],
          "U(bounds)", adds[0]["U_at_bounds"], "U", adds[0]["U"], flush=True)
    json.dump(out, open(f"targets_k{ktag(k)}.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ regret and design b
def U_joint_factory(l, u, afun=None, b2=None, me=2001, mx=6001):
    """E[(afun(eta) + X'b2) 1(T + eta > phi) 1(l <= eta <= u)] under the joint law of
    (eta, X), eta = s(X) zeta. With afun = alpha, b2 = beta2 it is the true utility of
    design b; with the additive projection (a~, b~) it is the limit of the own-score
    check (evaluation over the observed pairs, no independence used)."""
    afun = C.alpha_fn if afun is None else afun
    b2 = C.BETA2 if b2 is None else np.asarray(b2)
    e = np.linspace(l, u, me)
    we = np.full(me, e[1] - e[0]); we[[0, -1]] /= 2
    x1 = np.linspace(-9, 9, mx)
    hx = x1[1] - x1[0]
    S = {}
    for x2 in (-0.5, 0.5):
        sx = C.s_true(x1, x2)
        a = (stats.norm.pdf(x1)[None, :] * stats.norm.pdf(e[:, None] / sx[None, :]) / sx[None, :]
             * (afun(e)[:, None] + b2[0] * x1[None, :] + b2[1] * x2) * hx)
        S[x2] = np.cumsum(a[:, ::-1], axis=1)[:, ::-1] - a / 2  # trapezoid: int_{x1_j}^inf
    rows = np.arange(me)

    def U(phi):
        tot = 0.0
        for x2 in (-0.5, 0.5):
            thr = phi - e - C.GAMMA[2] * x2                      # treated iff X1 > thr
            pos = np.clip((thr - x1[0]) / hx, 0, mx - 1)
            j = np.minimum(pos.astype(int), mx - 2)
            fr = pos - j
            val = (1 - fr) * S[x2][rows, j] + fr * S[x2][rows, j + 1]
            tot += 0.5 * np.sum(we * val)
        return float(tot)
    return U


def extra(k):
    """Regret of the additive limit in design a; split of its gap into the outcome-model
    and evaluation (independence) channels via the own-score check; design b."""
    t = json.load(open(f"targets_k{ktag(k)}.json"))
    set_k(k)
    xq = C.XQuad()
    xc = C.XiCum("normal", C.alpha_fn)
    l0, u0 = t["window_zeta"]
    Ua = lambda p: C.U_pop(p, xq, xc, l0, u0)
    ga = Ua(t["phi_ls"]) - Ua(C.PHI0)
    out = dict(k=k, a_gain_phi0=ga, a_loss_lim_add=Ua(t["phi_ls"]) - Ua(t["lim_add"]),
               a_loss_lim_ls=Ua(t["phi_ls"]) - Ua(t["lim_ls"]))
    out["a_loss_share"] = out["a_loss_lim_add"] / ga
    l, u = window_eta()
    pa = additive_projection(k, K=20, design="a")
    out["a_lim_own"] = argmax_fn(U_joint_factory(l, u, pa[0], pa[1]))[0]
    Ub = U_joint_factory(l, u)
    phi_b, Ub_star, _, _ = argmax_fn(Ub)
    pb = additive_projection(k, K=20, design="b")
    out["b_lim_own"] = argmax_fn(U_joint_factory(l, u, pb[0], pb[1]))[0]
    out["b_beta2_tilde"] = pb[1].tolist()
    lb = [additive_limit(k, design="b", proj=pb)["phi"], additive_limit(k, K=12, design="b")["phi"]]
    gb = Ub(phi_b) - Ub(C.PHI0)
    out.update(b_phi=phi_b, b_lim_add=lb[0], b_lim_add_K12=lb[1], b_gain_phi0=gb,
               b_loss_lim_add=Ub(phi_b) - Ub(lb[0]))
    out["b_loss_share"] = out["b_loss_lim_add"] / gb
    print(ktag(k), {a: (round(b, 5) if isinstance(b, float) else b) for a, b in out.items()},
          flush=True)
    json.dump(out, open(f"extra_k{ktag(k)}.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ Monte Carlo
def simulate_b(n, rng):
    """Design b: as C.simulate, but b and alpha act on eta = s(X) zeta."""
    X1 = rng.standard_normal(n)
    X2 = rng.integers(0, 2, n) - 0.5
    eta = C.s_true(X1, X2) * rng.standard_normal(n)
    Q = C.T_true(X1, X2) + eta
    D = (Q > C.PHI0).astype(float)
    X = np.column_stack([X1, X2])
    Y = (C.b_fn(eta) + D * C.alpha_fn(eta) + X @ C.BETA1 + D * (X @ C.BETA2)
         + C.SIG_Y * rng.standard_normal(n))
    return dict(X=X, Q=Q, D=D, Y=Y)


def one(task):
    k, n, rep = task[:3]
    design = task[3] if len(task) > 3 else "a"
    set_k(k)
    t0 = time.time()
    if design == "a":
        rng = np.random.default_rng([20261109, int(round(k * 1e6)), n, rep])
        d = C.simulate(n, rng)
    else:
        rng = np.random.default_rng([20261110, int(round(k * 1e6)), n, rep])
        d = simulate_b(n, rng)
    d["folds"] = rng.permutation(n) % 4
    ad, ad_rot = C.four_role(d, scale=False)
    rec = dict(k=k, n=n, rep=rep, design=design, add=ad, add_rot=ad_rot.tolist())
    if design == "a":
        ls, ls_rot = C.four_role(d, scale=True)
        rec.update(ls=ls, ls_rot=ls_rot.tolist())
    rec["secs"] = time.time() - t0
    return rec


def mc(outp, specs):
    tasks = []
    for spec in specs:
        p = spec.split(":")
        k, n, reps = float(p[0]), int(p[1]), int(p[2])
        start = int(p[3]) if len(p) > 3 else 0
        design = p[4] if len(p) > 4 else "a"
        tasks += [(k, n, r, design) for r in range(start, start + reps)]
    tasks.sort(key=lambda t: -t[1])
    t0 = time.time()
    with Pool(2) as pool, open(outp, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(tasks) - 1:
                print(f"{i+1}/{len(tasks)} k={rec['k']:.4f} n={rec['n']} rep={rec['rep']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


def bign(k, n, seed):
    """Full pipeline at large n; stores the rotation-averaged U-hat curves of both estimators."""
    set_k(k)
    grid = np.linspace(C.PHI_LO, C.PHI_HI, 121)
    rng = np.random.default_rng([99, int(k * 1e6), n, seed])
    d = C.simulate(n, rng)
    X, Q, D, Y = d["X"], d["Q"], d["D"], d["Y"]
    folds = rng.permutation(n) % 4
    rec = dict(k=k, n=n, seed=seed, grid=grid.tolist())
    for scale, nm in [(False, "add"), (True, "ls")]:
        t0 = time.time()
        ests, curves = [], []
        for r in range(4):
            fg, fo, fi, fu = [(folds == ((r + j) % 4)) for j in range(4)]
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
        rec[nm] = dict(phi=float(np.mean(ests)), rot=ests, curves=curves, secs=time.time() - t0)
        print(nm, rec[nm]["phi"], np.round(ests, 4), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(rec, open(f"bign_k{ktag(k)}_n{n}_s{seed}.json", "w"), default=float)


# ------------------------------------------------------------------ summary
def summary(files):
    """Bias, sd and RMSE against the true target; design a: phi_ls (b_eta: against phi_eta),
    design b: phi_b. b_lim = mean - the estimator's large-n limit."""
    from collections import defaultdict
    recs = [json.loads(l) for f in files for l in open(f)]
    tg, ex = {}, {}
    for f in glob.glob("targets_k*.json"):
        t = json.load(open(f)); tg[ktag(t["k"])] = t
    for f in glob.glob("extra_k*.json"):
        x = json.load(open(f)); ex[ktag(x["k"])] = x
    g = defaultdict(list)
    for r in recs:
        g[(r.get("design", "a"), ktag(r["k"]), r["n"])].append(r)
    rows = []
    print(f"{'des':>3s} {'k':>7s} {'ratio':>5s} {'est':>3s} {'n':>6s} {'R':>4s} {'mean':>7s} "
          f"{'bias':>7s} {'mcse':>6s} {'sd':>6s} {'rn*sd':>6s} {'RMSE':>6s} {'b_eta':>7s} "
          f"{'b_lim':>7s} {'bnd':>5s}")
    for key in sorted(g, key=lambda q: (q[0], float(q[1]), q[2])):
        des, kk, n = key
        t = tg[kk]
        for est in ("add", "ls"):
            if est not in g[key][0]:
                continue
            x = np.array([r[est] for r in g[key]])
            rot = np.array([r[est + "_rot"] for r in g[key]])
            R, sd = len(x), x.std(ddof=1)
            if des == "a":
                tgt, alt = t["phi_ls"], t["phi_eta"]
                lim = t["lim_add"] if est == "add" else t["lim_ls"]
            else:
                tgt, alt, lim = ex[kk]["b_phi"], np.nan, ex[kk]["b_lim_add"]
            row = dict(design=des, k=float(kk), ratio=t["ratio"], est=est, n=n, R=R,
                       mean=x.mean(), target=tgt, limit=lim, bias=x.mean() - tgt,
                       mcse=sd / np.sqrt(R), sd=sd, rn_sd=np.sqrt(n) * sd,
                       rmse=np.sqrt(np.mean((x - tgt) ** 2)), bias_eta=x.mean() - alt,
                       bias_lim=x.mean() - lim,
                       bnd=float(np.mean((np.abs(rot - C.PHI_LO) < 1e-9) |
                                         (np.abs(rot - C.PHI_HI) < 1e-9))))
            rows.append(row)
            print(f"{des:>3s} {float(kk):7.4f} {t['ratio']:5.2f} {est:>3s} {n:6d} {R:4d} "
                  f"{row['mean']:7.4f} {row['bias']:7.4f} {row['mcse']:6.4f} {sd:6.4f} "
                  f"{row['rn_sd']:6.2f} {row['rmse']:6.4f} {row['bias_eta']:7.4f} "
                  f"{row['bias_lim']:7.4f} {row['bnd']:5.3f}")
    print("\nLimits; gap = additive limit - true target, in units of the additive sd at "
          f"n = {N_TAXI:,} (sqrt(n) sd from the largest n, rescaled)")
    for des in ("a", "b"):
        for kk in sorted(tg, key=float):
            t, x = tg[kk], ex.get(kk)
            if des == "b" and x is None:
                continue
            tgt = t["phi_ls"] if des == "a" else x["b_phi"]
            lim = t["lim_add"] if des == "a" else x["b_lim_add"]
            big = [r for r in rows if r["design"] == des and ktag(r["k"]) == kk and r["est"] == "add"]
            sd_t = max(big, key=lambda r: r["n"])["rn_sd"] / np.sqrt(N_TAXI) if big else np.nan
            extra_txt = ""
            if des == "a":
                extra_txt = (f" phi_eta={t['phi_eta']:.4f} lim_ls={t['lim_ls']:.4f} "
                             f"own-check={x['a_lim_own']:.4f}" if x else "")
                ls_ = f" add K=12: {t['lim_add_seeds'][1]:.4f}"
            else:
                extra_txt = f" own-check={x['b_lim_own']:.4f}"
                ls_ = f" add K=12: {x['b_lim_add_K12']:.4f}"
            share = (x["a_loss_share"] if des == "a" else x["b_loss_share"]) if x else np.nan
            print(f"{des} k={float(kk):.4f} ratio={t['ratio']:.2f} r2(eta^2)={t['r2_eta2']:.3f} "
                  f"target={tgt:.4f} lim_add={lim:.4f} gap={lim - tgt:+.4f} "
                  f"sd_add(taxi n)={sd_t:.4f} gap/sd={(lim - tgt) / sd_t:+.2f} "
                  f"loss share={share:.3f}{extra_txt}{ls_}")
    json.dump(dict(rows=rows, targets={k: {a: b for a, b in v.items() if a != "add_detail"}
                                       for k, v in tg.items()}, extra=ex),
              open("summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calib":
        calib([float(a) for a in sys.argv[2:]])
    elif cmd == "targets":
        for k in sys.argv[2].split(","):
            targets(float(k))
    elif cmd == "extra":
        for k in sys.argv[2].split(","):
            extra(float(k))
    elif cmd == "mc":
        mc(sys.argv[2], sys.argv[3:])
    elif cmd == "bign":
        bign(float(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4]))
    elif cmd == "summary":
        summary(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
