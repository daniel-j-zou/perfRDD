"""Differing slopes: numerical checks of the identification and curvature results.

Checks, by quadrature, the identities and conditions written up in the
"Identification and curvature" section of ``this_week.tex`` (2026-10-13):

1. Marginal-effect representation.  U_eps'(phi) = -p_eps(phi) tau_eps(phi),
   where p_eps is the density of Q on the supported population and
   tau_eps(phi) = E[W - c | Q = phi, l0 <= eta <= u0].
2. Derivative of the marginal effect, two exact forms:
     A: tau' = Cov(a(E), s_T(phi-E)) + E[lam'(phi-E)] + Cov(lam(phi-E), s_T(phi-E))
     B: tau' = Cov(a(E), s_T(phi-E)) + Cov(lam(phi-E), s_eta(E)) + B_eps(phi)
   with E ~ h_phi(e) proportional to g(phi-e) f_eta(e) on [l0, u0],
   lam(t) = beta2' E(X | T = t), s_T = (log g)', s_eta = (log f_eta)' and the
   trim-boundary term B_eps.  Both are compared with a finite difference.
3. Curvature at the optimum: U_eps''(phi*) = -p_eps(phi*) tau_eps'(phi*).
4. Aligned-heterogeneity bounds a(l0)-c+lam(phi-u0) <= tau <= a(u0)-c+lam(phi-l0)
   and the endpoint condition on Phi.
5. Gaussian-index formula tau' = s_beta + Cov(a(E) - s_beta E, E) / sigma_T^2,
   and the opposing-heterogeneity example where tau_eps is not monotone.
6. Remark on phi*_eps versus the untrimmed phi*: sign and bound.
7. d phi*_eps / d eps: implicit-function formula against a finite difference.
8. Why log-concavity of f_eta is needed: a bimodal f_eta with constant a and
   lam(t) = t, where tau_eps is not monotone although (A1) for g, (A2), (A3b)
   and (A4) hold.
9. The half-space constant lambda_J of the identification lemma.
10. The old Gauss-Hermite target: with the hard trim indicator applied at
    240 Gauss-Hermite nodes only 12 nodes fall inside the window, which moves
    the four-role design optimum from -0.1197699 to -0.1200638.

Designs: the four-role differing-slopes check design (Appendix 5 of
``differing_slopes.tex``), the displayed Section 5 design, a non-Gaussian
design with a nonlinear lam, and an opposing-heterogeneity Gaussian design.

Run:
    python -m experiments.scripts.ds_identification_check
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import integrate, optimize, stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_identification_check"

Fn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Design:
    name: str
    f_eta: Fn          # density of eta
    df_eta: Fn         # its derivative
    q_eta: Callable[[float], float]  # quantile function of eta (for untrimmed checks)
    g: Fn              # density of T
    dg: Fn             # its derivative
    q_T: Callable[[float], float]    # quantile function of T
    a: Fn              # treatment effect at X = 0, a(eta)
    lam: Fn            # beta2' E(X | T = t)
    dlam: Fn
    c: float
    phi0: float
    eps: float
    phi_lo: float
    phi_hi: float
    sigma_T2: float | None = None    # set when T is Gaussian and lam is linear
    s_beta: float | None = None

    @property
    def window(self) -> tuple[float, float]:
        return (self.phi0 - self.q_T(1 - self.eps), self.phi0 - self.q_T(self.eps))


# Fixed 400-node Gauss-Legendre rule on the trim window; the integrands are smooth there.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)


def _nodes(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo), 0.5 * (hi - lo) * _GL_W


def tilt(d: Design, phi: float, lo: float | None = None, hi: float | None = None):
    """Nodes, normalized weights of h_phi, and the mass p(phi) on [lo, hi]."""
    if lo is None:
        lo, hi = d.window
    e, w = _nodes(lo, hi)
    dens = d.g(phi - e) * d.f_eta(e)
    p = float(np.sum(w * dens))
    return e, w * dens / p, p


def tau(d: Design, phi: float, lo=None, hi=None) -> float:
    e, h, _ = tilt(d, phi, lo, hi)
    return float(np.sum(h * (d.a(e) - d.c + d.lam(phi - e))))


def p_eps(d: Design, phi: float) -> float:
    return tilt(d, phi)[2]


def tail_H(d: Design, s: float) -> float:
    """H(s) = E[lam(T) 1{T > s}] = beta2' H_X(s)."""
    val, _ = integrate.quad(lambda t: d.lam(np.array([t]))[0] * d.g(np.array([t]))[0], s, np.inf, limit=200)
    return val


def tail_G(d: Design, s: float) -> float:
    val, _ = integrate.quad(lambda t: d.g(np.array([t]))[0], s, np.inf, limit=200)
    return val


def utility(d: Design, phi: float) -> float:
    """U_eps(phi) = E[{(a - c) Gbar(phi - eta) + beta2' H_X(phi - eta)} I0]."""
    lo, hi = d.window
    e, w = _nodes(lo, hi)
    e, w = e[::4], w[::4] * 4  # coarser grid: quad inside; still smooth
    vals = np.array([(d.a(np.array([x]))[0] - d.c) * tail_G(d, phi - x) + tail_H(d, phi - x) for x in e])
    return float(np.sum(w * d.f_eta(e) * vals))


def dtau_forms(d: Design, phi: float) -> dict[str, float]:
    lo, hi = d.window
    e, h, p = tilt(d, phi)
    t = phi - e
    sT = d.dg(t) / d.g(t)
    seta = d.df_eta(e) / d.f_eta(e)
    A = d.a(e)
    L = d.lam(t)

    def cov(x, y):
        return float(np.sum(h * x * y) - np.sum(h * x) * np.sum(h * y))

    c_a = cov(A, sT)
    lam_mean = float(np.sum(h * L))
    form_a = c_a + float(np.sum(h * d.dlam(t))) + cov(L, sT)
    bnd = (d.g(np.array([phi - lo]))[0] * d.f_eta(np.array([lo]))[0] * (d.lam(np.array([phi - lo]))[0] - lam_mean)
           + d.g(np.array([phi - hi]))[0] * d.f_eta(np.array([hi]))[0] * (lam_mean - d.lam(np.array([phi - hi]))[0])) / p
    c_b = cov(L, seta)
    form_b = c_a + c_b + bnd
    out = {"C_a": c_a, "C_b": c_b, "B_eps": float(bnd), "form_A": form_a, "form_B": form_b,
           "var_E": cov(e, e)}
    if d.sigma_T2 is not None:
        out["gaussian_formula"] = d.s_beta + cov(A - d.s_beta * e, e) / d.sigma_T2
    return out


def fd(fun: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    return (fun(x + h) - fun(x - h)) / (2 * h)


def fd2(fun: Callable[[float], float], x: float, h: float = 2e-3) -> float:
    return (fun(x + h) - 2 * fun(x) + fun(x - h)) / h ** 2


def check_design(d: Design) -> dict:
    lo, hi = d.window
    grid = np.linspace(d.phi_lo, d.phi_hi, 121)
    taus = np.array([tau(d, x) for x in grid])
    res: dict = {"name": d.name, "window": [lo, hi], "Phi": [d.phi_lo, d.phi_hi]}

    # 1. U' = -p tau at a few points (finite difference of U).
    pts = np.linspace(d.phi_lo + 0.3, d.phi_hi - 0.3, 5)
    rep = []
    for x in pts:
        dU = fd(lambda y: utility(d, y), x, 1e-3)
        rep.append({"phi": float(x), "dU_fd": dU, "minus_p_tau": -p_eps(d, x) * tau(d, x)})
    res["representation"] = rep
    res["representation_max_abs_err"] = max(abs(r["dU_fd"] - r["minus_p_tau"]) for r in rep)

    # 2. tau' forms against finite differences on the grid.
    errs_a, errs_b, errs_g = [], [], []
    dt_min = np.inf
    for x in grid[1:-1]:
        forms = dtau_forms(d, x)
        num = fd(lambda y: tau(d, y), x)
        errs_a.append(abs(forms["form_A"] - num))
        errs_b.append(abs(forms["form_B"] - num))
        if "gaussian_formula" in forms:
            errs_g.append(abs(forms["gaussian_formula"] - num))
        dt_min = min(dt_min, forms["form_B"])
    res["dtau_form_A_max_err"] = max(errs_a)
    res["dtau_form_B_max_err"] = max(errs_b)
    if errs_g:
        res["dtau_gaussian_formula_max_err"] = max(errs_g)
    res["min_dtau_on_Phi"] = float(dt_min)
    res["tau_monotone_on_grid"] = bool(np.all(np.diff(taus) > 0))
    res["tau_at_ends"] = [float(taus[0]), float(taus[-1])]

    # 3. Zero of tau and curvature.
    sign_change = np.where(np.sign(taus[:-1]) != np.sign(taus[1:]))[0]
    res["n_sign_changes"] = int(len(sign_change))
    if len(sign_change) >= 1:
        k = sign_change[0]
        phi_star = optimize.brentq(lambda y: tau(d, y), grid[k], grid[k + 1], xtol=1e-12)
        forms = dtau_forms(d, phi_star)
        res["phi_star_eps"] = phi_star
        res["dtau_at_star"] = forms
        res["U2_formula"] = -p_eps(d, phi_star) * forms["form_B"]
        res["U2_fd"] = fd2(lambda y: utility(d, y), phi_star)
        res["p_at_star"] = p_eps(d, phi_star)
    return res


def aligned_bounds(d: Design) -> dict:
    lo, hi = d.window
    grid = np.linspace(d.phi_lo, d.phi_hi, 61)
    ok = True
    for x in grid:
        t = tau(d, x)
        lb = d.a(np.array([lo]))[0] - d.c + d.lam(np.array([x - hi]))[0]
        ub = d.a(np.array([hi]))[0] - d.c + d.lam(np.array([x - lo]))[0]
        ok &= (lb - 1e-12 <= t <= ub + 1e-12)
    end_lo = d.a(np.array([hi]))[0] - d.c + d.lam(np.array([d.phi_lo - lo]))[0]
    end_hi = d.a(np.array([lo]))[0] - d.c + d.lam(np.array([d.phi_hi - hi]))[0]
    return {"bounds_hold_on_grid": bool(ok), "endpoint_lo": float(end_lo), "endpoint_hi": float(end_hi),
            "endpoint_condition": bool(end_lo < 0 < end_hi)}


def untrimmed_remark(d: Design, phi_star_eps: float) -> dict:
    """Untrimmed optimum, tau_out, and the bound of the remark."""
    lo_all, hi_all = d.q_eta(1e-10), d.q_eta(1 - 1e-10)
    lo, hi = d.window

    def tau_all(x):
        return tau(d, x, lo_all, hi_all)

    phi_star = optimize.brentq(tau_all, d.phi_lo, d.phi_hi, xtol=1e-12)
    # mass split at phi_star
    _, _, p_in = tilt(d, phi_star, lo, hi)
    _, _, p_all = tilt(d, phi_star, lo_all, hi_all)
    pi = p_in / p_all
    t_all = tau_all(phi_star)
    t_in = tau(d, phi_star)
    t_out = (t_all * p_all - t_in * p_in) / (p_all - p_in)
    pred_t_in = -(1 - pi) / pi * t_out
    seg = np.linspace(min(phi_star, phi_star_eps), max(phi_star, phi_star_eps), 41)
    min_dt = min(dtau_forms(d, x)["form_B"] for x in seg)
    bound = abs(pred_t_in) / min_dt
    return {"phi_star_untrimmed": phi_star, "phi_star_eps": phi_star_eps, "pi": pi, "tau_out": t_out,
            "tau_eps_at_phi_star": t_in, "predicted_tau_eps_at_phi_star": pred_t_in,
            "sign_rule_holds": bool(np.sign(phi_star_eps - phi_star) == np.sign(t_out)),
            "gap": abs(phi_star_eps - phi_star), "bound": bound, "bound_holds": bool(abs(phi_star_eps - phi_star) <= bound + 1e-10)}


def epsilon_sensitivity(d: Design, h: float = 1e-4) -> dict:
    """d phi*_eps / d eps: implicit-function formula against a finite difference."""
    from dataclasses import replace

    def star(eps):
        dd = replace(d, eps=eps)
        grid = np.linspace(d.phi_lo, d.phi_hi, 61)
        t = np.array([tau(dd, x) for x in grid])
        k = np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0][0]
        return optimize.brentq(lambda y: tau(dd, y), grid[k], grid[k + 1], xtol=1e-13)

    phi = star(d.eps)
    lo, hi = d.window
    _, _, p = tilt(d, phi)
    hl = d.g(np.array([phi - lo]))[0] * d.f_eta(np.array([lo]))[0] / p
    hu = d.g(np.array([phi - hi]))[0] * d.f_eta(np.array([hi]))[0] / p
    kl = d.a(np.array([lo]))[0] - d.c + d.lam(np.array([phi - lo]))[0]
    ku = d.a(np.array([hi]))[0] - d.c + d.lam(np.array([phi - hi]))[0]
    gq_hi = d.g(np.array([d.q_T(1 - d.eps)]))[0]
    gq_lo = d.g(np.array([d.q_T(d.eps)]))[0]
    formula = (hl * kl / gq_hi + hu * ku / gq_lo) / dtau_forms(d, phi)["form_B"]
    numeric = (star(d.eps + h) - star(d.eps - h)) / (2 * h)
    return {"phi_star_eps": phi, "kappa_l0": kl, "kappa_u0": ku, "dphi_deps_formula": formula,
            "dphi_deps_fd": numeric}


def gaussian(mu=0.0, sd=1.0):
    f = lambda x: stats.norm.pdf(x, mu, sd)
    df = lambda x: -(x - mu) / sd ** 2 * stats.norm.pdf(x, mu, sd)
    q = lambda p: float(stats.norm.ppf(p, mu, sd))
    return f, df, q


def logistic(scale):
    f = lambda x: stats.logistic.pdf(x, scale=scale)
    df = lambda x: -np.tanh(x / (2 * scale)) / scale * stats.logistic.pdf(x, scale=scale)
    q = lambda p: float(stats.logistic.ppf(p, scale=scale))
    return f, df, q


def designs() -> list[Design]:
    fe, dfe, qe = gaussian()
    fT, dfT, qT = gaussian()
    out = []
    # Four-role DS check design (Appendix 5): T = X1, beta2 = (0.8, 0.25) => lam(t) = 0.8 t.
    out.append(Design("ds_four_role_check", fe, dfe, qe, fT, dfT, qT,
                      a=lambda e: 0.35 + 0.90 * e, lam=lambda t: 0.80 * t, dlam=lambda t: 0.80 + 0 * t,
                      c=0.25, phi0=0.0, eps=0.10, phi_lo=-3.0, phi_hi=3.0, sigma_T2=1.0, s_beta=0.80))
    # Displayed Section 5 design: T = (X1+X2+X3)/sqrt3, beta2 = (0.4,-0.25,0.15) => lam(t) = (0.3/sqrt3) t.
    sb = 0.3 / np.sqrt(3.0)
    out.append(Design("displayed_section5", fe, dfe, qe, fT, dfT, qT,
                      a=lambda e: 2.0 + e, lam=lambda t: sb * t, dlam=lambda t: sb + 0 * t,
                      c=2.25, phi0=0.0, eps=0.10, phi_lo=-3.0, phi_hi=3.0, sigma_T2=1.0, s_beta=sb))
    # Non-Gaussian: logistic T and eta, nonlinear increasing lam, cubic a.
    fe2, dfe2, qe2 = logistic(0.55)
    fT2, dfT2, qT2 = logistic(0.60)
    out.append(Design("logistic_nonlinear", fe2, dfe2, qe2, fT2, dfT2, qT2,
                      a=lambda e: 0.20 + 0.60 * e + 0.20 * e ** 3,
                      lam=lambda t: 0.30 * t + 0.50 * np.tanh(t), dlam=lambda t: 0.30 + 0.50 / np.cosh(t) ** 2,
                      c=0.40, phi0=0.3, eps=0.10, phi_lo=-3.0, phi_hi=3.0))
    # Opposing heterogeneity: lam(t) = -0.3 t, a increasing; untrimmed tau is monotone.
    out.append(Design("opposing_gaussian", fe, dfe, qe, fT, dfT, qT,
                      a=lambda e: 0.35 + 0.90 * e, lam=lambda t: -0.30 * t, dlam=lambda t: -0.30 + 0 * t,
                      c=0.25, phi0=0.0, eps=0.10, phi_lo=-3.0, phi_hi=3.0, sigma_T2=1.0, s_beta=-0.30))
    return out


def opposing_profile(d: Design) -> dict:
    """tau_eps and U_eps over a wide range: tau is eventually decreasing."""
    grid = np.linspace(-12, 12, 97)
    taus = [tau(d, x) for x in grid]
    dts = [dtau_forms(d, x)["form_B"] for x in grid]
    sc = np.where(np.sign(np.array(taus[:-1])) != np.sign(np.array(taus[1:])))[0]
    zeros = [optimize.brentq(lambda y: tau(d, y), grid[k], grid[k + 1]) for k in sc]
    lo, hi = d.window
    treat_all = float(integrate.quad(lambda e: (d.a(np.array([e]))[0] - d.c) * d.f_eta(np.array([e]))[0], lo, hi)[0])
    return {"zeros_of_tau": zeros, "dtau_at_zeros": [dtau_forms(d, z)["form_B"] for z in zeros],
            "U_at_zeros": [utility(d, z) for z in zeros], "U_treat_all_limit": treat_all,
            "U_treat_none_limit": 0.0, "min_dtau": float(min(dts)), "argmin_dtau": float(grid[int(np.argmin(dts))]),
            "dtau_negative_beyond": [float(grid[i]) for i in range(len(grid)) if dts[i] < 0][:1] + [float(grid[i]) for i in range(len(grid)) if dts[i] < 0][-1:]}


def bimodal_counterexample() -> dict:
    """f_eta = 0.5 N(-1.1, 0.15^2) + 0.5 N(1.1, 0.15^2), g = N(0,1), a constant, lam(t) = t."""
    m, s = 1.1, 0.15

    def f(x):
        return 0.5 * (stats.norm.pdf(x, -m, s) + stats.norm.pdf(x, m, s))

    def df(x):
        return 0.5 * (-(x + m) / s ** 2 * stats.norm.pdf(x, -m, s) - (x - m) / s ** 2 * stats.norm.pdf(x, m, s))

    fT, dfT, qT = gaussian()
    d = Design("bimodal_f_eta", f, df, lambda p: float("nan"), fT, dfT, qT,
               a=lambda e: 0.0 * e, lam=lambda t: t, dlam=lambda t: 1.0 + 0 * t,
               c=0.0, phi0=0.0, eps=0.10, phi_lo=-3.0, phi_hi=3.0, sigma_T2=1.0, s_beta=1.0)
    grid = np.linspace(-3, 3, 241)
    t = np.array([tau(d, x) for x in grid])
    ks = np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0]
    zeros = [optimize.brentq(lambda y: tau(d, y), grid[k], grid[k + 1]) for k in ks]
    zeros = sorted({round(z, 10) for z in zeros})
    return {"tau_at_ends": [float(t[0]), float(t[-1])], "dtau_at_0": dtau_forms(d, 0.0),
            "zeros": zeros, "dtau_at_zeros": [dtau_forms(d, z)["form_B"] for z in zeros]}


def halfspace_constant(J_half: float) -> float:
    """lambda_J for X ~ N(0, I_2), T = X1, phi0 = 0 (the four-role check design)."""
    def M(s, side):
        P = stats.norm.sf(s) if side else stats.norm.cdf(s)
        ph = stats.norm.pdf(s)
        m1 = ph if side else -ph
        m2 = (stats.norm.sf(s) + s * ph) if side else (stats.norm.cdf(s) - s * ph)
        return np.array([[P, m1, 0], [m1, m2, 0], [0, 0, P]])
    es = np.linspace(-J_half, J_half, 801)
    return float(min(min(np.linalg.eigvalsh(M(-e, 1))[0], np.linalg.eigvalsh(M(-e, 0))[0]) for e in es))


def gauss_hermite_target() -> dict:
    """The old truth: 240-node Gauss-Hermite with the trim indicator applied at the nodes."""
    from scipy.special import roots_hermitenorm
    x, w = roots_hermitenorm(240)
    w = w / np.sqrt(2 * np.pi)
    z = stats.norm.ppf(0.9)
    keep = (x >= -z) & (x <= z)

    def U(phi):
        zz = phi - x
        return float(np.sum(w * keep * ((0.35 + 0.9 * x - 0.25) * stats.norm.sf(zz) + 0.8 * stats.norm.pdf(zz))))
    r = optimize.minimize_scalar(lambda p: -U(p), bounds=(-3, 3), method="bounded", options={"xatol": 1e-10})
    return {"gh_nodes_inside_window": int(keep.sum()), "gh_argmax": float(r.x)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in designs():
        r = check_design(d)
        if d.name != "opposing_gaussian":
            r["aligned"] = aligned_bounds(d)
            if "phi_star_eps" in r:
                r["remark"] = untrimmed_remark(d, r["phi_star_eps"])
                r["eps_sensitivity"] = epsilon_sensitivity(d)
        else:
            r["profile"] = opposing_profile(d)
        summary[d.name] = r
        print(json.dumps(r, indent=1, default=float))
    summary["bimodal_counterexample"] = bimodal_counterexample()
    summary["halfspace_constant"] = {"J=[-1.7,1.7]": halfspace_constant(1.7), "J=[-2.2,2.2]": halfspace_constant(2.2)}
    summary["gauss_hermite_target"] = gauss_hermite_target()
    for key in ("bimodal_counterexample", "halfspace_constant", "gauss_hermite_target"):
        print(key, json.dumps(summary[key], default=float))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
