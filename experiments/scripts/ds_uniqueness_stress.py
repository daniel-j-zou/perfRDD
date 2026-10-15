"""Stress test of the aligned-heterogeneity theorem and the role of log-concavity.

`this_week.tex` Theorem (thm:week-id-aligned): under log-concave g (and f_eta when
lambda is not constant), nondecreasing a and lambda, and a sign change on Phi, the
marginal net effect tau_eps is strictly increasing, so U_eps has a unique interior
maximizer with curvature.

Part 1 checks the theorem on non-Gaussian designs: Gumbel, logistic and
exponential-power densities, nonlinear a and lambda. For each design it checks
  - the closed form B of tau_eps' against a finite difference of tau_eps,
  - U_eps' = -p_eps tau_eps against a finite difference of U_eps,
  - tau_eps' >= C_a + B_eps > 0 on a grid over Phi,
  - U_eps''(phi*) = -p_eps tau_eps'(phi*).

Part 2 asks whether log-concavity is necessary.
  - Not for a given design: Student-t3 T (log g convex for |t| > sqrt 3), and a
    bimodal f_eta with Gaussian T, still give an increasing tau_eps.
  - In the worst case, yes: where log g is strictly convex, a Gaussian f_eta of
    small variance and a linear a give tau_eps' < 0; where log f_eta is strictly
    convex, a Gaussian g of small variance and a linear lambda do the same. Tuning
    c then places a zero of tau_eps where it decreases, and U_eps has two local
    maxima.

Usage: python experiments/scripts/ds_uniqueness_stress.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_identification_check import (  # noqa: E402
    Design, dtau_forms, fd, fd2, gaussian, logistic, p_eps, tau, utility,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "ds_uniqueness_stress"


@dataclass
class WDesign(Design):
    """Design with an explicit trim window instead of one built from T quantiles."""
    win: tuple[float, float] = (-1.2816, 1.2816)

    @property
    def window(self) -> tuple[float, float]:
        return self.win


def gumbel(scale=1.0, loc=0.0):
    f = lambda x: stats.gumbel_r.pdf(x, loc, scale)
    df = lambda x: f(x) * (-1.0 + np.exp(-(x - loc) / scale)) / scale
    return f, df, lambda p: float(stats.gumbel_r.ppf(p, loc, scale))


def exppower(beta=4.0, scale=1.0):
    f = lambda x: stats.gennorm.pdf(x, beta, scale=scale)
    df = lambda x: f(x) * (-beta * np.sign(x) * np.abs(x / scale) ** (beta - 1) / scale)
    return f, df, lambda p: float(stats.gennorm.ppf(p, beta, scale=scale))


def student(nu=3.0, scale=1.0):
    f = lambda x: stats.t.pdf(x, nu, scale=scale)
    df = lambda x: f(x) * (-(nu + 1) * (x / scale) / (nu + (x / scale) ** 2) / scale)
    return f, df, lambda p: float(stats.t.ppf(p, nu, scale=scale))


def mixture(m, s):
    f = lambda x: 0.5 * (stats.norm.pdf(x, -m, s) + stats.norm.pdf(x, m, s))
    df = lambda x: 0.5 * (-(x + m) / s ** 2 * stats.norm.pdf(x, -m, s) - (x - m) / s ** 2 * stats.norm.pdf(x, m, s))
    return f, df, lambda p: float("nan")


def make(name, eta, T, a, lam, dlam, win, phi_lo=-2.0, phi_hi=2.0, target=0.3, c=None):
    """Design with c set so that tau_eps(target) = 0 unless c is given."""
    d = WDesign(name, *eta, *T, a=a, lam=lam, dlam=dlam, c=0.0, phi0=0.0, eps=0.1,
                phi_lo=phi_lo, phi_hi=phi_hi, win=win)
    d.c = tau(d, target) if c is None else c
    return d


def zeros_of_tau(d, grid):
    t = np.array([tau(d, x) for x in grid])
    ks = np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0]
    z = [optimize.brentq(lambda y: tau(d, y), grid[k], grid[k + 1], xtol=1e-12) for k in ks]
    return sorted({round(x, 9) for x in z})


def theorem_check(d: Design) -> dict:
    grid = np.linspace(d.phi_lo, d.phi_hi, 81)
    forms = [dtau_forms(d, x) for x in grid]
    lower = np.array([f["C_a"] + f["B_eps"] for f in forms])
    dtau = np.array([f["form_B"] for f in forms])
    cl = np.array([f["C_b"] for f in forms])
    pts = [d.phi_lo + 0.1, 0.0, d.phi_hi - 0.1]
    form_err = max(abs(dtau_forms(d, x)["form_B"] - fd(lambda y: tau(d, y), x)) for x in pts)
    formA_err = max(abs(f["form_A"] - f["form_B"]) for f in forms)
    rep_err = max(abs(fd(lambda y: utility(d, y), x, 1e-3) + p_eps(d, x) * tau(d, x)) for x in pts[:2])
    z = zeros_of_tau(d, grid)
    out = {"c": d.c, "min_dtau": float(dtau.min()), "min_Ca_plus_B": float(lower.min()),
           "min_C_lambda": float(cl.min()), "dtau_ge_lower": bool(np.all(dtau >= lower - 1e-10)),
           "formB_vs_fd": form_err, "formA_vs_formB": formA_err, "Uprime_vs_minus_p_tau": rep_err,
           "zeros": z}
    if len(z) == 1:
        zs = z[0]
        out["curvature_fd"] = fd2(lambda y: utility(d, y), zs, 2e-2)
        out["curvature_formula"] = -p_eps(d, zs) * dtau_forms(d, zs)["form_B"]
    return out


def part1() -> dict:
    res = {}
    W = (-1.2816, 1.2816)
    res["gumbel_T_logistic_eta_nonlinear"] = theorem_check(make(
        "gumbel_T", logistic(0.5), gumbel(0.8), a=lambda e: 0.2 + 0.6 * e + 0.2 * e ** 3,
        lam=lambda t: 0.3 * t + 0.5 * np.tanh(t), dlam=lambda t: 0.3 + 0.5 / np.cosh(t) ** 2, win=W))
    res["exppower_T_gumbel_eta_arctan"] = theorem_check(make(
        "exppower_T", gumbel(0.6), exppower(4.0, 1.5), a=lambda e: np.tanh(2 * e),
        lam=lambda t: 0.4 * np.arctan(2 * t), dlam=lambda t: 0.8 / (1 + 4 * t ** 2), win=(-1.0, 1.5)))
    res["logistic_T_flat_a_A3b"] = theorem_check(make(
        "flat_a", gaussian(0.0, 0.8), logistic(0.7), a=lambda e: 0.0 * e,
        lam=lambda t: np.exp(0.3 * np.clip(t, -60, 60)) - 1,
        dlam=lambda t: 0.3 * np.exp(0.3 * np.clip(t, -60, 60)), win=W))
    res["common_slope_gumbel_T_bimodal_eta"] = theorem_check(make(
        "common_bimodal", mixture(1.0, 0.35), gumbel(1.0), a=lambda e: 1 / (1 + np.exp(-4 * e)),
        lam=lambda t: 0.0 * t, dlam=lambda t: 0.0 * t, win=W))
    return res


def part2() -> dict:
    res = {}
    W = (-1.2816, 1.2816)
    # (a) Not necessary for a given design: t3 T is not log-concave beyond sqrt 3,
    #     yet with the paper's window and Phi = [-3, 3] tau_eps is increasing.
    d = make("t3_common", gaussian(), student(3.0), a=lambda e: 0.35 + 0.9 * e,
             lam=lambda t: 0.0 * t, dlam=lambda t: 0.0 * t, win=W, phi_lo=-3.0, phi_hi=3.0)
    grid = np.linspace(-3, 3, 121)
    res["t3_paper_window"] = {"logconcave_region": [-np.sqrt(3), np.sqrt(3)],
                              "V_Phi": [-3 - W[1], 3 - W[0]],
                              "min_dtau_on_Phi": float(min(dtau_forms(d, x)["form_B"] for x in grid)),
                              "zeros": zeros_of_tau(d, grid)}
    # (b) Not necessary for a given design: bimodal f_eta, Gaussian T, lambda(t) = t.
    d = make("bimodal_mild", mixture(0.7, 0.5), gaussian(), a=lambda e: 0.0 * e,
             lam=lambda t: t, dlam=lambda t: 1.0 + 0 * t, win=W, phi_lo=-3.0, phi_hi=3.0, target=0.0)
    ee = np.linspace(-1.2, 1.2, 241)
    f, df, _ = mixture(0.7, 0.5)
    s_eta = df(ee) / f(ee)
    res["bimodal_eta_mild"] = {"s_eta_increasing_somewhere": bool(np.any(np.diff(s_eta) > 0)),
                               "min_dtau_on_Phi": float(min(dtau_forms(d, x)["form_B"] for x in grid)),
                               "max_var_margin": float(max(dtau_forms(d, x)["var_E"] for x in grid)),
                               "zeros": zeros_of_tau(d, grid)}
    # (c) Worst case for g: t3 T at t = 2.5 (log g strictly convex), concentrated
    #     Gaussian eta, linear a. tau_eps' < 0 near phi = 2.5.
    win = (-0.6, 0.6)
    d = make("t3_worst", gaussian(0.0, 0.15), student(3.0), a=lambda e: e,
             lam=lambda t: 0.0 * t, dlam=lambda t: 0.0 * t, win=win, phi_lo=-6.0, phi_hi=6.0, target=2.5)
    grid = np.linspace(-6, 6, 241)
    z = zeros_of_tau(d, grid)
    res["t3_worst_case"] = {"dtau_at_2.5": dtau_forms(d, 2.5)["form_B"], "c": d.c, "zeros": z,
                            "dtau_at_zeros": [dtau_forms(d, x)["form_B"] for x in z],
                            "U_at_zeros": [utility(d, x) for x in z], "U_at_phi_hi": utility(d, 6.0),
                            "U_treat_none_limit": 0.0}
    # (d) Worst case for f_eta: bimodal f_eta (log f_eta strictly convex at 0),
    #     concentrated Gaussian T, linear lambda, constant a. tau_eps' < 0 at 0.
    d = make("feta_worst", mixture(0.7, 0.5), gaussian(0.0, 0.3), a=lambda e: 0.0 * e,
             lam=lambda t: t, dlam=lambda t: 1.0 + 0 * t, win=W, phi_lo=-1.5, phi_hi=1.5, target=0.0)
    res["feta_worst_case"] = {"dtau_at_0": dtau_forms(d, 0.0), "zeros": zeros_of_tau(d, np.linspace(-1.5, 1.5, 121))}
    # (e) Worst case for g in the common-slope model: a bimodal index, as a strong
    #     binary covariate in the score would give. Symmetric design, so the two
    #     local maxima of U_eps tie.
    d = make("bimodal_T_common", gaussian(0.0, 0.3), mixture(1.0, 0.4), a=lambda e: e,
             lam=lambda t: 0.0 * t, dlam=lambda t: 0.0 * t, win=W, phi_lo=-3.0, phi_hi=3.0, c=0.0)
    z = zeros_of_tau(d, np.linspace(-3, 3, 241))
    res["bimodal_T_common_slope"] = {"dtau_at_0": dtau_forms(d, 0.0)["form_B"], "zeros": z,
                                     "dtau_at_zeros": [dtau_forms(d, x)["form_B"] for x in z],
                                     "U_at_zeros": [utility(d, x) for x in z]}
    return res


def main() -> None:
    out = {"part1_theorem": part1(), "part2_logconcavity": part2()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
