"""Checks behind Section 5 of manuscript/binary_outcome_sketch.tex (a generalized linear outcome model).

1. Chebyshev error of the separable expansion of z -> h(z+v) on [-3,3], v in [-6,6], for the
   logit, complementary log-log and probit links, against the rate rho^-M set by the strip width.
2. Rank of the matrix h(z_i + v_j): exactly 2 for the identity, 1 for exp, 2 for 1+2e^z; full for the logit.
3. A simulation of Lambda(e,t) = G1(c1(e),t) - G0(c0(e),t) under the arm-wise conditions (P1)-(P2).

    python experiments/scripts/binary_glm_link_checks.py
"""
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.stats import norm
rng = np.random.default_rng(0)
A = 3.0
zz = np.linspace(-A, A, 2001); vv = np.linspace(-6, 6, 121)
links = {"logit": (lambda z: 1/(1+np.exp(-z)), np.pi),
         "cloglog": (lambda z: 1-np.exp(-np.exp(z)), np.pi/2),
         "probit": (norm.cdf, None)}
print("1. Chebyshev error and ratio to rho^-M, rho = b/A + sqrt(1+b^2/A^2)")
for name, (h, b) in links.items():
    rho = b/A + np.sqrt(1 + b**2/A**2) if b else None
    out = []
    for M in (4, 8, 12, 16, 20):
        x = np.cos((2*np.arange(M)+1)*np.pi/(2*M))
        err = max(np.max(np.abs(C.chebval(zz/A, C.chebfit(x, h(A*x+v), M-1)) - h(zz+v))) for v in vv)
        out.append(f"M={M}: {err:.1e}" + (f" (ratio {err*rho**M:.2f})" if rho else ""))
    print(f"  {name:8s} rho={rho if rho is None else round(rho,3)}  " + "; ".join(out))
print("2. Rank of [h(z_i+v_j)], 200x200, singular values relative to the largest")
z = np.linspace(-3, 3, 200); v = np.linspace(-6, 6, 200)
for name, h in (("identity", lambda u: u), ("exp", np.exp), ("1+2e^z", lambda u: 1 + 2*np.exp(u)), ("logit", links["logit"][0])):
    s = np.linalg.svd(h(z[:, None] + v[None, :]), compute_uv=False); s = s / s[0]
    print(f"  {name:8s} sv[1..6]: " + ", ".join(f"{x:.1e}" for x in s[:6]))
print("3. Arm-wise conditions: Lambda(e,t) = G1(c1(e),t) - G0(c0(e),t) monotone in e and t")
for name, h in (("logit", links["logit"][0]), ("log", np.exp), ("cloglog", links["cloglog"][0])):
    e = np.linspace(-1, 1, 21); t = np.linspace(-1, 1, 21)
    c1 = 0.5 + 0.8*e; c0 = 0.8 - 0.6*e          # (P1): c1 up, c0 down
    n = 20000
    Lam = np.empty((len(e), len(t)))
    for j, tj in enumerate(t):                   # (P2): X'delta1 | T=t shifts up, X'delta0 | T=t shifts down
        v1 = 0.7*tj + rng.normal(0, 0.5, n); v0 = -0.4*tj + rng.normal(0, 0.5, n)
        for i in range(len(e)):
            Lam[i, j] = np.mean(h(c1[i] + v1)) - np.mean(h(c0[i] + v0))
    de, dt = np.diff(Lam, axis=0), np.diff(Lam, axis=1)
    print(f"  {name:8s} share de>=0 {np.mean(de>=-1e-3*np.abs(Lam).max()):.3f}, share dt>=0 {np.mean(dt>=-1e-3*np.abs(Lam).max()):.3f}, "
          f"Lambda range [{Lam.min():.3f}, {Lam.max():.3f}] (sign changes: {bool(Lam.min()<0<Lam.max())})")
