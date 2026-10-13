"""When is the stacked least-squares estimator of Mukherjee et al. efficient?

Checks the conditions behind Theorem 2.10 of Mukherjee, Banerjee and Ritov
(Bernoulli 2026) numerically (``this_week.tex``, section "Local efficiency at
Gaussian residuals", remark on their Theorem 2.10).  With W = (S, X, b'(eta) Z),
A = E cov(W | eta) (optionally on |eta| <= t) and B = blockdiag(0, 0, Sigma_Z):

    OLS stacking:  e1' (A+B)^-1 (s_eps^2 A + var_eta B) (A+B)^-1 e1   (their Thm 2.8)
    WLS stacking:  e1' (A/s_eps^2 + B/var_eta)^-1 e1                  (their Remark 2.11)
    bound:         e1' (A_full/s_eps^2 + I_eta B)^-1 e1               (their supplement 3.5-3.16)

with I_eta the location Fisher information of eta.  The three agree only for
Gaussian eta with var_eta = s_eps^2 and no truncation.  Design: X = Z ~ N(0, 1),
Q = Z + eta, S = 1{Q >= 0}, b(eta) = eta^3 / 3, s_eps^2 = 1.

Run:
    python -m experiments.scripts.mbr_efficiency_check
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats


def a_matrix(eta_pdf, lo, hi):
    def cov_given(e):
        p, ph, bp = stats.norm.cdf(e), stats.norm.pdf(e), e ** 2
        return np.array([[p * (1 - p), ph, bp * ph], [ph, 1.0, bp], [bp * ph, bp, bp ** 2]])
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = integrate.quad(lambda e: cov_given(e)[i, j] * eta_pdf(e),
                                     lo, hi, limit=200)[0]
    return A


B = np.diag([0.0, 0.0, 1.0])
E1 = np.array([1.0, 0.0, 0.0])


def v_ols(A, s2, v_eta):
    M = np.linalg.inv(A + B)
    return E1 @ M @ (s2 * A + v_eta * B) @ M @ E1


def v_wls(A, s2, v_eta):
    return E1 @ np.linalg.inv(A / s2 + B / v_eta) @ E1


def v_bound(A_full, s2, I_eta):
    return E1 @ np.linalg.inv(A_full / s2 + I_eta * B) @ E1


def main() -> None:
    s2 = 1.0
    print("eta law            var_eta  I_eta*var  trunc     OLS      WLS   bound(untruncated)")
    for name, sd in [("N(0,0.5^2)", 0.5), ("N(0,1)", 1.0), ("N(0,2^2)", 2.0)]:
        pdf = stats.norm(scale=sd).pdf
        v, I = sd ** 2, 1.0 / sd ** 2
        A_full = a_matrix(pdf, -np.inf, np.inf)
        for t in (np.inf, 1.5 * sd):
            A = a_matrix(pdf, -t, t)
            print(f"{name:18s} {v:7.3f} {I * v:9.3f}  {t:6.2f}  {v_ols(A, s2, v):7.4f}  "
                  f"{v_wls(A, s2, v):7.4f}  {v_bound(A_full, s2, I):7.4f}")
    nu = 5
    scale = np.sqrt((nu - 2) / nu)                 # t5 rescaled to variance 1
    pdf = stats.t(df=nu, scale=scale).pdf
    I = (nu + 1) / ((nu + 3) * scale ** 2)
    A_full = a_matrix(pdf, -np.inf, np.inf)
    print(f"{'t5 (var 1)':18s} {1.0:7.3f} {I:9.3f}  {np.inf:6.2f}  {v_ols(A_full, s2, 1.0):7.4f}  "
          f"{v_wls(A_full, s2, 1.0):7.4f}  {v_bound(A_full, s2, I):7.4f}")


if __name__ == "__main__":
    main()
