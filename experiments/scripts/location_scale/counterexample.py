"""Single crossing can fail under location-scale although the additive analog has a
unique crossing (identification reviewer, 2026-10-21).

phi0 = 0, eps = 0.1, T = X1 ~ N(0,1), X2 = +-1 (probability 1/2 each), independent of X1,
s(X) = exp(X2/2), zeta ~ N(0,1) independent of X, alpha(e) = 0.2 e, beta2 = 0.5 on X2, c = 0.
tau_eps(phi) = E[s^-1 f(t) 1{l0 < t < u0} (alpha(t) + 0.5 X2)] / E[s^-1 f(t) 1{l0 < t < u0}],
t = t_phi(X) = (phi - T)/s(X); [l0, u0] are the 0.1 and 0.9 quantiles of t_phi0(X).
In the additive analog (s = 1) the zero of tau is unique (common-slope corollary: beta2 X2
is independent of the index, so lambda = 0)."""
import numpy as np
from scipy import optimize, stats

x1 = np.linspace(-9, 9, 20001); w1 = stats.norm.pdf(x1); w1 /= w1.sum()


def window(scale=True, eps=0.1):
    t0 = np.concatenate([(0.0 - x1) / (np.exp(x2 / 2) if scale else 1.0) for x2 in (-1.0, 1.0)])
    w = np.concatenate([w1 / 2, w1 / 2])
    o = np.argsort(t0); c = np.cumsum(w[o])
    return t0[o][np.searchsorted(c, eps)], t0[o][np.searchsorted(c, 1 - eps)]


def tau(phi, scale=True, eps=0.1):
    l0, u0 = window(scale, eps)
    num = den = 0.0
    for x2 in (-1.0, 1.0):
        s = np.exp(x2 / 2) if scale else 1.0
        t = (phi - x1) / s
        k = w1 / 2 * stats.norm.pdf(t) / s * ((t > l0) & (t < u0))
        num += np.sum(k * (0.2 * t + 0.5 * x2)); den += np.sum(k)
    return num / den


for scale in (True, False):
    grid = np.linspace(-3, 3, 1200)   # even count: no grid point at the symmetric zero
    v = np.array([tau(p, scale) for p in grid])
    zeros = [optimize.brentq(lambda p: tau(p, scale), a, b)
             for a, b, fa, fb in zip(grid[:-1], grid[1:], v[:-1], v[1:]) if fa * fb < 0]
    print(("location-scale" if scale else "additive analog") + f": window {np.round(window(scale), 3).tolist()}, "
          f"zeros of tau on [-3, 3]: {np.round(zeros, 3).tolist()}, "
          f"tau' sign at zeros: {[int(np.sign(tau(z + 1e-4, scale) - tau(z - 1e-4, scale))) for z in zeros]}")
