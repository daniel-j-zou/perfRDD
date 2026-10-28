"""Cheap recomputations for the 2026-10-27 review of differing_slopes.tex.

All designs: phi_0 = 0, eps = 0.1 (unless stated), window [l0,u0] = [-q, q] with
q = Phi^{-1}(0.9) for a N(0,1) index.  One-dimensional quadrature only.
Run with OMP_NUM_THREADS=1.
"""
import numpy as np
from scipy import integrate, optimize, stats

Z = stats.norm
Q90 = Z.ppf(0.9)


def make_design(alpha, c, lam, g=Z.pdf, Gbar=Z.sf, f_eta=Z.pdf, l0=-Q90, u0=Q90,
                tail_lam_int=None):
    """Return a dict of population functions for one design.

    alpha(e): effect at latent ability e;  lam(t): beta2' E(X | T=t);
    tail_lam_int(s) = int_s^inf lam(t) g(t) dt  (= beta2' H_X(s)); if None it is
    computed by quadrature.
    """
    if tail_lam_int is None:
        def tail_lam_int(s):
            return integrate.quad(lambda t: lam(t) * g(t), s, np.inf)[0]

    def U(phi):
        f = lambda e: ((alpha(e) - c) * Gbar(phi - e) + tail_lam_int(phi - e)) * f_eta(e)
        return integrate.quad(f, l0, u0, limit=200)[0]

    def p_eps(phi):
        return integrate.quad(lambda e: g(phi - e) * f_eta(e), l0, u0, limit=200)[0]

    def tau(phi):
        num = integrate.quad(lambda e: (alpha(e) - c + lam(phi - e)) * g(phi - e) * f_eta(e),
                             l0, u0, limit=200)[0]
        return num / p_eps(phi)

    def Uprime(phi):
        return -p_eps(phi) * tau(phi)

    def Udd(phi, h=1e-4):
        return (Uprime(phi + h) - Uprime(phi - h)) / (2 * h)

    return dict(U=U, p=p_eps, tau=tau, Up=Uprime, Udd=Udd, alpha=alpha, c=c, lam=lam,
                g=g, f_eta=f_eta, l0=l0, u0=u0)


def var_psi_U(d, phi):
    """Var of psi_U = -F*(eta), F*(e) = {(alpha(e)-c) + lam(phi-e)} g(phi-e) 1{window}."""
    F = lambda e: (d['alpha'](e) - d['c'] + d['lam'](phi - e)) * d['g'](phi - e)
    m1 = integrate.quad(lambda e: F(e) * d['f_eta'](e), d['l0'], d['u0'], limit=200)[0]
    m2 = integrate.quad(lambda e: F(e) ** 2 * d['f_eta'](e), d['l0'], d['u0'], limit=200)[0]
    return m2 - m1 ** 2, m1


def var_psi_T(d, phi, cond_var_beta2X, eps=0.1, phi0=0.0):
    """Var of psi_T = -r_T(T)'Xtilde - b_l z_{1-eps}(T) + b_u z_eps(T) for a N(0,1) index.

    cond_var_beta2X = Var(beta2'X | T), constant in the Gaussian designs.
    """
    g, f_eta, l0, u0 = d['g'], d['f_eta'], d['l0'], d['u0']
    q_lo, q_hi = Z.ppf(eps), Z.ppf(1 - eps)
    F = lambda e: (d['alpha'](e) - d['c'] + d['lam'](phi - e)) * g(phi - e)
    b_l, b_u = F(l0) * f_eta(l0), F(u0) * f_eta(u0)
    a, b = phi - u0, phi - l0          # support of r_T in t

    def z(p, q, t):
        return (p - (t <= q)) / g(q)

    def m(t):  # E[psi_T | T=t]
        r = (d['alpha'](phi - t) - d['c'] + d['lam'](t)) * f_eta(phi - t) if a <= t <= b else 0.0
        return -r - b_l * z(1 - eps, q_hi, t) + b_u * z(eps, q_lo, t)

    pts = sorted([a, b, q_lo, q_hi])
    mean = integrate.quad(lambda t: m(t) * g(t), -8, 8, points=pts, limit=400)[0]
    m2 = integrate.quad(lambda t: m(t) ** 2 * g(t), -8, 8, points=pts, limit=400)[0]
    cv = integrate.quad(lambda t: f_eta(phi - t) ** 2 * g(t), a, b, limit=200)[0] * cond_var_beta2X
    return m2 - mean ** 2 + cv, mean


def solve(d, lo, hi):
    phi = optimize.brentq(d['tau'], lo, hi, xtol=1e-10)
    return phi, d['Udd'](phi)


out = []
P = out.append

# ---------------------------------------------------------------- displayed §5 design
# X ~ N(0,I3), gamma = (0,1,1,1)/sqrt3, T ~ N(0,1), alpha(e) = 2+e, c = 2.25,
# beta2 = (0.4,-0.25,0.15): lam(t) = 0.3 t/sqrt3; Var(beta2'X | T) = |beta2|^2 - (sum beta2)^2/3.
s3 = np.sqrt(3.0)
beta2 = np.array([0.4, -0.25, 0.15])
cvX = beta2 @ beta2 - beta2.sum() ** 2 / 3
dD = make_design(lambda e: 2 + e, 2.25, lambda t: 0.3 * t / s3,
                 tail_lam_int=lambda s: 0.3 / s3 * Z.pdf(s))  # int_s^inf t phi(t) dt = phi(s)
phi, H = solve(dD, 0.0, 1.0)
vU, mU = var_psi_U(dD, phi)
vT, mT = var_psi_T(dD, phi, cvX)
P(f"Displayed design: phi*_eps = {phi:.6f} (paper 0.545103), H = {H:.6f} (paper -0.107534)")
P(f"  Var psi_U = {vU:.5f} (paper 0.0145), E F* = {mU:.2e};  Var psi_T = {vT:.5f} (paper 0.0434), E psi_T = {mT:.2e}")
P(f"  oracle Var psi_U/H^2 = {vU / H**2:.4f} (paper 1.256)")
sig2 = 0.0145 + 0.2181 + 0.0434 + 0.0834
P(f"  sigma^2 from the four paper values = {sig2:.4f}; sigma^2/H^2 with paper H = {sig2 / 0.107534**2:.3f} (paper 31.09); x4 = {4 * sig2 / 0.107534**2:.1f} (paper 124.4)")

# ---------------------------------------------------------------- §5.4 nonlinear design
# T = X1, X ~ N(0,I2): lam(t) = 0.8 t, Var(beta2'X | T) = 0.25^2.
dN = make_design(lambda e: 0.35 + 0.9 * e + 0.35 * (e * e - 1), 0.25, lambda t: 0.8 * t,
                 tail_lam_int=lambda s: 0.8 * Z.pdf(s))
phiN, HN = solve(dN, -1.0, 1.0)
vUN, _ = var_psi_U(dN, phiN)
vTN, mTN = var_psi_T(dN, phiN, 0.25 ** 2)
P(f"Nonlinear design: phi*_eps = {phiN:.6f} (paper 0.152143), H = {HN:.6f} (paper -0.219539)")
P(f"  Var psi_U = {vUN:.6f}; oracle Var psi_U/H^2 = {vUN / HN**2:.4f} (paper 0.0375);  Var psi_T = {vTN:.6f} (log 0.008753)")
P(f"  sigma^2/H^2 from the log's four values = {(0.001809 + 0.21731 + 0.008753 + 6.3e-5) / 0.219539**2:.3f} (paper 4.73)")

# ---------------------------------------------------------------- Appendix 5 common-slope baseline
# alpha(e) = 2+e, c = 2.25, beta2 = 0, T ~ N(0,1), eta ~ N(0,1).
dC = make_design(lambda e: 2 + e, 2.25, lambda t: 0.0, tail_lam_int=lambda s: 0.0)
phiC, HC = solve(dC, 0.0, 1.5)
vUC, _ = var_psi_U(dC, phiC)
vTC, mTC = var_psi_T(dC, phiC, 0.0)
P(f"App 5 common-slope baseline: phi*_eps = {phiC:.5f} (paper 0.7313), H = {HC:.5f} (paper -0.0720)")
P(f"  Var psi_U = {vUC:.5f} (paper 0.0185);  Var psi_T = {vTC:.5f} (paper 0.0363)")
s2C = 0.0185 + 0.1700 + 0.0363 + 0.0018
P(f"  sigma^2 sum = {s2C:.4f} (paper 0.2265); 4 sigma^2/H^2 = {4 * s2C / 0.0720**2:.1f} (paper 174.7); sigma^2/H^2 = {s2C / 0.0720**2:.2f} (paper 43.68)")
P(f"  App 5 Table ratios: 175.2/42.4={175.2/42.4:.3f} (4.14), 173.5/40.1={173.5/40.1:.3f} (4.32), 175.8/43.3={175.8/43.3:.3f} (4.06)")
P(f"  App 5 ds table: 17.53/4.52={17.53/4.52:.3f} (3.88) 17.53/4.31={17.53/4.31:.3f} (4.07) 17.32/4.22={17.32/4.22:.3f} (4.11) 17.32/4.25={17.32/4.25:.3f} (4.08) 17.40/4.38={17.40/4.38:.3f} (3.98) 17.40/4.37={17.40/4.37:.3f} (3.98)")

# ---------------------------------------------------------------- Appendix 4 targets
# baseline: alpha = 0.35+0.9e, c = 0.25, lam = 0.8 t  (four-role check design)
dB = make_design(lambda e: 0.35 + 0.9 * e, 0.25, lambda t: 0.8 * t, tail_lam_int=lambda s: 0.8 * Z.pdf(s))
phiB, HB = solve(dB, -1.0, 1.0)
P(f"App 4 baseline: phi*_eps = {phiB:.6f} (paper -0.119770), H = {HB:.4f} (App 3 table -0.2180)")
P(f"  window Phi^-1(0.9) = {Q90:.5f} (App 4: 1.28155)")
# quadratic design: treated effect + 0.9 (X1^2 - 1), T = X1: E[W | eta, T=t] adds 0.9(t^2-1).
# int_s^inf (t^2 - 1) phi(t) dt = s phi(s)
dQ = make_design(lambda e: 0.35 + 0.9 * e, 0.25, lambda t: 0.8 * t + 0.9 * (t * t - 1),
                 tail_lam_int=lambda s: 0.8 * Z.pdf(s) + 0.9 * s * Z.pdf(s))
grid = np.linspace(-3, 3, 241)
Ug = np.array([dQ['U'](x) for x in grid])
x0 = grid[Ug.argmax()]
res = optimize.minimize_scalar(lambda x: -dQ['U'](x), bounds=(x0 - 0.05, x0 + 0.05), method='bounded',
                               options={'xatol': 1e-9})
P(f"App 4 quadratic design: argmax U_eps on [-3,3] = {res.x:.6f} (paper 0.484416); grid argmax {x0:.3f}")
zeros = [optimize.brentq(dQ['tau'], grid[i], grid[i + 1]) for i in range(len(grid) - 1)
         if dQ['tau'](grid[i]) * dQ['tau'](grid[i + 1]) < 0]
P(f"  zeros of tau_eps on [-3,3]: {np.round(zeros, 4)}")

# t5 target 0.587810: try T ~ t5 (raw and standardized), common-slope baseline, eta ~ N(0,1)
for name, dist in [("t5 raw", stats.t(5)), ("t5 standardized", stats.t(5, scale=np.sqrt(3 / 5)))]:
    qq = dist.ppf(0.9)
    dT = make_design(lambda e: 2 + e, 2.25, lambda t: 0.0, g=dist.pdf, Gbar=dist.sf,
                     l0=-qq, u0=qq, tail_lam_int=lambda s: 0.0)
    try:
        ph, _ = solve(dT, -1.0, 2.0)
        P(f"App 5 t5 target, {name} index, common-slope baseline: phi*_eps = {ph:.6f} (paper 0.587810)")
    except ValueError:
        P(f"App 5 t5 target, {name}: no single zero in [-1,2]")
    # eta ~ t5 instead
    dT2 = make_design(lambda e: 2 + e, 2.25, lambda t: 0.0, f_eta=dist.pdf, tail_lam_int=lambda s: 0.0)
    try:
        ph2, _ = solve(dT2, -1.0, 2.0)
        P(f"App 5 t5 target, {name} residual, common-slope baseline: phi*_eps = {ph2:.6f}")
    except ValueError:
        pass

# ---------------------------------------------------------------- Table 3 (eps sensitivity) arithmetic
P(f"Table 3: slope (0.590-0.505)/0.05 = {(0.590 - 0.505) / 0.05:.2f}; 143.3/20.1 = {143.3 / 20.1:.2f}; 1.24/sqrt(136.9) = {1.24 / np.sqrt(136.9):.3f}")
ratios = [19.2 / 20.1, 23.6 / 24.8, 29.8 / 31.1, 37.8 / 39.5, 48.7 / 50.7, 81.0 / 84.9, 136.9 / 143.3]
P(f"  n=64000 nVar / V: {np.round(1 - np.array(ratios), 3)} (paper: 4-5% below); MC se of a variance from 1000 sets = {np.sqrt(2 / 999):.3f}")
P(f"  step sd: sqrt(34.2/16000) = {np.sqrt(34.2 / 16000):.4f} (0.046); sqrt(29.8/64000) = {np.sqrt(29.8 / 64000):.4f} (0.022)")
# Table 1 / §5.2 arithmetic
P(f"Table 1: 132.2/33.0 = {132.2 / 33.0:.2f}, 125.1/30.3 = {125.1 / 30.3:.2f}; 30.3/31.09-1 = {30.3 / 31.09 - 1:.3f}, 30.1/31.09-1 = {30.1 / 31.09 - 1:.3f}")
P(f"  0.41/sqrt(30.3) = {0.41 / np.sqrt(30.3):.3f} sd (paper 0.07); 4B/sqrt n = {4 * 21.6 / np.sqrt(16000):.3f}, {4 * 21.6 / np.sqrt(64000):.3f}; B/sqrt n = {21.6 / np.sqrt(16000):.3f}, {21.6 / np.sqrt(64000):.3f}")
P(f"  combination bias (4*0.36-0.89)/3 = {(4 * 0.36 - 0.89) / 3:.3f}, (4*0.15-0.41)/3 = {(4 * 0.15 - 0.41) / 3:.3f}")
P(f"Table 2: 42.2/36.3 = {42.2 / 36.3:.3f} (16%), 33.7/34.3 = {33.7 / 34.3:.3f}, 31.4/30.8 = {31.4 / 30.8:.3f}; cov se sqrt(.95*.05/500) = {np.sqrt(.95 * .05 / 500):.4f}, /200 = {np.sqrt(.95 * .05 / 200):.4f}")
P(f"Table 4: 5.49/4.73 = {5.49 / 4.73:.3f}, 5.45/4.73 = {5.45 / 4.73:.3f}, 5.63/4.73 = {5.63 / 4.73:.3f}; MC se sqrt(2/499) = {np.sqrt(2 / 499):.3f}, sqrt(2/299) = {np.sqrt(2 / 299):.3f}; 0.21/sqrt(12800) = {0.21 / np.sqrt(12800):.4f}")

# ---------------------------------------------------------------- 199-draw percentile coverage of an exact normal bootstrap
rng = np.random.default_rng(1)
R, B = 200000, 199
W = rng.standard_normal(R)                       # theta-hat - theta
draws = W[:, None] + rng.standard_normal((R, B))  # theta* - theta
lo = np.percentile(draws, 2.5, axis=1)
hi = np.percentile(draws, 97.5, axis=1)
cov_lin = np.mean((lo <= 0) & (0 <= hi))
srt = np.sort(draws, axis=1)
cov_ord = np.mean((srt[:, 4] <= 0) & (0 <= srt[:, 194]))   # order statistics 5 and 195
P(f"199-draw percentile coverage, exact normal bootstrap: numpy linear interpolation {cov_lin:.4f}; order stats (5,195) {cov_ord:.4f} (paper: about 0.940)")

# ---------------------------------------------------------------- taxi fare grid
fares = 2.5 + 0.4 * np.arange(0, 80)
cell = lambda phi: fares[fares >= phi - 1e-9].min() if (fares >= phi).any() else np.nan
P("Taxi cells (smallest grid fare >= estimate): " + ", ".join(f"{x:.2f}->{cell(x):.2f}" for x in [12.87, 13.00, 13.08, 12.88, 12.94, 12.92, 12.93, 12.89, 11.87, 11.76, 6.00, 13.15, 12.56, 13.18, 15.0]))
P(f"  Q = 2(sqrt(fare)-1): Box-Cox 1/2 check at 15: {2 * (np.sqrt(15) - 1):.4f}; grid fares around 15: {fares[(fares > 14.5) & (fares < 15.5)]}")
P(f"  design variations max move: 13.08-12.87 = {13.08 - 12.87:.2f}; 540426/633616 = {540426 / 633616:.3f}")

print("\n".join(out))
