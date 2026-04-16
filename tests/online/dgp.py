"""
DGP adapted from plm.py.
eta: bimodal via a ~ +/-Exp(0.3), eta|a ~ N(a, sqrt(1-2*0.3^2)) => Var(eta)=1
alpha(eta) = ALPHA1 + DELTA1*eta  (linear, matches working model)
nu ~ N(0,1) independent of eta
"""
import numpy as np

ALPHA1  = 1.0
DELTA1  = 0.7
SIGMA_W = 0.3

_A_SCALE   = 0.3                        # a ~ +/-Exp(0.3)
_ETA_NOISE = np.sqrt(1 - 2*_A_SCALE**2) # cond std so Var(eta)=1

def pGen_online(n, beta, gamma, rng):
    signs = rng.choice([-1., 1.], size=n)
    a     = signs * rng.exponential(_A_SCALE, size=n)   # Var(a)=2*0.09=0.18
    eta   = rng.normal(loc=a, scale=_ETA_NOISE, size=n) # Var(eta)=0.18+0.82=1
    nu    = rng.normal(0.0, 1.0, size=n)                # independent of eta
    w     = rng.normal(ALPHA1 + DELTA1*eta, SIGMA_W, size=n)
    x     = rng.normal(0.0, 1.0, size=n)
    Q     = gamma*x + eta
    return x, eta, nu, w, Q, a

def compute_phi_star(beta, gamma, C, rng_seed=0, N_big=200000):
    rng_big = np.random.default_rng(rng_seed)
    x_b, eta_b, nu_b, w_b, Q_b, _ = pGen_online(N_big, beta, gamma, rng_big)
    order = np.argsort(Q_b)
    Q_s, w_s = Q_b[order], w_b[order]
    diff  = w_s - C
    util0 = (diff.sum() - np.cumsum(diff)) / N_big
    idx   = np.argmax(util0)
    return Q_s[idx], Q_s, util0

# ── jax_online-style DGP ──────────────────────────────────────────────────────
# Simple linear Gaussian: perfectly specified, matches working model
# X ~ N(0,1), eta ~ N(0,1), eps ~ N(0, 0.5^2)
# Q = gamma*x + eta
# w = delta_1*eta + alpha_1        (treatment effect, linear, no extra noise)
# nu = delta_0*eta + alpha_0 + eps (base outcome component)
# Y = D*w + beta*x + nu
JAX_GAMMA   = 1.0
JAX_BETA    = 0.5
JAX_DELTA0  = 1.0
JAX_ALPHA0  = 0.0
JAX_DELTA1  = 2.0
JAX_ALPHA1  = 1.0
JAX_SIGMA_E = 0.5

def pGen_jax(n, rng,
             gamma=JAX_GAMMA, beta=JAX_BETA,
             delta_0=JAX_DELTA0, alpha_0=JAX_ALPHA0,
             delta_1=JAX_DELTA1, alpha_1=JAX_ALPHA1,
             sigma_e=JAX_SIGMA_E):
    x   = rng.normal(0.0, 1.0, size=n)
    eta = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, sigma_e, size=n)
    Q   = gamma * x + eta
    w   = delta_1 * eta + alpha_1          # no noise — pure signal
    nu  = delta_0 * eta + alpha_0 + eps    # base outcome noise
    return x, eta, nu, w, Q

def compute_phi_star_jax(C, rng_seed=0, N_big=500000):
    rng_big = np.random.default_rng(rng_seed)
    x_b, eta_b, nu_b, w_b, Q_b = pGen_jax(N_big, rng_big)
    order = np.argsort(Q_b)
    Q_s, w_s = Q_b[order], w_b[order]
    diff  = w_s - C
    util0 = (diff.sum() - np.cumsum(diff)) / N_big
    idx   = np.argmax(util0)
    return Q_s[idx], Q_s, util0
# Append to dgp.py on server — complex nonlinear DGP
# alpha(eta) = 2*sigmoid(eta)  — truly nonlinear treatment effect
# eta: bimodal via a ~ +/-Exp(0.5), more pronounced than pGen_online
# nu ~ N(0, 0.5), independent
# Q = gamma*x + eta

import numpy as np

def pGen_complex(n, gamma, rng):
    """
    Complex nonlinear DGP:
      a    ~ +/-Exp(0.5)           bimodal latent (more spread than pGen_online)
      eta  | a ~ N(a, sqrt(1-2*0.5^2)) = N(a, sqrt(0.5))   Var(eta)=1
      nu   ~ N(0, 0.5)             independent noise
      w    = 2*exp(eta)/(1+exp(eta)) + N(0, 0.3)  nonlinear treatment effect
      x    ~ N(0,1)
      Q    = gamma*x + eta
    True alpha(eta) = E[w|eta] = 2*sigmoid(eta)  -- nonlinear, no closed form for phi*
    """
    _A_SCALE   = 0.5
    _ETA_NOISE = np.sqrt(1 - 2*_A_SCALE**2)   # sqrt(0.5)
    signs = rng.choice([-1., 1.], size=n)
    a     = signs * rng.exponential(_A_SCALE, size=n)
    eta   = rng.normal(loc=a, scale=_ETA_NOISE, size=n)
    nu    = rng.normal(0.0, 0.5, size=n)
    w     = 2.0 / (1.0 + np.exp(-eta)) + rng.normal(0.0, 0.3, size=n)
    x     = rng.normal(0.0, 1.0, size=n)
    Q     = gamma * x + eta
    return x, eta, nu, w, Q

def compute_phi_star_complex(gamma, C, rng_seed=0, N_big=500000):
    rng_big = np.random.default_rng(rng_seed)
    x_b, eta_b, nu_b, w_b, Q_b = pGen_complex(N_big, gamma, rng_big)
    order  = np.argsort(Q_b)
    Q_s, w_s = Q_b[order], w_b[order]
    diff   = w_s - C
    util0  = (diff.sum() - np.cumsum(diff)) / N_big
    idx    = np.argmax(util0)
    return Q_s[idx], Q_s, util0
# Append to dgp.py on server — complex nonlinear DGP
# alpha(eta) = 2*sigmoid(eta)  — truly nonlinear treatment effect
# eta: bimodal via a ~ +/-Exp(0.5), more pronounced than pGen_online
# nu ~ N(0, 0.5), independent
# Q = gamma*x + eta

import numpy as np

def pGen_complex(n, gamma, rng):
    """
    Complex nonlinear DGP:
      a    ~ +/-Exp(0.5)           bimodal latent (more spread than pGen_online)
      eta  | a ~ N(a, sqrt(1-2*0.5^2)) = N(a, sqrt(0.5))   Var(eta)=1
      nu   ~ N(0, 0.5)             independent noise
      w    = 2*exp(eta)/(1+exp(eta)) + N(0, 0.3)  nonlinear treatment effect
      x    ~ N(0,1)
      Q    = gamma*x + eta
    True alpha(eta) = E[w|eta] = 2*sigmoid(eta)  -- nonlinear, no closed form for phi*
    """
    _A_SCALE   = 0.5
    _ETA_NOISE = np.sqrt(1 - 2*_A_SCALE**2)   # sqrt(0.5)
    signs = rng.choice([-1., 1.], size=n)
    a     = signs * rng.exponential(_A_SCALE, size=n)
    eta   = rng.normal(loc=a, scale=_ETA_NOISE, size=n)
    nu    = rng.normal(0.0, 0.5, size=n)
    w     = 2.0 / (1.0 + np.exp(-eta)) + rng.normal(0.0, 0.3, size=n)
    x     = rng.normal(0.0, 1.0, size=n)
    Q     = gamma * x + eta
    return x, eta, nu, w, Q

def compute_phi_star_complex(gamma, C, rng_seed=0, N_big=500000):
    rng_big = np.random.default_rng(rng_seed)
    x_b, eta_b, nu_b, w_b, Q_b = pGen_complex(N_big, gamma, rng_big)
    order  = np.argsort(Q_b)
    Q_s, w_s = Q_b[order], w_b[order]
    diff   = w_s - C
    util0  = (diff.sum() - np.cumsum(diff)) / N_big
    idx    = np.argmax(util0)
    return Q_s[idx], Q_s, util0

# ── Inverted-U DGP (right-shifted peaks) ──────────────────────────────────────
# alpha(eta) = 2*exp(-(eta-0.75)^2)  treatment effect: inverted U, peak at eta=0.75
# b(eta)     = exp(-0.5*(eta-0.5)^2) base outcome: inverted U, peak at eta=0.5
# Interpretation: treatment benefits mid eta individuals most, phi* near 0 (~50/50 treat ratio)
# phi* falls near center of Q distribution

def pGen_invU(n, gamma, beta, rng, sigma_w=0.3, sigma_nu=0.5):
    x        = rng.normal(0.0, 1.0, size=n)
    eta      = rng.normal(0.0, 1.0, size=n)
    noise_w  = rng.normal(0.0, sigma_w,  size=n)
    noise_nu = rng.normal(0.0, sigma_nu, size=n)
    Q  = gamma * x + eta
    w  = 2.0 * np.exp(-(eta - 0.75)**2) + noise_w    # peak at eta=0.75
    nu = np.exp(-0.5 * (eta - 0.5)**2) + noise_nu     # peak at eta=0.5
    return x, eta, nu, w, Q

def compute_phi_star_invU(gamma, beta, C, rng_seed=0, N_big=500000):
    rng_big = np.random.default_rng(rng_seed)
    x_b, eta_b, nu_b, w_b, Q_b = pGen_invU(N_big, gamma, beta, rng_big)
    order  = np.argsort(Q_b)
    Q_s, w_s = Q_b[order], w_b[order]
    diff   = w_s - C
    util0  = (diff.sum() - np.cumsum(diff)) / N_big
    idx    = np.argmax(util0)
    return Q_s[idx], Q_s, util0
