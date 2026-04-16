"""
Exp13: Online sieve (growing Hermite basis) + TSplm offline phi initialization.
  - DGP: pGen_invU  — inverted-U nonlinear alpha and b
      alpha(eta) = 2*exp(-(eta-0.75)^2)  treatment effect: inverted U
      b(eta)     = exp(-0.5*eta^2)    base outcome: nonlinear inverted U
  - Phase 1 (t < m0): RLS warm-up only, collect data
  - At t = m0: TSplm offline estimate -> phi_init
  - Phase 2 (t >= m0): online SGD for phi with Hermite sieve
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermevander
from datetime import datetime
import os
sys.path.insert(0, '/home/cyuhao/simulation_online')
from dgp import pGen_invU, compute_phi_star_invU

# ── Config ─────────────────────────────────────────────────────────────────────
rng        = np.random.default_rng(42)
n          = 540000
m0         = 2000
gamma      = 2.0
beta       = 0.5
C          = 1.0
phi_init   = 0.0
lambda0    = 0.5
phi_clip   = 5.0
KDE_WINDOW = 1000
GRAD_FREQ  = 5
K0         = 3
PLOT_FREQ  = 10000

phi_star, _, _ = compute_phi_star_invU(gamma, beta, C)
print(f"phi* = {phi_star:.4f}")

x, eta, nu, w, Q = pGen_invU(n, gamma, beta, rng)

# ── TSplm (from plm.py) ────────────────────────────────────────────────────────
def TSplm(y, xv, q, kn=12, bwf=1.0):
    y = np.asarray(y); xv = np.asarray(xv); q = np.asarray(q)
    ql = q.min(); qr = float(q.max() - q.min()) or 1.0
    mu = ql + (np.arange(1, kn+1) - 0.5) * qr / kn
    bw = bwf * qr / kn
    Phi = norm.pdf((q[:,None] - mu[None,:]), scale=bw)
    H   = np.column_stack((xv, Phi))
    v   = H.var(axis=0, ddof=1)
    m1  = v > 0.03 * np.max(v[1:]) if len(v) > 1 else np.ones(len(v), bool)
    be1 = np.zeros(H.shape[1])
    if m1.any():
        s, *_ = np.linalg.lstsq(H[:,m1], y, rcond=None)
        be1[m1] = s
    Phi2 = norm.pdf((q[:,None] + 0.01*xv[:,None] - mu[None,:]), scale=bw)
    H2   = np.column_stack((xv, (np.column_stack((xv,Phi2)) - H) @ be1, Phi))
    v2   = H2.var(axis=0, ddof=1)
    m2   = v2 > (0.03*np.max(v2[2:]) if len(v2)>2 else 0.0)
    be   = np.zeros(H2.shape[1])
    if m2.any():
        s, *_ = np.linalg.lstsq(H2[:,m2], y, rcond=None)
        be[m2] = s
    return be[2:], be[0], mu, bw   # h, b, mu, bw

def tsplm_phi_init(x_b, y_b, Q_b, etaH_b, kn=12):
    """Use burn-in data to find phi via TSplm utility maximization."""
    D_b   = (Q_b > 0.0).astype(float)   # use phi=0 for labelling during burn-in
    iTr   = np.where(D_b == 1)[0]
    iCon  = np.where(D_b == 0)[0]
    if len(iTr) < 20 or len(iCon) < 20:
        return 0.0

    hTr, bTr, muTr, bwTr   = TSplm(y_b[iTr],  x_b[iTr],  etaH_b[iTr],  kn)
    hCon, bCon, muCon, bwCon = TSplm(y_b[iCon], x_b[iCon], etaH_b[iCon], kn)

    # alpha(eta) = h_Tr(eta) - h_Con(eta) for each obs
    def alpha_tsplm(eta_vals):
        dmt = norm.pdf((eta_vals[:,None] - muTr[None,:]),  scale=bwTr)
        dmc = norm.pdf((eta_vals[:,None] - muCon[None,:]), scale=bwCon)
        return dmt @ hTr - dmc @ hCon

    # Utility grid search: U(phi) = mean[(alpha(eta)-C) * 1{Q>phi}]
    th     = np.linspace(-3.0, 3.0, 80)
    util   = np.array([np.mean((alpha_tsplm(etaH_b) - C) * (Q_b > phi))
                       for phi in th])
    best   = th[np.argmax(util)]
    print(f"  TSplm offline phi_init = {best:.4f}")
    return best

# ── Hermite sieve helpers ──────────────────────────────────────────────────────
def get_k(t):
    return K0 + int((t + 1) ** 0.2)

def expand_rls(theta, P, k_old, k_new):
    add = k_new - k_old
    if add == 0:
        return theta, P
    base_end  = 1 + k_old
    new_theta = np.concatenate([theta[:base_end], np.zeros(add),
                                 theta[base_end:],  np.zeros(add)])
    d_old = len(theta); d_new = len(new_theta)
    P_new = np.zeros((d_new, d_new))
    oi = list(range(base_end)) + list(range(base_end, d_old))
    ni = list(range(base_end)) + list(range(base_end + add, d_new))
    for io, in_ in zip(oi, ni):
        for jo, jn in zip(oi, ni):
            P_new[in_, jn] = P[io, jo]
    for i in range(base_end, base_end + add):            P_new[i,i] = 1e4
    for i in range(base_end + add + k_old, d_new):       P_new[i,i] = 1e4
    return new_theta, P_new

def build_z(x_t, eta_t, D_t, k):
    hb = hermevander(np.array([eta_t]), k-1)[0]
    return np.concatenate([[x_t], hb, D_t * hb])

def alpha_hat_vec(theta, eta_arr, k):
    return hermevander(eta_arr, k-1) @ theta[1+k : 1+2*k]

def rls_s(P, g, xt, qt):
    K = P*xt / (1 + xt*P*xt)
    return g + K*(qt - xt*g), P - K*xt*P

def rls_v(P, p, z, r):
    Pz = P@z; K = Pz / (1 + z@Pz)
    return p + K*(r - z@p), P - np.outer(K, z@P)

def uprime_kde(phi, theta, k, hist, epool):
    T_s = np.asarray(hist[-KDE_WINDOW:])
    if len(T_s) < 10: return 0.0
    h   = 1.06 * max(np.std(T_s), 1e-6) * len(T_s)**(-0.2)
    eta_arr = np.asarray(epool[-KDE_WINDOW:])
    u   = phi - eta_arr
    kde = np.mean(norm.pdf((u[:,None] - T_s[None,:])/h)/h, axis=1)
    return -np.mean((alpha_hat_vec(theta, eta_arr, k) - C) * kde)

# ── Plot snapshot function ─────────────────────────────────────────────────────
results_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
eta_grid         = np.linspace(-3, 3, 200)
true_alpha_curve = 2.0 * np.exp(-(eta_grid - 0.75)**2)
true_b_curve     = np.exp(-0.5 * (eta_grid - 0.5)**2)

def save_snapshot(t_snap, phi_snap, phi_path_s, Uprime_path_s, k_path_s, theta_s, k_s):
    est_alpha = alpha_hat_vec(theta_s, eta_grid, k_s)
    est_b     = hermevander(eta_grid, k_s-1) @ theta_s[1 : 1+k_s]
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(phi_path_s[:t_snap+1], color='C0', lw=0.8, label=r'$\hat\phi_t$')
    ax0.axhline(phi_star, ls='--', color='red', lw=1.5, label=f'phi*={phi_star:.4f}')
    ax0.axvline(m0, color='orange', ls=':', lw=1.5, label=f'TSplm init @ m0={m0}')
    ax0.set_title(f'Exp13 t={t_snap+1:,} | phi={phi_snap:.4f}  phi*={phi_star:.4f}')
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax1 = fig.add_subplot(gs[1, 0])
    valid = ~np.isnan(Uprime_path_s[:t_snap+1])
    ax1.plot(np.where(valid)[0], Uprime_path_s[:t_snap+1][valid], color='C1', lw=0.5, alpha=0.8)
    ax1.axhline(0, ls='--', color='black', alpha=0.5)
    ax1.axvline(m0, color='orange', ls=':')
    ax1.set_title("Gradient U'(phi)"); ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(k_path_s[:t_snap+1], color='C3', lw=1.0)
    ax2.set_title('Hermite bases k_t'); ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.plot(eta_grid, true_alpha_curve, 'r--', lw=2, label='True: 2*exp(-(eta-0.75)^2)')
    ax3.plot(eta_grid, est_alpha, color='C0', lw=1.5, label=f'Sieve alpha (k={k_s})')
    ax3.set_title('alpha(eta): true vs sieve estimate')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.plot(eta_grid, true_alpha_curve, 'r--', lw=2, label='True alpha(eta) = 2*exp(-(eta-0.75)^2)')
    ax4.plot(eta_grid, est_alpha, 'C0', lw=1.5, label=f'Est alpha (k={k_s})')
    err_a = np.abs(est_alpha - true_alpha_curve)
    ax4.fill_between(eta_grid, est_alpha-err_a, est_alpha+err_a, alpha=0.15, color='C0')
    ax4.set_title(f'alpha(eta) | max_err={err_a.max():.4f}  mean_err={err_a.mean():.4f}')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    ax5 = fig.add_subplot(gs[2, 2:])
    ax5.plot(eta_grid, true_b_curve, 'r--', lw=2, label='True b(eta) = exp(-0.5*(eta-0.5)^2)')
    ax5.plot(eta_grid, est_b, 'C1', lw=1.5, label=f'Est b (k={k_s})')
    err_b = np.abs(est_b - true_b_curve)
    ax5.fill_between(eta_grid, est_b-err_b, est_b+err_b, alpha=0.15, color='C1')
    ax5.set_title(f'b(eta) | max_err={err_b.max():.4f}  mean_err={err_b.mean():.4f}')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.text(0.99, 0.005, ts_str, ha='right', va='bottom', fontsize=7, color='gray')
    fname = os.path.join(results_dir, f'exp13_invU_t{t_snap+1:07d}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [snapshot] t={t_snap+1:,} -> {fname}")

# ── Init ───────────────────────────────────────────────────────────────────────
k_cur      = get_k(0)
gamma_hat  = 0.0;  P_gamma = 1e4
theta_hat  = np.zeros(1 + 2*k_cur);  P_theta = np.eye(1+2*k_cur) * 1e4
T_hat_hist = [];  eta_pool = []
phi_hat    = phi_init

# storage for burn-in data (for TSplm)
x_buf = np.zeros(m0); y_buf = np.zeros(m0)
Q_buf = np.zeros(m0); eH_buf = np.zeros(m0)

phi_path    = np.full(n, np.nan)
Uprime_path = np.full(n, np.nan)
k_path      = np.zeros(n, dtype=int)
true_alpha_0 = 2.0 * np.exp(-(0.0 - 0.75)**2)   # alpha(0) = 2*exp(-0.5625)

# ── Main loop ──────────────────────────────────────────────────────────────────
for t in range(n):
    k_new = get_k(t)
    if k_new > k_cur:
        save_snapshot(t, phi_hat, phi_path.copy(), Uprime_path.copy(),
                      k_path.copy(), theta_hat.copy(), k_cur)
        theta_hat, P_theta = expand_rls(theta_hat, P_theta, k_cur, k_new)
        k_cur = k_new

    gamma_hat, P_gamma = rls_s(P_gamma, gamma_hat, x[t], Q[t])
    eta_hat_t = Q[t] - gamma_hat * x[t]
    T_hat_hist.append(gamma_hat * x[t])
    eta_pool.append(eta_hat_t)

    Dt = float(Q[t] > phi_hat)
    yt = Dt*w[t] + beta*x[t] + nu[t]   # nu already includes b(eta)
    zt = build_z(x[t], eta_hat_t, Dt, k_cur)
    theta_hat, P_theta = rls_v(P_theta, theta_hat, zt, yt)

    # buffer burn-in data for TSplm
    if t < m0:
        x_buf[t] = x[t]; y_buf[t] = yt
        Q_buf[t] = Q[t]; eH_buf[t] = eta_hat_t

    # at end of burn-in: offline TSplm phi initialization
    if t == m0 - 1:
        phi_hat = tsplm_phi_init(x_buf, y_buf, Q_buf, eH_buf)

    phi_path[t] = phi_hat
    k_path[t]   = k_cur

    if t >= m0 and t % GRAD_FREQ == 0:
        Up      = uprime_kde(phi_hat, theta_hat, k_cur, T_hat_hist, eta_pool)
        Uprime_path[t] = Up
        lam_t   = lambda0 / np.sqrt((t - m0)//GRAD_FREQ + 1)
        phi_hat = np.clip(phi_hat + lam_t*Up, -phi_clip, phi_clip)

    if (t+1) % 5000 == 0:
        ae = abs(alpha_hat_vec(theta_hat, np.array([0.0]), k_cur)[0] - true_alpha_0)
        print(f"  [{100*(t+1)/n:4.1f}%] t={t+1} k={k_cur} | "
              f"phi={phi_hat:.4f} | alpha_err@0={ae:.4f} | gamma={gamma_hat:.4f}")

    if (t+1) % PLOT_FREQ == 0:
        save_snapshot(t, phi_hat, phi_path.copy(), Uprime_path.copy(),
                      k_path.copy(), theta_hat.copy(), k_cur)

print(f"=== Exp13 === phi*={phi_star:.4f}  Final phi_hat={phi_hat:.6f}")
