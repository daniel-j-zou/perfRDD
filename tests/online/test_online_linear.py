"""
test_online_linear.py

Test the online threshold learning algorithm from online.tex, simple linear case.

Model:
    Q = gamma * X + eta
    D = 1{Q > phi_t}
    Y = D*(alpha_1 + delta_1*eta) + beta*X + (alpha_0 + delta_0*eta) + epsilon

Algorithm (per observation):
    1. RLS for gamma
    2. RLS for theta = (beta, delta_0, alpha_0, delta_1, alpha_1)
    3. Plug-in g(u) = N(0, gamma_hat^2) density (known X distribution)
    4. Gradient ascent: phi_{t+1} = phi_t + lambda_t * U'_hat(phi_t)

True phi*:
    phi* = -(alpha_1-c)*(1+gamma^2)/delta_1
"""

import numpy as np
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════

GAMMA = 1.0
BETA = 1.0
ALPHA_0 = 0.0
DELTA_0 = 0.3
ALPHA_1 = 1.2
DELTA_1 = 0.5
SIGMA_EPS = 0.5
C = 1.0

PHI_STAR = -(ALPHA_1 - C) * (1 + GAMMA**2) / DELTA_1
PHI_INIT = 0.0


# ═══════════════════════════════════════════════════════════════
# DGP
# ═══════════════════════════════════════════════════════════════

def draw_one(phi, rng):
    x = rng.normal(0, 1)
    eta = rng.normal(0, 1)
    q = GAMMA * x + eta
    D = float(q > phi)
    w = ALPHA_1 + DELTA_1 * eta + rng.normal(0, 0.01)
    b = ALPHA_0 + DELTA_0 * eta
    eps = rng.normal(0, SIGMA_EPS)
    y = D * w + BETA * x + b + eps
    return x, q, eta, D, y


# ═══════════════════════════════════════════════════════════════
# Online algorithm
# ═══════════════════════════════════════════════════════════════

def compute_U_prime_plugin(phi, gamma_hat, alpha_1_hat, delta_1_hat):
    """
    U'(phi) = -integral (alpha_hat(eta) - c) * g_hat(phi - eta) * f_eta(eta) deta

    With alpha_hat(eta) = alpha_1_hat + delta_1_hat*eta, g_hat = N(0, gamma_hat^2),
    f_eta = N(0,1), this has a closed form:

    U'(phi) = -[(alpha_1_hat-c) + delta_1_hat * phi / (1+gamma_hat^2)]
              * phi_{sqrt(1+gamma_hat^2)}(phi)
    """
    sig2 = 1.0 + gamma_hat**2
    sig = np.sqrt(sig2)
    bracket = (alpha_1_hat - C) + delta_1_hat * phi / sig2
    return -bracket * sp_norm.pdf(phi, 0, sig)


def run_one(T, seed, step_schedule="sqrt", step_const=0.5):
    """Run one trajectory of T observations. Return phi and gamma histories."""
    rng = np.random.default_rng(seed)
    phi = PHI_INIT
    gamma_hat = 0.0
    P_gamma = 100.0
    theta_hat = np.zeros(5)
    P_theta = 100.0 * np.eye(5)

    phi_hist = np.empty(T + 1)
    gamma_hist = np.empty(T + 1)
    theta_hist = np.empty((T + 1, 5))
    phi_hist[0] = phi
    gamma_hist[0] = gamma_hat
    theta_hist[0] = theta_hat

    for t in range(1, T + 1):
        x, q, eta, D_true, y = draw_one(phi, rng)

        # Step 1: RLS for gamma
        k = P_gamma * x / (1 + x * P_gamma * x)
        gamma_hat += k * (q - x * gamma_hat)
        P_gamma -= k * x * P_gamma
        eta_hat = q - x * gamma_hat

        # Step 2: RLS for theta
        D = float(q > phi)
        z = np.array([x, eta_hat, 1.0, D * eta_hat, D])
        Pz = P_theta @ z
        k_th = Pz / (1 + z @ Pz)
        theta_hat = theta_hat + k_th * (y - z @ theta_hat)
        P_theta = P_theta - np.outer(k_th, z @ P_theta)

        # Step 3+4: gradient update for phi
        if t >= 30:
            alpha_1_hat = theta_hat[4]
            delta_1_hat = theta_hat[3]
            Up = compute_U_prime_plugin(phi, gamma_hat, alpha_1_hat, delta_1_hat)

            if step_schedule == "sqrt":
                lam = step_const / np.sqrt(t)
            elif step_schedule == "linear":
                lam = step_const / t
            elif step_schedule == "pow23":
                lam = step_const / t ** (2/3)
            else:
                lam = step_const / np.sqrt(t)
            phi += lam * Up

        phi_hist[t] = phi
        gamma_hist[t] = gamma_hat
        theta_hist[t] = theta_hat

    return phi_hist, gamma_hist, theta_hist


def run_mc(T, R, **kwargs):
    """Run R trajectories. Return (R, T+1) arrays."""
    phis = np.empty((R, T + 1))
    gammas = np.empty((R, T + 1))
    for r in range(R):
        phis[r], gammas[r], _ = run_one(T, seed=r, **kwargs)
        if (r + 1) % 100 == 0:
            print(f"    rep {r+1}/{R}", flush=True)
    return phis, gammas


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_results(phis, gammas, T, R, tag=""):
    ts = np.arange(T + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Online Threshold Learning — Linear Case "
                 f"(T={T:,}, R={R}, $\\phi^*={PHI_STAR:.3f}$)"
                 f"{' — '+tag if tag else ''}", fontsize=13)

    # (a) Trajectories
    ax = axes[0, 0]
    for r in range(min(R, 30)):
        ax.plot(ts, phis[r], alpha=0.2, linewidth=0.5)
    ax.plot(ts, np.mean(phis, axis=0), "k-", linewidth=2, label="Mean")
    ax.axhline(PHI_STAR, color="red", linewidth=2, linestyle="--",
               label=f"$\\phi^* = {PHI_STAR:.3f}$")
    ax.set_xlabel("t"); ax.set_ylabel("$\\hat\\phi_t$")
    ax.set_title("Trajectories"); ax.legend()

    # (b) Convergence rate (log-log)
    ax = axes[0, 1]
    log_ts = np.unique(np.logspace(1, np.log10(T), 300).astype(int))
    mean_err = np.mean(np.abs(phis - PHI_STAR), axis=0)
    ax.loglog(log_ts, mean_err[log_ts], "b-", linewidth=1.5,
              label="Mean $|\\hat\\phi_t - \\phi^*|$")
    ax.loglog(log_ts, mean_err[100] * np.sqrt(100) / np.sqrt(log_ts), "r--",
              linewidth=1, label="$O(t^{-1/2})$")
    ax.loglog(log_ts, mean_err[100] * 100**0.33 / log_ts**0.33, "k:",
              linewidth=1, label="$O(t^{-1/3})$")
    ax.set_xlabel("t"); ax.set_ylabel("$|\\hat\\phi_t - \\phi^*|$")
    ax.set_title("Convergence rate"); ax.legend(fontsize=8)

    # (c) Scaled MSE
    ax = axes[1, 0]
    mse = np.mean((phis - PHI_STAR)**2, axis=0)
    ax.plot(ts[200:], ts[200:] * mse[200:], "b-", linewidth=1)
    ax.set_xlabel("t"); ax.set_ylabel("$t \\cdot \\mathrm{MSE}$")
    ax.set_title("Scaled MSE (stabilizes $\\Rightarrow$ rate $t^{-1}$)")

    # (d) Gamma convergence
    ax = axes[1, 1]
    gamma_err = np.mean(np.abs(gammas - GAMMA), axis=0)
    ax.loglog(log_ts, gamma_err[log_ts], "b-", linewidth=1.5, label="$|\\hat\\gamma_t - \\gamma|$")
    ax.loglog(log_ts, gamma_err[100] * np.sqrt(100) / np.sqrt(log_ts), "r--",
              linewidth=1, label="$O(t^{-1/2})$")
    ax.set_xlabel("t"); ax.set_title("First-stage convergence"); ax.legend(fontsize=8)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, f"online_convergence{'_'+tag if tag else ''}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_step_comparison(results_dict, T):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Step Size Comparison (T={T:,}, $\\phi^*={PHI_STAR:.3f}$)", fontsize=13)
    log_ts = np.unique(np.logspace(1, np.log10(T), 300).astype(int))
    colors = {"sqrt": "blue", "linear": "red", "pow23": "green"}

    for name, (phis, _) in results_dict.items():
        mse = np.mean((phis - PHI_STAR)**2, axis=0)
        mean_err = np.mean(np.abs(phis - PHI_STAR), axis=0)
        c = colors.get(name, "gray")
        axes[0].loglog(log_ts, mean_err[log_ts], color=c, linewidth=1.5, label=name)
        axes[1].plot(np.arange(len(mse))[200:], np.arange(len(mse))[200:] * mse[200:],
                     color=c, linewidth=1.5, label=name)

    axes[0].loglog(log_ts, 0.5 / np.sqrt(log_ts), "k--", linewidth=1, alpha=0.5, label="$t^{-1/2}$")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("Mean $|\\hat\\phi_t - \\phi^*|$")
    axes[0].set_title("Convergence rate"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("$t \\cdot \\mathrm{MSE}$")
    axes[1].set_title("Scaled MSE"); axes[1].legend(fontsize=8)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "online_step_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Online Threshold Learning — Linear Case")
    print("=" * 60)
    print(f"  alpha(eta) = {ALPHA_1} + {DELTA_1}*eta")
    print(f"  b(eta)     = {ALPHA_0} + {DELTA_0}*eta")
    print(f"  gamma={GAMMA}, beta={BETA}, sigma_eps={SIGMA_EPS}, c={C}")
    print(f"  phi*       = {PHI_STAR:.4f}")
    print(f"  phi_init   = {PHI_INIT}")

    T = 50000
    R = 500

    results = {}
    for schedule in ["sqrt", "linear", "pow23"]:
        print(f"\n  Schedule: {schedule}")
        t0 = time.time()
        phis, gammas = run_mc(T, R, step_schedule=schedule, step_const=0.5)
        elapsed = time.time() - t0

        final_mse = np.mean((phis[:, -1] - PHI_STAR)**2)
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Final MSE: {final_mse:.6f}, t*MSE: {T*final_mse:.2f}")
        results[schedule] = (phis, gammas)

    # Best schedule for detailed plot
    best = min(results, key=lambda k: np.mean((results[k][0][:, -1] - PHI_STAR)**2))
    print(f"\nBest: {best}")
    plot_results(*results[best], T, R, tag=best)
    plot_step_comparison(results, T)

    # Convergence table
    checkpoints = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    checkpoints = [c for c in checkpoints if c <= T]
    print(f"\n{'t':>8s}", end="")
    for s in results:
        print(f"  {'MSE_'+s:>12s} {'t*MSE':>8s}", end="")
    print()
    print("-" * (8 + 22 * len(results)))
    for tc in checkpoints:
        print(f"{tc:>8d}", end="")
        for s in results:
            mse = np.mean((results[s][0][:, tc] - PHI_STAR)**2)
            print(f"  {mse:>12.6f} {tc*mse:>8.2f}", end="")
        print()


if __name__ == "__main__":
    main()
