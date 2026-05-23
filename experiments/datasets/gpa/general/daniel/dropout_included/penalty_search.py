"""
Search for a per-capita treatment cost c that yields an interior optimal threshold.

Uses the dropout-inclusive PLM (nextGPA = 0 - GPA_year1 for dropouts) with ridge
regularization, then sweeps over c values in the utility:

    U(phi) = E[alpha(eta) * P(Q < phi | eta)] - c * P(Q < phi)

The first term is the total GPA benefit (with dropout harm internalized).
The second term is the per-capita cost of putting someone on probation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpa_analysis import _basis_params, _eval_basis, X_COLS, Q_COL, Y_COL

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "Dep_Data", "final_processed_data.csv",
)
OUTDIR = os.path.dirname(os.path.abspath(__file__))


def estimate():
    """Run the dropout-inclusive ridge PLM. Returns estimation objects."""
    df = pd.read_csv(DATA_PATH)

    # Impute nextGPA = 0 - GPA_year1 for dropouts
    missing_y = df[Y_COL].isna()
    left = df["left_school"] == 1
    impute_mask = missing_y & left
    df.loc[impute_mask, Y_COL] = 0.0 - df.loc[impute_mask, "GPA_year1"]
    df = df[df[Y_COL].notna()].copy()
    n = len(df)

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    # First stage
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    # B-spline
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    # Ridge PLM
    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)

    omega_treat = be[p + n_basis:]

    # Eval region (trimmed)
    eval_lo = max(support[0], np.percentile(eta_Tr, 5))
    eval_hi = min(support[1], np.percentile(eta_Tr, 95))

    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[in_eval], info)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)

    # Objects needed for utility computation
    gX_vals = q - eta_hat
    gX_sorted = np.sort(gX_vals)

    in_eval_all = (eta_hat >= eval_lo) & (eta_hat <= eval_hi)
    eta_eval_obs = eta_hat[in_eval_all]
    Phi_eval_obs = _eval_basis(eta_eval_obs, info)
    alpha_eval_obs = Phi_eval_obs @ omega_treat

    return {
        "alpha_hat": alpha_hat,
        "gX_sorted": gX_sorted,
        "eta_eval_obs": eta_eval_obs,
        "alpha_eval_obs": alpha_eval_obs,
        "n": n,
        "n_tr": n_tr,
    }


def compute_utility(est, c, phi_grid):
    """Compute U(phi) = E[alpha(eta)*P(Q<phi|eta)] - c*P(Q<phi) on a grid."""
    gX_sorted = est["gX_sorted"]
    eta_eval_obs = est["eta_eval_obs"]
    alpha_eval_obs = est["alpha_eval_obs"]
    n_total = len(gX_sorted)

    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_eval_obs
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        benefit = np.mean(alpha_eval_obs * probs)
        cost = c * np.mean(probs)
        util[j] = benefit - cost

    return util


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Estimate once ----
est = estimate()
print(f"Plug-in avg alpha: {est['alpha_hat']:.4f}")
print(f"N = {est['n']}, N_treated = {est['n_tr']}")

phi_grid = np.linspace(-3.0, 3.0, 600)

# ---- Sweep cost values ----
c_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print(f"\n{'c':>6s}  {'phi*':>8s}  {'U(phi*)':>10s}  {'U(0)':>10s}  {'P(treat at phi*)':>16s}")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for c in c_vals:
    util = compute_utility(est, c, phi_grid)
    opt_idx = np.argmax(util)
    opt_phi = phi_grid[opt_idx]
    u_at_zero = np.interp(0.0, phi_grid, util)

    # Fraction treated at optimal phi
    frac_treated = np.searchsorted(est["gX_sorted"],
                                    opt_phi - est["eta_eval_obs"]) / len(est["gX_sorted"])
    avg_frac = np.mean(frac_treated)

    print(f"{c:6.3f}  {opt_phi:8.3f}  {util[opt_idx]:10.4f}  {u_at_zero:10.4f}  {avg_frac:16.3f}")

    axes[0].plot(phi_grid, util, linewidth=1.5,
                 label=f"c={c:.2f}, phi*={opt_phi:.2f}")

axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(0, color="gray", linestyle=":", linewidth=0.5)
axes[0].set_xlabel(r"Threshold $\phi$")
axes[0].set_ylabel("Utility")
axes[0].set_title("Utility curves for different per-capita costs c")
axes[0].legend(fontsize=8)

# ---- Fine sweep: phi*(c) ----
c_fine = np.linspace(0.0, 0.40, 100)
opt_phis = np.zeros(len(c_fine))
opt_utils = np.zeros(len(c_fine))
for i, c in enumerate(c_fine):
    util = compute_utility(est, c, phi_grid)
    idx = np.argmax(util)
    opt_phis[i] = phi_grid[idx]
    opt_utils[i] = util[idx]

axes[1].plot(c_fine, opt_phis, "b.-", markersize=3)
axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[1].set_xlabel("Per-capita cost c")
axes[1].set_ylabel(r"Optimal threshold $\phi^*$")
axes[1].set_title(r"Optimal $\phi^*$ as a function of cost c")

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "cost_search_diagnostic.png"), dpi=150)
plt.close()
print("\nSaved cost_search_diagnostic.png")
