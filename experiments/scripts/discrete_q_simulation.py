"""Discrete-Q + continuous-surrogate simulation.

DGP (realistic version, with disjoint X-loadings and surrogate noise):
  X = (X_a, X_b, X_c) in R^d, d = 3 * d_block
  eta ~ N(0, sigma_eta^2), eta indep X
  Q_tilde = X_a^T delta_a + eta + e1                       (continuous surrogate)
  S       = X_b^T delta_b + lambda * eta + u, u ~ Logistic (latent score)
  Q       = bin(S) in {1,...,J}                            (discrete observed)
  T       = 1{Q > k0}                                      (treatment)
  Y       = alpha_0 * T + X_c^T beta_c + X_b^T beta_b + b(eta) + eps
  with b(eta) = eta + 0.5*eta^2 (nonlinear confounding)

The DGP satisfies: same eta drives Q_tilde and Q; Q has independent logistic
noise so Q is NOT a deterministic function of Q_tilde; surrogate noise e1
means even eta_hat = Q_tilde - X_a^T delta_a has irreducible noise.

Estimators compared (each targets alpha_0, the coefficient on T = 1{Q>k0}):
  1. Oracle PLM:        knows eta, regresses Y on [1, X, spline(eta), T]
  2. Proposed (stacked): eta_hat from OLS on Q_tilde, then PLM on (T, X, spline(eta_hat))
  3. Naive OLS:         Y on (1, X, T), no eta adjustment
  4. Continuous-perfrdd analog: targets effect of T' = 1{Q_tilde > phi} via PLM
                                with spline(Q_tilde). This is a DIFFERENT estimand;
                                its truth is computed separately by large-sample regression.

Output: JSON with per-seed estimates; PNG with bias / RMSE / CI coverage by n.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "runs" / "discrete_q_simulation.json"
OUT_FIG  = ROOT / "runs" / "discrete_q_simulation.png"


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

DGP = dict(
    d_block            = 2,        # X has 3 blocks of size d_block each
    sigma_eta          = 1.0,
    lambda_eta_in_S    = 1.5,      # eta's contribution to S
    sigma_eps          = 0.5,
    alpha_0            = 1.0,
    k0                 = 2,        # treatment is 1{Q > k0}
    J                  = 5,        # number of discrete categories
    beta_b_scale       = 0.3,
    beta_c_scale       = 0.5,
)
# Q_tilde has NO measurement noise: Q_tilde = X_a^T delta_a + eta.
# The "residual" of Q_tilde after projecting out X_a IS eta -- that's what we
# want to recover.  All the conditional-on-(X,eta) randomness lives in Q,
# through the logistic noise u in the latent score S.


def generate(n: int, seed: int, params=DGP) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_block = params['d_block']
    d = 3 * d_block

    X   = rng.standard_normal((n, d))
    X_a = X[:, 0:d_block]
    X_b = X[:, d_block:2*d_block]
    X_c = X[:, 2*d_block:3*d_block]

    eta = rng.normal(0.0, params['sigma_eta'], n)
    delta_a = np.ones(d_block) / np.sqrt(d_block)
    delta_b = np.ones(d_block) / np.sqrt(d_block)

    Q_tilde = X_a @ delta_a + eta             # NO extra measurement noise: eta IS the residual

    u = rng.logistic(0.0, 1.0, n)
    S = X_b @ delta_b + params['lambda_eta_in_S'] * eta + u

    tau = np.quantile(S, np.linspace(0, 1, params['J'] + 1))[1:-1]
    Q = np.digitize(S, tau) + 1     # in {1, ..., J}
    T = (Q > params['k0']).astype(int)

    beta_b = np.full(d_block, params['beta_b_scale'])
    beta_c = np.full(d_block, params['beta_c_scale'])
    b_eta = eta + 0.5 * eta**2
    eps = rng.normal(0.0, params['sigma_eps'], n)
    Y = params['alpha_0'] * T + X_c @ beta_c + X_b @ beta_b + b_eta + eps

    return dict(
        X=X, X_a=X_a, X_b=X_b, X_c=X_c, eta=eta,
        Q_tilde=Q_tilde, S=S, Q=Q, T=T, Y=Y,
        delta_a=delta_a, delta_b=delta_b, tau=tau,
    )


# ---------------------------------------------------------------------------
# Spline basis and estimators
# ---------------------------------------------------------------------------

def cubic_bspline_basis(x: np.ndarray, n_knots: int) -> np.ndarray:
    """Cubic B-spline basis with n_knots interior knots placed at quantiles of x."""
    x_lo, x_hi = float(x.min()), float(x.max())
    pad = 1e-6 * max(1.0, x_hi - x_lo)
    x_lo -= pad; x_hi += pad
    if n_knots > 0:
        interior = np.quantile(x, np.linspace(0, 1, n_knots + 2)[1:-1])
    else:
        interior = np.array([])
    degree = 3
    knots = np.concatenate([
        [x_lo] * (degree + 1),
        interior,
        [x_hi] * (degree + 1),
    ])
    n_basis = len(knots) - degree - 1
    B = np.empty((len(x), n_basis))
    xc = np.clip(x, x_lo, x_hi)
    for i in range(n_basis):
        c = np.zeros(n_basis); c[i] = 1.0
        B[:, i] = BSpline(knots, c, degree, extrapolate=False)(xc)
    B = np.nan_to_num(B, nan=0.0)
    return B


def _plm_fit(Y, T, X, eta_arg, n_knots, ridge_scale):
    n = len(Y)
    P = cubic_bspline_basis(eta_arg, n_knots)
    H = np.column_stack([np.ones(n), X, P, T])
    p_pre = 1 + X.shape[1]
    n_b = P.shape[1]
    lam = ridge_scale / np.sqrt(n)
    Pen = np.zeros((H.shape[1], H.shape[1]))
    for i in range(p_pre, p_pre + n_b):
        Pen[i, i] = lam
    A = H.T @ H + n * Pen
    co = np.linalg.solve(A, H.T @ Y)
    resid = Y - H @ co
    sigma2 = float(np.mean(resid**2))
    A_inv = np.linalg.inv(A)
    V = sigma2 * A_inv @ (H.T @ H) @ A_inv      # sandwich on penalized OLS
    se = float(np.sqrt(V[-1, -1]))
    return float(co[-1]), se


def oracle_plm(d, n_knots=3, ridge_scale=0.01):
    return _plm_fit(d['Y'], d['T'], d['X'], d['eta'], n_knots, ridge_scale)


def proposed_plm(d, n_knots=3, ridge_scale=0.01):
    # eta_hat = Q_tilde - X_a^T delta_hat   (OLS)
    n = len(d['Q_tilde'])
    H_s = np.column_stack([np.ones(n), d['X_a']])
    co, *_ = np.linalg.lstsq(H_s, d['Q_tilde'], rcond=None)
    eta_hat = d['Q_tilde'] - H_s @ co
    return _plm_fit(d['Y'], d['T'], d['X'], eta_hat, n_knots, ridge_scale)


def naive_ols(d):
    n = len(d['Y'])
    H = np.column_stack([np.ones(n), d['X'], d['T']])
    co, *_ = np.linalg.lstsq(H, d['Y'], rcond=None)
    resid = d['Y'] - H @ co
    sigma2 = float(np.mean(resid**2))
    A_inv = np.linalg.inv(H.T @ H)
    se = float(np.sqrt(sigma2 * A_inv[-1, -1]))
    return float(co[-1]), se


def continuous_perfrdd(d, q_threshold, n_knots=3, ridge_scale=0.01):
    """Continuous-perfrdd analog: regress Y on spline(Q_tilde) and T' = 1{Q_tilde > phi}."""
    T_prime = (d['Q_tilde'] > q_threshold).astype(int)
    return _plm_fit(d['Y'], T_prime, d['X'], d['Q_tilde'], n_knots, ridge_scale)


# ---------------------------------------------------------------------------
# Truth for the continuous-perfrdd estimand
# ---------------------------------------------------------------------------

def compute_continuous_truth(q_threshold, params=DGP, n_big=200_000, seed=999):
    """The 'true' coefficient on T' = 1{Q_tilde > phi} in the PLM regression
    Y ~ T' + X + spline(eta), computed by large-sample regression."""
    d = generate(n_big, seed=seed, params=params)
    T_prime = (d['Q_tilde'] > q_threshold).astype(int)
    P = cubic_bspline_basis(d['eta'], n_knots=5)
    H = np.column_stack([np.ones(n_big), d['X'], P, T_prime])
    co, *_ = np.linalg.lstsq(H, d['Y'], rcond=None)
    return float(co[-1])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

N_GRID  = [500, 2000, 8000, 32_000]
N_SEEDS = 50


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute the q_threshold (Q_tilde quantile matching k0/J fraction of population)
    pilot = generate(100_000, seed=99_999)
    q_threshold = float(np.quantile(pilot['Q_tilde'], DGP['k0'] / DGP['J']))
    true_cont = compute_continuous_truth(q_threshold, DGP)
    true_alpha = DGP['alpha_0']
    pi_treated = float(np.mean(pilot['T']))
    pi_treated_cont = float(np.mean(pilot['Q_tilde'] > q_threshold))
    print(f"True alpha_0 (discrete tier effect): {true_alpha:.4f}")
    print(f"True continuous-threshold effect (Q_tilde > {q_threshold:.3f}): {true_cont:.4f}")
    print(f"  fraction treated under T = 1{{Q>{DGP['k0']}}}:        {pi_treated:.3f}")
    print(f"  fraction treated under T' = 1{{Q_tilde>{q_threshold:.3f}}}: {pi_treated_cont:.3f}")

    # Compute simulated R^2(Q_tilde, Q) for realism check
    pilot_Q = pilot['Q'].astype(float)
    pilot_Qt = pilot['Q_tilde']
    r2 = float(np.corrcoef(pilot_Qt, pilot_Q)[0, 1]**2)
    print(f"R^2(Q_tilde, Q) in pilot sample: {r2:.3f}")

    results: List[Dict[str, Any]] = []
    for n in N_GRID:
        print(f"\n=== n = {n} ===")
        for seed in range(N_SEEDS):
            d = generate(n, seed=seed)
            row = dict(n=n, seed=seed)
            try:
                a, se = oracle_plm(d);  row['oracle_alpha']=a;  row['oracle_se']=se
            except Exception as e:
                row['oracle_alpha']=np.nan; row['oracle_se']=np.nan
            try:
                a, se = proposed_plm(d); row['proposed_alpha']=a; row['proposed_se']=se
            except Exception as e:
                row['proposed_alpha']=np.nan; row['proposed_se']=np.nan
            try:
                a, se = naive_ols(d);    row['naive_alpha']=a;    row['naive_se']=se
            except Exception as e:
                row['naive_alpha']=np.nan; row['naive_se']=np.nan
            try:
                a, se = continuous_perfrdd(d, q_threshold)
                row['cont_alpha']=a; row['cont_se']=se
            except Exception as e:
                row['cont_alpha']=np.nan; row['cont_se']=np.nan
            results.append(row)
        # Quick log
        arr_oracle = np.array([r['oracle_alpha']  for r in results if r['n']==n])
        arr_prop   = np.array([r['proposed_alpha']for r in results if r['n']==n])
        arr_naive  = np.array([r['naive_alpha']   for r in results if r['n']==n])
        arr_cont   = np.array([r['cont_alpha']    for r in results if r['n']==n])
        print(f"  oracle:   med={np.nanmedian(arr_oracle):.3f}  RMSE_vs_alpha0={np.sqrt(np.nanmean((arr_oracle-true_alpha)**2)):.3f}")
        print(f"  proposed: med={np.nanmedian(arr_prop):.3f}  RMSE_vs_alpha0={np.sqrt(np.nanmean((arr_prop-true_alpha)**2)):.3f}")
        print(f"  naive:    med={np.nanmedian(arr_naive):.3f}  RMSE_vs_alpha0={np.sqrt(np.nanmean((arr_naive-true_alpha)**2)):.3f}")
        print(f"  cont:     med={np.nanmedian(arr_cont):.3f}   (truth for cont = {true_cont:.3f}, RMSE_vs_cont={np.sqrt(np.nanmean((arr_cont-true_cont)**2)):.3f})")

    OUT_JSON.write_text(json.dumps({
        'dgp': DGP,
        'true_alpha_0': true_alpha,
        'true_continuous_effect': true_cont,
        'q_threshold': q_threshold,
        'pi_treated': pi_treated,
        'pi_treated_cont': pi_treated_cont,
        'r2_Qtilde_Q': r2,
        'results': results,
    }, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    estimators_for_alpha = [
        ('oracle',   'Oracle (knows η)',    'C2'),
        ('proposed', 'Proposed (η̂ from Q̃)','C0'),
        ('naive',    'Naive OLS',           'C3'),
    ]
    estimators_for_cont = [
        ('cont',     'Cont. perfrdd on Q̃ (different estimand)', 'C1'),
    ]

    # Panel 1: distribution of α̂ - α_0 by n (boxplots), for the three methods targeting α_0
    ax = axes[0]
    positions = []
    box_data = []
    box_colors = []
    labels_x = []
    width = 0.22
    offset = {'oracle': -width, 'proposed': 0.0, 'naive': width}
    for j, n in enumerate(N_GRID):
        for key, lbl, col in estimators_for_alpha:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)] - true_alpha
            box_data.append(arr)
            positions.append(j + offset[key])
            box_colors.append(col)
        labels_x.append(f"n={n}")
    bp = ax.boxplot(box_data, positions=positions, widths=width*0.95,
                    patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xticks(range(len(N_GRID)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel(r"$\hat\alpha - \alpha_0$")
    ax.set_title("Estimator distribution around true $\\alpha_0=1$ (discrete tier effect)")
    handles = [plt.Rectangle((0,0),1,1, fc=col, alpha=0.5, label=lbl) for _,lbl,col in estimators_for_alpha]
    ax.legend(handles=handles, fontsize=9, loc='best')
    ax.grid(alpha=0.25)

    # Panel 2: RMSE vs n (log-log) for the three methods + continuous estimator
    ax = axes[1]
    for key, lbl, col in estimators_for_alpha:
        rmse = []
        for n in N_GRID:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)]
            rmse.append(float(np.sqrt(np.mean((arr - true_alpha)**2))))
        ax.loglog(N_GRID, rmse, 'o-', color=col, lw=2, ms=8, label=lbl)
    # Continuous (RMSE vs its own truth)
    for key, lbl, col in estimators_for_cont:
        rmse = []
        for n in N_GRID:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)]
            rmse.append(float(np.sqrt(np.mean((arr - true_cont)**2))))
        ax.loglog(N_GRID, rmse, 's--', color=col, lw=2, ms=8, label=lbl + f" (vs its truth {true_cont:.3f})")
    # Reference slope -1/2
    n0 = N_GRID[0]; r0 = 0.3
    ax.loglog(N_GRID, [r0 * np.sqrt(n0/n) for n in N_GRID], 'k:', lw=1, label=r'$n^{-1/2}$ reference')
    ax.set_xlabel("n"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs n  (slope $-1/2$ ⇒ $\\sqrt{n}$ rate)")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.25, which='both')

    # Panel 3: bias bar chart
    ax = axes[2]
    bar_w = 0.2
    xs = np.arange(len(N_GRID))
    for idx, (key, lbl, col) in enumerate(estimators_for_alpha):
        bias = []
        for n in N_GRID:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)]
            bias.append(float(np.mean(arr) - true_alpha))
        ax.bar(xs + (idx-1)*bar_w, bias, bar_w, color=col, alpha=0.7, label=lbl)
    # Continuous estimator's bias relative to alpha_0 (showing it targets a different number)
    bias_c = []
    for n in N_GRID:
        arr = np.array([r['cont_alpha'] for r in results if r['n']==n])
        arr = arr[~np.isnan(arr)]
        bias_c.append(float(np.mean(arr) - true_alpha))
    ax.bar(xs + 2*bar_w, bias_c, bar_w, color='C1', alpha=0.7,
           label=f"Cont. perfrdd (rel. $\\alpha_0$; targets {true_cont:.3f})")
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"n={n}" for n in N_GRID])
    ax.set_ylabel(r"mean($\hat\alpha$) - $\alpha_0$")
    ax.set_title("Bias relative to discrete tier effect $\\alpha_0=1$")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Discrete-Q + continuous-surrogate simulation, {N_SEEDS} seeds/n.  "
        f"$\\tilde Q = X_a^\\top\\delta_a + \\eta$ (no measurement noise),  "
        f"$R^2(\\tilde Q, Q)$={r2:.2f},  $\\lambda$={DGP['lambda_eta_in_S']},  "
        f"J={DGP['J']},  $k_0$={DGP['k0']}",
        fontsize=11, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
