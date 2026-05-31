"""Discrete-Q simulation, EIV-amplified DGP.

Companion to discrete_q_simulation.py. Same four estimators and figure
layout (oracle, proposed, naive, continuous-perfRDD on Q_tilde), but with
the DGP adjusted to make the EIV loading A_{α,η} = E[ξ X r_α] non-negligible:

  - b(η) = exp(η/2) (vs. η + 0.5 η² in the standard DGP). Gives b'(η) =
    0.5 exp(η/2) which varies by ~20× across the support of η, increasing
    |ξ| and hence |A_{α,η}|. (Mirrors the e^{η/2} row of Table 5 in the
    continuous-Q empirical section, which produced the largest VarRatio
    deviations from 1.00.)

  - δ_b amplified from 1.0·1_2/√2 to 1.5·1_2/√2. Makes the X-driven and
    η-driven contributions to the latent score S = X_b^T δ_b + λη + u
    balanced (Var(X_b^T δ_b) = 1.5² = 2.25 ≈ λ²·σ_η² = 2.25), maximizing
    the within-treatment-group correlation Cor(X_b^T δ_b, η̂ | T=1).

Per the Section 5 empirical analysis: these are the two regimes where the
EIV correction visibly affects the Robinson/Naive variance ratio. We
expect to see RMSE(proposed) > RMSE(oracle) here, in contrast to the
standard-DGP result where they match to three decimal places.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.scripts.discrete_q_simulation import (
    cubic_bspline_basis, _plm_fit, naive_ols,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "runs" / "discrete_q_simulation_amplified.json"
OUT_FIG  = ROOT / "runs" / "discrete_q_simulation_amplified.png"


DGP = dict(
    d_block            = 2,
    sigma_eta          = 1.0,
    lambda_eta_in_S    = 1.5,
    sigma_eps          = 0.5,
    alpha_0            = 1.0,
    k0                 = 2,
    J                  = 5,
    delta_a_norm       = 1.0,    # |delta_a| -- standard
    delta_b_norm       = 3.0,    # |delta_b| -- AMPLIFIED for strong selection (was 1.0 in standard DGP)
    beta_b_scale       = 0.3,
    beta_c_scale       = 0.5,
)
# E[b'(eta)^2] for amplification target:
#   standard b(eta) = eta + 0.5 eta^2: b'(eta) = 1 + eta, E[b'^2] = 2.0
#   amplified b(eta) = exp(eta):       b'(eta) = exp(eta), E[b'^2] = e^2 ~ 7.4
# So the amplified DGP genuinely increases the magnitude of A_{alpha,eta} loading.


def generate(n: int, seed: int, params=DGP) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_block = params['d_block']
    d = 3 * d_block

    X   = rng.standard_normal((n, d))
    X_a = X[:, 0:d_block]
    X_b = X[:, d_block:2*d_block]
    X_c = X[:, 2*d_block:3*d_block]

    eta = rng.normal(0.0, params['sigma_eta'], n)
    delta_a = params['delta_a_norm'] * np.ones(d_block) / np.sqrt(d_block)
    delta_b = params['delta_b_norm'] * np.ones(d_block) / np.sqrt(d_block)

    Q_tilde = X_a @ delta_a + eta             # surrogate, no noise: eta is the residual

    u = rng.logistic(0.0, 1.0, n)
    S = X_b @ delta_b + params['lambda_eta_in_S'] * eta + u

    tau = np.quantile(S, np.linspace(0, 1, params['J'] + 1))[1:-1]
    Q = np.digitize(S, tau) + 1
    T = (Q > params['k0']).astype(int)

    beta_b = np.full(d_block, params['beta_b_scale'])
    beta_c = np.full(d_block, params['beta_c_scale'])
    # AMPLIFIED b(eta): exp(eta) instead of eta + 0.5*eta^2.
    # E[b'(eta)^2] = E[exp(2 eta)] = e^2 ~ 7.4 (vs. 2.0 for the standard DGP).
    b_eta = np.exp(eta)
    eps = rng.normal(0.0, params['sigma_eps'], n)
    Y = params['alpha_0'] * T + X_c @ beta_c + X_b @ beta_b + b_eta + eps

    return dict(
        X=X, X_a=X_a, X_b=X_b, X_c=X_c, eta=eta,
        Q_tilde=Q_tilde, S=S, Q=Q, T=T, Y=Y,
        delta_a=delta_a, delta_b=delta_b, tau=tau,
    )


def oracle_plm(d, n_knots=3, ridge_scale=0.01):
    return _plm_fit(d['Y'], d['T'], d['X'], d['eta'], n_knots, ridge_scale)


def proposed_plm(d, n_knots=3, ridge_scale=0.01):
    n = len(d['Q_tilde'])
    H_s = np.column_stack([np.ones(n), d['X_a']])
    co, *_ = np.linalg.lstsq(H_s, d['Q_tilde'], rcond=None)
    eta_hat = d['Q_tilde'] - H_s @ co
    return _plm_fit(d['Y'], d['T'], d['X'], eta_hat, n_knots, ridge_scale)


def continuous_perfrdd(d, q_threshold, n_knots=3, ridge_scale=0.01):
    T_prime = (d['Q_tilde'] > q_threshold).astype(int)
    return _plm_fit(d['Y'], T_prime, d['X'], d['Q_tilde'], n_knots, ridge_scale)


def compute_continuous_truth(q_threshold, params=DGP, n_big=200_000, seed=999):
    d = generate(n_big, seed=seed, params=params)
    T_prime = (d['Q_tilde'] > q_threshold).astype(int)
    P = cubic_bspline_basis(d['eta'], n_knots=5)
    H = np.column_stack([np.ones(n_big), d['X'], P, T_prime])
    co, *_ = np.linalg.lstsq(H, d['Y'], rcond=None)
    return float(co[-1])


N_GRID = [500, 2000, 8000, 32_000]
N_SEEDS = 50


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    pilot = generate(100_000, seed=99_999)
    q_threshold = float(np.quantile(pilot['Q_tilde'], DGP['k0'] / DGP['J']))
    true_cont = compute_continuous_truth(q_threshold, DGP)
    true_alpha = DGP['alpha_0']
    pi_treated = float(np.mean(pilot['T']))
    r2 = float(np.corrcoef(pilot['Q_tilde'], pilot['Q'].astype(float))[0, 1]**2)

    # Diagnostic: within-treatment-group correlation Cor(X_b^T delta_b, eta_hat | T=1)
    n_pilot = len(pilot['Q_tilde'])
    H_s = np.column_stack([np.ones(n_pilot), pilot['X_a']])
    co, *_ = np.linalg.lstsq(H_s, pilot['Q_tilde'], rcond=None)
    eta_hat_pilot = pilot['Q_tilde'] - H_s @ co
    treated = pilot['T'] == 1
    Xb_index = pilot['X_b'] @ pilot['delta_b']
    cor_treated  = float(np.corrcoef(Xb_index[treated], eta_hat_pilot[treated])[0, 1])
    cor_overall  = float(np.corrcoef(Xb_index, eta_hat_pilot)[0, 1])

    print(f"True alpha_0:                                {true_alpha:.4f}")
    print(f"True continuous-threshold effect:            {true_cont:.4f}")
    print(f"fraction treated:                            {pi_treated:.3f}")
    print(f"R^2(Q_tilde, Q):                             {r2:.3f}")
    print(f"Cor(X_b^T delta_b, eta_hat) marginal:        {cor_overall:+.3f}")
    print(f"Cor(X_b^T delta_b, eta_hat | T=1) treated:   {cor_treated:+.3f}")

    results: List[Dict[str, Any]] = []
    for n in N_GRID:
        print(f"\n=== n = {n} ===")
        for seed in range(N_SEEDS):
            d = generate(n, seed=seed)
            row = dict(n=n, seed=seed)
            try:
                a, se = oracle_plm(d);   row['oracle_alpha']=a;  row['oracle_se']=se
            except Exception:
                row['oracle_alpha']=np.nan; row['oracle_se']=np.nan
            try:
                a, se = proposed_plm(d); row['proposed_alpha']=a; row['proposed_se']=se
            except Exception:
                row['proposed_alpha']=np.nan; row['proposed_se']=np.nan
            try:
                a, se = naive_ols(d);    row['naive_alpha']=a;    row['naive_se']=se
            except Exception:
                row['naive_alpha']=np.nan; row['naive_se']=np.nan
            try:
                a, se = continuous_perfrdd(d, q_threshold)
                row['cont_alpha']=a; row['cont_se']=se
            except Exception:
                row['cont_alpha']=np.nan; row['cont_se']=np.nan
            results.append(row)

        ora = np.array([r['oracle_alpha']  for r in results if r['n']==n])
        pro = np.array([r['proposed_alpha']for r in results if r['n']==n])
        nai = np.array([r['naive_alpha']   for r in results if r['n']==n])
        con = np.array([r['cont_alpha']    for r in results if r['n']==n])
        rmse = lambda v, t: float(np.sqrt(np.nanmean((v-t)**2)))
        r_o = rmse(ora, true_alpha)
        r_p = rmse(pro, true_alpha)
        print(f"  oracle:   med={np.nanmedian(ora):.3f}  RMSE={r_o:.3f}")
        print(f"  proposed: med={np.nanmedian(pro):.3f}  RMSE={r_p:.3f}  ({r_p/r_o:.2f}x oracle)")
        print(f"  naive:    med={np.nanmedian(nai):.3f}  RMSE={rmse(nai, true_alpha):.3f}")
        print(f"  cont:     med={np.nanmedian(con):.3f}  RMSE_vs_cont={rmse(con, true_cont):.3f}")

    OUT_JSON.write_text(json.dumps({
        'dgp': DGP,
        'true_alpha_0': true_alpha,
        'true_continuous_effect': true_cont,
        'q_threshold': q_threshold,
        'pi_treated': pi_treated,
        'r2_Qtilde_Q': r2,
        'cor_overall': cor_overall,
        'cor_treated': cor_treated,
        'results': results,
    }, indent=2, default=float))

    # ----------------------------- figure -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    estimators_for_alpha = [
        ('oracle',   'Oracle (knows η)',    'C2'),
        ('proposed', 'Proposed (η̂ from Q̃)','C0'),
        ('naive',    'Naive OLS',           'C3'),
    ]
    estimators_for_cont = [
        ('cont', 'Cont. perfRDD on Q̃ (different estimand)', 'C1'),
    ]

    # Panel 1: distribution of alpha_hat - alpha_0 (boxplots for the three targeting alpha_0)
    ax = axes[0]
    positions, box_data, box_colors, labels_x = [], [], [], []
    width = 0.22
    offset = {'oracle': -width, 'proposed': 0.0, 'naive': width}
    for j, n in enumerate(N_GRID):
        for key, lbl, col in estimators_for_alpha:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)] - true_alpha
            box_data.append(arr); positions.append(j + offset[key]); box_colors.append(col)
        labels_x.append(f"n={n}")
    bp = ax.boxplot(box_data, positions=positions, widths=width*0.95,
                    patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xticks(range(len(N_GRID))); ax.set_xticklabels(labels_x)
    ax.set_ylabel(r"$\hat\alpha - \alpha_0$")
    ax.set_title(r"Distribution around true $\alpha_0=1$")
    handles = [plt.Rectangle((0,0),1,1, fc=col, alpha=0.5, label=lbl) for _,lbl,col in estimators_for_alpha]
    ax.legend(handles=handles, fontsize=9, loc='best')
    ax.grid(alpha=0.25)

    # Panel 2: RMSE vs n, all four columns
    ax = axes[1]
    for key, lbl, col in estimators_for_alpha:
        rmse_arr = []
        for n in N_GRID:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)]
            rmse_arr.append(float(np.sqrt(np.mean((arr - true_alpha)**2))))
        ax.loglog(N_GRID, rmse_arr, 'o-', color=col, lw=2, ms=8, label=lbl)
    for key, lbl, col in estimators_for_cont:
        rmse_arr = []
        for n in N_GRID:
            arr = np.array([r[f'{key}_alpha'] for r in results if r['n']==n])
            arr = arr[~np.isnan(arr)]
            rmse_arr.append(float(np.sqrt(np.mean((arr - true_cont)**2))))
        ax.loglog(N_GRID, rmse_arr, 's--', color=col, lw=2, ms=8,
                  label=lbl + f" (vs own truth {true_cont:.3f})")
    n0 = N_GRID[0]; r0 = 0.3
    ax.loglog(N_GRID, [r0*np.sqrt(n0/n) for n in N_GRID], 'k:', lw=1, label=r'$n^{-1/2}$ reference')
    ax.set_xlabel("n"); ax.set_ylabel("RMSE")
    ax.set_title(r"RMSE vs $n$  (slope $-1/2$ $\Rightarrow$ $\sqrt{n}$ rate)")
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
    bias_c = []
    for n in N_GRID:
        arr = np.array([r['cont_alpha'] for r in results if r['n']==n])
        arr = arr[~np.isnan(arr)]
        bias_c.append(float(np.mean(arr) - true_alpha))
    ax.bar(xs + 2*bar_w, bias_c, bar_w, color='C1', alpha=0.7,
           label=f"Cont. perfRDD (rel. $\\alpha_0$; targets {true_cont:.3f})")
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"n={n}" for n in N_GRID])
    ax.set_ylabel(r"mean($\hat\alpha$) - $\alpha_0$")
    ax.set_title(r"Bias relative to $\alpha_0=1$")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.25)

    fig.suptitle(
        r"EIV-amplified DGP:  $b(\eta)=e^{\eta/2}$,  "
        fr"$\|\delta_b\|={DGP['delta_b_norm']}$,  "
        fr"$R^2(\tilde Q, Q)$={r2:.2f},  "
        fr"Cor($X_b^\top\delta_b,\,\hat\eta\,|\,T{{=}}1$)$={cor_treated:+.2f}$",
        fontsize=11, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nwrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
