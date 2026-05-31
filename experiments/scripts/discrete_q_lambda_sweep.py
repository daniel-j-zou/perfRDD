"""Lambda sweep: how does the strength of eta in Q affect the estimators?

DGP is the standard discrete_q_simulation DGP, but sweeps lambda (the
eta-loading in the latent score S = X_b^T delta_b + lambda * eta + u).

At lambda = 0, D = 1{Q > k_0} is independent of eta given (X_b, u), so
naive OLS becomes unbiased under (A7). As lambda grows, naive develops
bias from eta-driven selection. The proposed estimator should remain
unbiased at every lambda.

Reports bias and RMSE at fixed n for several lambda values.
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
    cubic_bspline_basis, _plm_fit, naive_ols, DGP as DGP_BASE, generate as generate_base,
    oracle_plm, proposed_plm,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "runs" / "discrete_q_lambda_sweep.json"
OUT_FIG  = ROOT / "runs" / "discrete_q_lambda_sweep.png"


LAMBDA_GRID = [0.0, 0.3, 0.7, 1.5, 3.0, 6.0]
N           = 4000
N_SEEDS     = 100


def generate_lambda(n: int, seed: int, lam: float) -> Dict[str, Any]:
    """Standard DGP but with lambda_eta_in_S overridden."""
    params = dict(DGP_BASE)
    params['lambda_eta_in_S'] = lam
    return generate_base(n, seed=seed, params=params)


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    true_alpha = DGP_BASE['alpha_0']

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for lam in LAMBDA_GRID:
        # Pilot for diagnostics
        pilot = generate_lambda(100_000, seed=99_999, lam=lam)
        pi_treated = float(np.mean(pilot['T']))
        r2 = float(np.corrcoef(pilot['Q_tilde'], pilot['Q'].astype(float))[0, 1]**2)
        treated = pilot['T'] == 1
        Xb_index = pilot['X_b'] @ pilot['delta_b']
        H_s = np.column_stack([np.ones(len(pilot['Q_tilde'])), pilot['X_a']])
        co, *_ = np.linalg.lstsq(H_s, pilot['Q_tilde'], rcond=None)
        eta_hat_pilot = pilot['Q_tilde'] - H_s @ co
        cor_treated = float(np.corrcoef(Xb_index[treated], eta_hat_pilot[treated])[0, 1])
        # Strength of eta in S: R^2 of S on eta after removing X_b contribution
        S_resid = pilot['S'] - Xb_index
        r2_eta_in_S = float(np.var(lam * pilot['eta']) / np.var(S_resid)) if np.var(S_resid) > 0 else 0.0

        print(f"\n=== lambda = {lam} ===")
        print(f"  fraction treated:               {pi_treated:.3f}")
        print(f"  R^2(Q_tilde, Q):                {r2:.3f}")
        print(f"  Cor(X_b'delta_b, eta_hat|T=1):  {cor_treated:+.3f}")
        print(f"  R^2(eta -> S | X_b):            {r2_eta_in_S:.3f}")

        ora_arr, pro_arr, nai_arr = [], [], []
        for seed in range(N_SEEDS):
            d = generate_lambda(N, seed=seed, lam=lam)
            try:
                a, _ = oracle_plm(d); ora_arr.append(a)
            except Exception: pass
            try:
                a, _ = proposed_plm(d); pro_arr.append(a)
            except Exception: pass
            try:
                a, _ = naive_ols(d); nai_arr.append(a)
            except Exception: pass
            rows.append(dict(lam=lam, seed=seed,
                              oracle=float(ora_arr[-1]) if ora_arr else None,
                              proposed=float(pro_arr[-1]) if pro_arr else None,
                              naive=float(nai_arr[-1]) if nai_arr else None))

        ora_arr = np.array(ora_arr); pro_arr = np.array(pro_arr); nai_arr = np.array(nai_arr)
        bias_o  = float(np.mean(ora_arr) - true_alpha)
        bias_p  = float(np.mean(pro_arr) - true_alpha)
        bias_n  = float(np.mean(nai_arr) - true_alpha)
        std_o   = float(np.std(ora_arr))
        std_p   = float(np.std(pro_arr))
        std_n   = float(np.std(nai_arr))
        rmse_o  = float(np.sqrt(bias_o**2 + std_o**2))
        rmse_p  = float(np.sqrt(bias_p**2 + std_p**2))
        rmse_n  = float(np.sqrt(bias_n**2 + std_n**2))
        print(f"  oracle:   bias={bias_o:+.3f}  std={std_o:.3f}  RMSE={rmse_o:.3f}")
        print(f"  proposed: bias={bias_p:+.3f}  std={std_p:.3f}  RMSE={rmse_p:.3f}")
        print(f"  naive:    bias={bias_n:+.3f}  std={std_n:.3f}  RMSE={rmse_n:.3f}")

        summary.append(dict(
            lam=lam, pi_treated=pi_treated, r2_QtQ=r2, cor_treated=cor_treated,
            r2_eta_in_S=r2_eta_in_S,
            bias_oracle=bias_o, bias_proposed=bias_p, bias_naive=bias_n,
            std_oracle=std_o,   std_proposed=std_p,   std_naive=std_n,
            rmse_oracle=rmse_o, rmse_proposed=rmse_p, rmse_naive=rmse_n,
        ))

    OUT_JSON.write_text(json.dumps({
        'n': N, 'n_seeds': N_SEEDS, 'true_alpha': true_alpha,
        'lambda_grid': LAMBDA_GRID,
        'summary': summary,
        'rows': rows,
    }, indent=2, default=float))

    # ----------------------------- figure -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    lams = np.array(LAMBDA_GRID)

    # Panel 1: bias vs lambda
    ax = axes[0]
    ax.plot(lams, [s['bias_oracle']  for s in summary], 'o-', color='C2', lw=2, ms=7, label='Oracle')
    ax.plot(lams, [s['bias_proposed']for s in summary], 'o-', color='C0', lw=2, ms=7, label='Proposed')
    ax.plot(lams, [s['bias_naive']   for s in summary], 'o-', color='C3', lw=2, ms=7, label='Naive')
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xlabel(r"$\lambda$ (eta-loading in $S$)")
    ax.set_ylabel(r"mean($\hat\alpha$) - $\alpha_0$")
    ax.set_title(r"Bias vs $\lambda$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Panel 2: standard deviation vs lambda
    ax = axes[1]
    ax.plot(lams, [s['std_oracle']  for s in summary], 'o-', color='C2', lw=2, ms=7, label='Oracle')
    ax.plot(lams, [s['std_proposed']for s in summary], 'o-', color='C0', lw=2, ms=7, label='Proposed')
    ax.plot(lams, [s['std_naive']   for s in summary], 'o-', color='C3', lw=2, ms=7, label='Naive')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"std($\hat\alpha$)")
    ax.set_title(r"Estimator std vs $\lambda$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Panel 3: RMSE vs lambda
    ax = axes[2]
    ax.plot(lams, [s['rmse_oracle']  for s in summary], 'o-', color='C2', lw=2, ms=7, label='Oracle')
    ax.plot(lams, [s['rmse_proposed']for s in summary], 'o-', color='C0', lw=2, ms=7, label='Proposed')
    ax.plot(lams, [s['rmse_naive']   for s in summary], 'o-', color='C3', lw=2, ms=7, label='Naive')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("RMSE")
    ax.set_title(r"RMSE vs $\lambda$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle(
        fr"Lambda sweep: how does $\lambda$ (the eta-loading in $S$) affect each estimator? "
        fr"($n={N}$, {N_SEEDS} seeds)",
        fontsize=12, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nwrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
