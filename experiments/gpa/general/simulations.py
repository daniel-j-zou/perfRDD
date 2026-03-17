# simulation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import DataGenConfig, generate_dataset

from algorithm import (
    run_algorithm,
    EstimationResult,
    true_U_evo_from_dataset,
    NormalProjectionDistribution,
    true_U_evo
)



# ============================================================
# 1. Algorithm config for simulations
# ============================================================

@dataclass
class AlgoSimConfig:
    """
    Configuration for algorithm-level choices in simulations.

    Currently:
      - name: label used in plots/tables
      - c: the cost parameter in (R_t - R_{s_t} - c)
      - phi_grid: optional grid for diagnostic U_evo/u_evo plots
                  (phi_hat is found by root-finding, not restricted to this grid)
    """
    name: str
    c: float = 0.0
    phi_grid: Optional[np.ndarray] = None
    match_mode: str = "mutual_nn"  # or "one_way"


@dataclass
class SimulationConfig:
    """
    High-level specification of a simulation study.
    """
    n_values: List[int]
    num_reps: int
    dgp_configs: List[DataGenConfig]
    algo_configs: List[AlgoSimConfig]

    master_seed: int = 12345  # for reproducible randomness


# ============================================================
# 2. Core simulation runner
# ============================================================
def run_simulation(sim_cfg: SimulationConfig) -> pd.DataFrame:
    """
    Run the full simulation study described by sim_cfg.

    For each:
      - DGP config,
      - n in n_values,
      - rep in {0, ..., num_reps-1},
      - algo config,

    we:
      1) generate a dataset,
      2) run the algorithm,
      3) compute phi_hat, bias, variance contributions, etc.,
      4) store everything in a long-form DataFrame.

    True optimal cutoff is taken to be:
        phi_true = I1 + c * (1 + gamma^2)
    where I1, gamma come from the DGP config and c from the algo config.
    """
    master_rng = np.random.default_rng(sim_cfg.master_seed)
    rows: List[Dict[str, Any]] = []

    for dgp_idx, dgp_conf in enumerate(sim_cfg.dgp_configs):
        dgp_id = dgp_conf.dgp_id

        # Extract DGP-level params we need for phi_true
        I1 = float(dgp_conf.I1)
        gamma = float(dgp_conf.gamma)

        for n in sim_cfg.n_values:
            for rep in range(sim_cfg.num_reps):
                # Unique seed per dataset for reproducibility
                dataset_seed = int(master_rng.integers(1, 2**31 - 1))
                dataset = generate_dataset(n=n, config=dgp_conf, seed=dataset_seed)

                for algo_idx, algo_conf in enumerate(sim_cfg.algo_configs):
                    # True optimal phi for this (DGP, algo) combo:
                    c_val = float(algo_conf.c)
                    phi_true = I1 + c_val * (1.0 + gamma**2)

                    est_res: EstimationResult = run_algorithm(
                        dataset=dataset,
                        c=algo_conf.c,
                        phi_grid=algo_conf.phi_grid,
                        match_mode=algo_conf.match_mode,
                    )

                    phi_hat = est_res.phi_hat
                    error = phi_hat - phi_true
                    sq_error = error**2

                    # Flatten DGP config
                    dgp_params_flat = {
                        f"dgp_{k}": v
                        for k, v in vars(dgp_conf).items()
                    }

                    row = {
                        "dgp_id": dgp_id,
                        "dgp_index": dgp_idx,
                        "algo_name": algo_conf.name,
                        "algo_index": algo_idx,
                        "n": n,
                        "rep": rep,
                        "phi_hat": phi_hat,
                        "phi_true": phi_true,
                        "bias": error,
                        "sq_error": sq_error,
                        "theta_hat": est_res.theta_hat,
                        "alpha_hat": est_res.alpha_hat,
                        "gamma_intercept_hat": est_res.gamma_intercept_hat,
                        "gamma_slope_hat": est_res.gamma_slope_hat,
                        "matching_num_pairs": est_res.matching.num_pairs(),
                    }
                    row.update(dgp_params_flat)

                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 3. Summaries for bias / variance vs n
# ============================================================

def summarize_results_by_n(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate simulation results to show:
      - mean bias,
      - variance of phi_hat,
      - MSE,
    as functions of n, for each (dgp_id, algo_name) pair.

    Returns a summary DataFrame with columns:
      dgp_id, algo_name, n, mean_phi_hat, phi_true,
      bias_mean, var_phi_hat, mse_phi_hat, num_reps
    """
    group_cols = ["dgp_id", "algo_name", "n"]
    summary = (
        df
        .groupby(group_cols)
        .agg(
            mean_phi_hat=("phi_hat", "mean"),
            phi_true=("phi_true", "mean"),
            bias_mean=("bias", "mean"),
            var_phi_hat=("phi_hat", "var"),
            mse_phi_hat=("sq_error", "mean"),
            num_reps=("rep", "nunique"),
        )
        .reset_index()
    )
    return summary


# ============================================================
# 4. Plotting helpers
# ============================================================
def plot_mean_and_variance_vs_n(
    df: pd.DataFrame,
    ax_mean: Optional[plt.Axes] = None,
    ax_var: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plot:
      - E[phi_hat] vs n (left),
      - Var(phi_hat) vs n (right),

    with a horizontal line at the true phi* (phi_true).
    """
    # If df doesn't look summarized, summarize it
    if "mean_phi_hat" not in df.columns or "var_phi_hat" not in df.columns:
        summary = summarize_results_by_n(df)
    else:
        summary = df

    if ax_mean is None or ax_var is None:
        fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax_mean.get_figure()

    for (dgp_id, algo_name), sub in summary.groupby(["dgp_id", "algo_name"]):
        label = f"{dgp_id} | {algo_name}"
        sub = sub.sort_values("n")

        # Left panel: E[phi_hat] vs n
        ax_mean.plot(sub["n"], sub["mean_phi_hat"], marker="o", label=label)

        # Horizontal line at phi_true (assumed constant in this group)
        phi_true_val = sub["phi_true"].iloc[0]
        ax_mean.axhline(phi_true_val, linestyle="--", color="gray")

        # Right panel: Var(phi_hat) vs n
        ax_var.plot(sub["n"], sub["var_phi_hat"], marker="o", label=label)

    ax_mean.set_xscale("log")
    ax_var.set_xscale("log")

    ax_mean.set_xlabel("n (log scale)")
    ax_var.set_xlabel("n (log scale)")
    ax_mean.set_ylabel("E[phi_hat]")
    ax_var.set_ylabel("Var(phi_hat)")

    ax_mean.set_title("E[phi_hat] vs n")
    ax_var.set_title("Variance of phi_hat vs n")

    ax_mean.legend()
    ax_var.legend()

    fig.tight_layout()
    return fig, ax_mean, ax_var


def plot_phi_hat_distribution(
    df: pd.DataFrame,
    n: int,
    dgp_id: Optional[str] = None,
    algo_name: Optional[str] = None,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    For a fixed n (and optionally fixed dgp_id and algo_name),
    plot the empirical distribution of phi_hat as a histogram.

    This lets you see the shape and spread of the estimator at a given sample size.
    """
    sub = df[df["n"] == n].copy()

    if dgp_id is not None:
        sub = sub[sub["dgp_id"] == dgp_id]

    if algo_name is not None:
        sub = sub[sub["algo_name"] == algo_name]

    if sub.empty:
        raise ValueError("No rows match the specified filters.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    ax.hist(sub["phi_hat"], bins=bins, alpha=0.7, edgecolor="black")
    phi_true = sub["phi_true"].iloc[0]
    ax.axvline(phi_true, color="red", linestyle="--", label="phi_true")

    title = f"Distribution of phi_hat (n={n}"
    if dgp_id is not None:
        title += f", dgp_id={dgp_id}"
    if algo_name is not None:
        title += f", algo={algo_name}"
    title += ")"

    ax.set_title(title)
    ax.set_xlabel("phi_hat")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    return ax


def plot_phi_hat_vs_dgp_param(
    df: pd.DataFrame,
    n: int,
    param_name: str,
    algo_name: Optional[str] = None,
    dgp_id: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    For a fixed sample size n, visualize how phi_hat changes with a DGP parameter,
    e.g., param_name = "gamma" or "rho".

    Assumes df has a column named f"dgp_{param_name}" (as created in run_simulation).

    We plot the mean phi_hat (with error bars representing std dev) against that parameter.
    """
    col = f"dgp_{param_name}"
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in df. "
            f"Make sure the DGP config has a field '{param_name}'."
        )

    sub = df[df["n"] == n].copy()

    if dgp_id is not None:
        sub = sub[sub["dgp_id"] == dgp_id]

    if algo_name is not None:
        sub = sub[sub["algo_name"] == algo_name]

    if sub.empty:
        raise ValueError("No rows match the specified filters.")

    # Aggregate by param value
    grp = sub.groupby(col).agg(
        mean_phi_hat=("phi_hat", "mean"),
        std_phi_hat=("phi_hat", "std"),
        phi_true=("phi_true", "mean"),
        num_reps=("rep", "nunique"),
    ).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    ax.errorbar(
        grp[col],
        grp["mean_phi_hat"],
        yerr=grp["std_phi_hat"],
        fmt="-o",
        capsize=4,
        label="mean phi_hat ± 1 sd",
    )
    # If phi_true varies with param, we plot its line as well
    ax.plot(grp[col], grp["phi_true"], linestyle="--", color="red", label="phi_true")

    ax.set_xlabel(param_name)
    ax.set_ylabel("phi_hat")
    ax.set_title(f"phi_hat vs {param_name} (n={n})")
    ax.legend()

    fig.tight_layout()
    return ax

# ============================================================
# X. Debugging: compare estimated vs true U_evo for one dataset
# ============================================================



def sample_U_evo_direct(
    dataset,
    c: float,
    phi: np.ndarray | float,
) -> np.ndarray:
    """
    Direct empirical analogue of

        U_evo(phi) = E[(W - c) * 1{Q > phi}]

    using the observed (W, Q) from a single dataset.

    Args:
        dataset: Dataset instance (from data.py) with fields W, Q.
        c: cost parameter.
        phi: scalar or array of phi values.

    Returns:
        Array of U_hat_direct(phi) values (same shape as phi).
    """
    W = dataset.W
    Q = dataset.Q
    n = dataset.n

    phi_arr = np.asarray(phi, dtype=float)
    U_vals = np.empty_like(phi_arr, dtype=float)

    for j, phi_val in enumerate(phi_arr):
        indicator = (Q > phi_val).astype(float)
        vals = (W - c) * indicator
        U_vals[j] = float(np.mean(vals))

    return U_vals

def plot_all_U_flavors_for_dataset(
    dataset,
    c: float,
    phi_grid: Optional[np.ndarray] = None,
    match_mode: str = "mutual_nn",
) -> plt.Axes:

    """
    For a single dataset, compare:

      - Analytic U_evo(phi) (closed form, population),
      - U_MC(phi): *population* Monte Carlo from the DGP (large N),
      - U_hat_direct(phi): direct sample analogue (1/n sum (W-c)1{Q>phi}),
      - U_full_oracle(phi): matched/residual formula with true G, true theta, true eta,
      - U_hat(phi): matched/residual formula with all nuisance estimated.

    This lets you see:
      - sample vs population for the *correct* definition,
      - how far the matched/residual representation departs from that target.
    """
    Y = dataset.Y
    Q = dataset.Q
    X = dataset.X
    eta_true = dataset.eta
    n = dataset.n

    # Build phi grid if not provided
    q_min = float(np.min(Q))
    q_max = float(np.max(Q))
    span = q_max - q_min
    margin = 0.1 * (span + 1e-8)

    if phi_grid is None:
        phi_grid = np.linspace(q_min - margin, q_max + margin, 400)
    else:
        phi_grid = np.asarray(phi_grid, dtype=float)

    # 1) Analytic U_evo (population)
    U_true = true_U_evo_from_dataset(dataset, c=c, phi=phi_grid)

    # 2) Population Monte Carlo U_evo (for visual reassurance)
    #    Use the underlying DGP config inferred from dataset.params.
    from data import DataGenConfig
    cfg = DataGenConfig(
        I0=float(dataset.params["I0"]),
        I1=float(dataset.params["I1"]),
        gamma=float(dataset.params["gamma"]),
        theta=float(dataset.params["theta"]),
        rho=float(dataset.params["rho"]),
        phi=float(dataset.params["phi"]),
        sigma_X=float(dataset.params["sigma_X"]),
        sigma_eta=float(dataset.params["sigma_eta"]),
        sigma_eps=float(dataset.params["sigma_eps"]),
        sigma_W_cond=float(dataset.params["sigma_W_cond"]),
        dgp_id=dataset.meta.get("dgp_id", "reconstructed"),
    )
    U_mc = monte_carlo_U_evo_from_config(
        phi=phi_grid,
        config=cfg,
        c=c,
        N_mc=200_000,
        seed=123,
    )

    # 3) Direct sample analogue
    U_direct = sample_U_evo_direct(dataset, c=c, phi=phi_grid)

    # 4) Run estimator to get U_hat and oracle curves
    est_res: EstimationResult = run_algorithm(
        dataset=dataset,
        c=c,
        phi_grid=phi_grid,
        match_mode=match_mode,
    )
    U_hat = est_res.U_evo
    phi_hat = est_res.phi_hat
    matching = est_res.matching
    theta_hat = est_res.theta_hat
    eta_hat = est_res.diagnostics["eta_hat"]
    G_params_hat = est_res.G_params
    G_est = NormalProjectionDistribution(
        mean=G_params_hat["mean"],
        std=G_params_hat["std"],
    )
    # Indices on the *admitted* side used by the matching procedure.
    # For mutual_nn this is a subset of S_A; for one_way it's all of S_A.
    admitted_idx = matching.admitted_idx

    # Direct sample U_hat but only on the admitted units that participate in pairs
    U_direct_matched = sample_U_evo_direct_on_indices(
        dataset=dataset,
        c=c,
        phi=phi_grid,
        indices=admitted_idx,
    )



    # True G and theta for oracle
    I1 = float(dataset.params["I1"])
    gamma = float(dataset.params["gamma"])
    G_true = NormalProjectionDistribution(
        mean=I1,
        std=np.sqrt(1.0 + gamma**2),
    )
    theta_true = float(dataset.params["theta"])

    U_full_oracle = _compute_U_evo_given_components(
        phi_grid=phi_grid,
        c=c,
        Y=Y,
        X=X,
        eta_vec=eta_true,
        matching=matching,
        theta_for_R=theta_true,
        G_dist=G_true,
        n_total=n,  # unused internally, but kept for API
    )

    # True optimizer
    phi_star = I1 + c * (1.0 + gamma**2)

    # 5) Plot everything
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(phi_grid, U_true, label="Analytic U_evo (population)", color="black", linewidth=2)
    ax.plot(phi_grid, U_mc, label="MC U_evo (population)", linestyle="--", color="blue")
    ax.plot(phi_grid, U_direct, label="Direct sample U_hat (full data)", linestyle="-.", color="purple")
    ax.plot(phi_grid, U_direct_matched, label="Direct sample U_hat (matched subset)", linestyle="--", color="brown")
    ax.plot(phi_grid, U_full_oracle, label="U_full_oracle (matching/residual, oracle nuisances)", linestyle=":", color="orange")
    ax.plot(phi_grid, U_hat, label="U_hat (matching/residual, estimated)", linestyle="--", color="red")


    ax.axvline(phi_star, color="green", linestyle=":", label=r"true $\phi^*$")
    ax.axvline(phi_hat, color="red", linestyle="-.", label=r"estimated $\hat{\phi}$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U_{\mathrm{evo}}(\phi)$")
    ax.set_title("All U_evo flavors for one dataset")

    ax.legend(fontsize=8)
    fig.tight_layout()
    return ax


def monte_carlo_U_evo_from_config(
    phi: np.ndarray | float,
    config: DataGenConfig,
    c: float,
    N_mc: int = 200_000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Monte Carlo approximation of

        U_evo(phi) = E[(W - c) * 1{Q > phi}]

    under the DGP given by `config`.

    We simulate a large sample of size N_mc using `generate_dataset`
    and compute the empirical mean of (W - c) * 1{Q > phi}.

    Args:
        phi: scalar or array of phi values.
        config: DataGenConfig describing the DGP.
        c: cost parameter.
        N_mc: Monte Carlo sample size.
        seed: random seed for reproducibility.

    Returns:
        Array of U_evo(phi) values (same shape as phi).
    """
    phi_arr = np.asarray(phi, dtype=float)

    # Generate one large dataset (we only need W and Q)
    ds_mc = generate_dataset(n=N_mc, config=config, seed=seed)
    W = ds_mc.W
    Q = ds_mc.Q

    U_vals = np.empty_like(phi_arr, dtype=float)

    # Loop over phi to avoid huge (N x len(phi)) matrices in memory
    for j, phi_val in enumerate(phi_arr):
        indicator = (Q > phi_val).astype(float)
        vals = (W - c) * indicator
        U_vals[j] = float(np.mean(vals))

    return U_vals


def _compute_U_evo_given_components(
    phi_grid: np.ndarray,
    c: float,
    Y: np.ndarray,
    X: np.ndarray,
    eta_vec: np.ndarray,
    matching,
    theta_for_R: float,
    G_dist: NormalProjectionDistribution,
    n_total: int,
) -> np.ndarray:
    """
    Generic plug-in version of

        U_hat_evo(phi') = (1/n) sum_{t in tilde S_A}
                            (R_t - R_{s_t} - c) * bar G(phi' - eta_t),

    where:
        R_t = Y_t - theta_for_R * X_t,
        G_dist is either the estimated or true NormalProjectionDistribution,
        eta_vec is either eta_hat or the true eta, etc.

    This lets us create "oracle" versions by swapping in true/estimated pieces.
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    t_idx = matching.admitted_idx
    s_idx = matching.denied_idx

    m = t_idx.size
    if m == 0:
        return np.zeros_like(phi_grid)

    if t_idx.size == 0:
        return np.zeros_like(phi_grid)

    # Build residuals using the supplied theta_for_R
    R = Y - theta_for_R * X

    diff_R_minus_c = R[t_idx] - R[s_idx] - c
    eta_t = eta_vec[t_idx]

    U_vals = np.empty_like(phi_grid, dtype=float)
    denom = float(m)
    for i, phi_prime in enumerate(phi_grid):
        arg = phi_prime - eta_t
        barG_vals = G_dist.survival(arg)
        U_vals[i] = (1.0 / denom) * np.sum(diff_R_minus_c * barG_vals)

    return U_vals


def plot_estimated_vs_true_U_evo(
    dataset,
    c: float,
    phi_grid: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    For a single dataset, plot:

      - estimated U_hat_evo(phi) from the algorithm,
      - true U_evo(phi) from the analytic formula,

    on the same phi grid, and mark:
      - true optimal phi_star = I1 + c * (1 + gamma^2),
      - estimated phi_hat from the algorithm.

    Args:
        dataset: a Dataset instance (from data.py).
        c: cost parameter used in U_evo.
        phi_grid: optional array of phi values. If None, we build a grid
                  around the range of Q in the dataset.
        ax: optional matplotlib Axes to plot into. If None, a new figure
            and axes are created.

    Returns:
        The matplotlib Axes with the plot.
    """
    # 1. Choose phi grid if not provided
    Q = dataset.Q
    q_min = float(np.min(Q))
    q_max = float(np.max(Q))
    span = q_max - q_min
    margin = 0.1 * (span + 1e-8)

    if phi_grid is None:
        phi_grid = np.linspace(q_min - margin, q_max + margin, 400)
    else:
        phi_grid = np.asarray(phi_grid, dtype=float)

    # 2. Run the estimator on this dataset using this phi_grid (for diagnostics)
    est_res: EstimationResult = run_algorithm(
        dataset=dataset,
        c=c,
        phi_grid=phi_grid,
    )
    phi_hat = est_res.phi_hat
    U_hat = est_res.U_evo  # evaluated on est_res.phi_grid == phi_grid

    # 3. Compute the true U_evo on the same grid
    U_true = true_U_evo_from_dataset(dataset, c=c, phi=phi_grid)

    # 4. True optimal phi_star = I1 + c * (1 + gamma^2)
    I1 = float(dataset.params["I1"])
    gamma = float(dataset.params["gamma"])
    phi_star = I1 + c * (1.0 + gamma**2)

    # 5. Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.plot(phi_grid, U_true, label="True U_evo(phi)", linewidth=2)
    ax.plot(phi_grid, U_hat, label="Estimated U_hat_evo(phi)", linestyle="--")

    ax.axvline(phi_star, color="green", linestyle=":", label=r"true $\phi^*$")
    ax.axvline(phi_hat, color="red", linestyle="-.", label=r"estimated $\hat{\phi}$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U_{\mathrm{evo}}(\phi)$")
    ax.set_title("Estimated vs True U_evo")

    ax.legend()
    fig.tight_layout()

    return ax

def diagnose_U_components(
    dataset,
    c: float,
    phi_grid: Optional[np.ndarray] = None,
) -> plt.Axes:
    """
    For a single dataset, compare several versions of U_evo:

      - True U_evo(phi): analytic expression.
      - Estimator's U_hat_evo(phi): fully estimated (as currently implemented).
      - U_Gtrue_Rhat: using true G (I1, gamma) but estimated residuals R_hat and eta_hat.
      - U_full_oracle: using true G, true theta, true eta.

    This helps isolate whether the main problem is
    G-estimation, PLM (theta), or eta-estimation/matching.
    """
    Y = dataset.Y
    Q = dataset.Q
    X = dataset.X
    eta_true = dataset.eta
    n = dataset.n

    # Choose phi grid if not provided
    q_min = float(np.min(Q))
    q_max = float(np.max(Q))
    span = q_max - q_min
    margin = 0.1 * (span + 1e-8)

    if phi_grid is None:
        phi_grid = np.linspace(q_min - margin, q_max + margin, 400)
    else:
        phi_grid = np.asarray(phi_grid, dtype=float)

    # Run the estimator to get eta_hat, theta_hat, matching, estimated G, etc.
    est_res: EstimationResult = run_algorithm(
        dataset=dataset,
        c=c,
        phi_grid=phi_grid,
    )

    phi_hat = est_res.phi_hat
    U_hat = est_res.U_evo
    matching = est_res.matching
    theta_hat = est_res.theta_hat

    # Pull diagnostics we stored in the algorithm
    eta_hat = est_res.diagnostics["eta_hat"]
    G_params_hat = est_res.G_params
    G_est = NormalProjectionDistribution(
        mean=G_params_hat["mean"],
        std=G_params_hat["std"],
    )

    # True G (for Q = I1 + gamma X + eta)
    I1 = float(dataset.params["I1"])
    gamma = float(dataset.params["gamma"])
    G_true = NormalProjectionDistribution(
        mean=I1,
        std=np.sqrt(1.0 + gamma**2),
    )

    # True theta (from DGP config)
    theta_true = float(dataset.params["theta"])

    # 1) Fully estimated curve (already computed as U_hat)
    U_estimated = U_hat

    # 2) Use true G but keep R_hat and eta_hat: isolate G-estimation error
    U_Gtrue_Rhat = _compute_U_evo_given_components(
        phi_grid=phi_grid,
        c=c,
        Y=Y,
        X=X,
        eta_vec=eta_hat,
        matching=matching,
        theta_for_R=theta_hat,
        G_dist=G_true,
        n_total=n,
    )

    # 3) "Full oracle": use true G, true theta, true eta
    U_full_oracle = _compute_U_evo_given_components(
        phi_grid=phi_grid,
        c=c,
        Y=Y,
        X=X,
        eta_vec=eta_true,
        matching=matching,
        theta_for_R=theta_true,
        G_dist=G_true,
        n_total=n,
    )

    # 4) Analytic true U_evo
    U_true = true_U_evo_from_dataset(dataset, c=c, phi=phi_grid)

    # True optimal phi*
    phi_star = I1 + c * (1.0 + gamma**2)

    # Plot everything
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(phi_grid, U_true, label="True U_evo(phi)", linewidth=2, color="black")
    ax.plot(phi_grid, U_full_oracle, label="U_full_oracle (true G, true theta, true eta)", linestyle="-", color="blue")
    ax.plot(phi_grid, U_Gtrue_Rhat, label="U_Gtrue_Rhat (true G, est theta & eta)", linestyle="--", color="orange")
    ax.plot(phi_grid, U_estimated, label="U_hat (all estimated)", linestyle=":", color="red")

    ax.axvline(phi_star, color="green", linestyle=":", label=r"true $\phi^*$")
    ax.axvline(phi_hat, color="red", linestyle="-.", label=r"estimated $\hat{\phi}$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U_{\mathrm{evo}}(\phi)$")
    ax.set_title("Decomposition of U_evo estimation")

    ax.legend()
    fig.tight_layout()
    return ax

def plot_analytic_vs_mc_U_evo(
    config: DataGenConfig,
    c: float,
    phi_grid: Optional[np.ndarray] = None,
    N_mc: int = 200_000,
    seed: int = 123,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare the analytic U_evo(phi) to a Monte Carlo approximation
    of U_evo(phi) = E[(W - c) * 1{Q > phi}] under the DGP `config`.

    This checks whether the closed-form expression true_U_evo matches
    the actual DGP you've coded.

    Args:
        config: DataGenConfig describing the DGP.
        c: cost parameter.
        phi_grid: optional array of phi values. If None, we build a grid
                  around the range where Q typically lies.
        N_mc: Monte Carlo sample size for the population approximation.
        seed: RNG seed for Monte Carlo sampling.
        ax: optional matplotlib Axes.

    Returns:
        The matplotlib Axes with the plot.
    """
    # If phi_grid not given, build a reasonable one
    if phi_grid is None:
        # quick probe dataset to get Q range
        probe_ds = generate_dataset(n=10_000, config=config, seed=seed)
        Q = probe_ds.Q
        q_min = float(np.min(Q))
        q_max = float(np.max(Q))
        span = q_max - q_min
        margin = 0.1 * (span + 1e-8)
        phi_grid = np.linspace(q_min - margin, q_max + margin, 400)
    else:
        phi_grid = np.asarray(phi_grid, dtype=float)

    # Analytic U_evo from the closed-form formula
    U_analytic = true_U_evo(
        phi=phi_grid,
        I1=float(config.I1),
        gamma=float(config.gamma),
        c=c,
    )

    # Monte Carlo approximation from the DGP
    U_mc = monte_carlo_U_evo_from_config(
        phi=phi_grid,
        config=config,
        c=c,
        N_mc=N_mc,
        seed=seed,
    )

    # True optimal phi* used in your bias definition
    phi_star = float(config.I1) + c * (1.0 + float(config.gamma) ** 2)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.plot(phi_grid, U_analytic, label="Analytic U_evo(phi)", linewidth=2, color="black")
    ax.plot(phi_grid, U_mc, label=f"Monte Carlo U_evo(phi) (N={N_mc})", linestyle="--", color="blue")

    ax.axvline(phi_star, color="green", linestyle=":", label=r"analytic $\phi^* = I_1 + c(1+\gamma^2)$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U_{\mathrm{evo}}(\phi)$")
    ax.set_title("Analytic vs Monte Carlo U_evo (population level)")

    ax.legend()
    fig.tight_layout()
    return ax

def distance_true_vs_oracle_vs_hat(
    config: DataGenConfig,
    c: float,
    n_values: list[int],
    N_mc_phi: int = 400,
    master_seed: int = 12345,
) -> pd.DataFrame:
    """
    For each n in n_values, generate ONE dataset and compute
    L2 and sup-norm distances between:

      - analytic U_true(phi),
      - U_full_oracle(phi),
      - U_hat(phi),

    over a common phi grid.

    Returns a DataFrame with columns:
      n, sup_true_vs_oracle, sup_true_vs_hat,
         l2_true_vs_oracle, l2_true_vs_hat
    """
    rng = np.random.default_rng(master_seed)
    rows = []

    for n in n_values:
        seed = int(rng.integers(1, 2**31 - 1))
        ds = generate_dataset(n=n, config=config, seed=seed)

        # Build phi grid around Q range
        Q = ds.Q
        q_min = float(np.min(Q))
        q_max = float(np.max(Q))
        span = q_max - q_min
        margin = 0.1 * (span + 1e-8)
        phi_grid = np.linspace(q_min - margin, q_max + margin, N_mc_phi)

        # Analytic true U
        U_true = true_U_evo_from_dataset(ds, c=c, phi=phi_grid)

        # Run algorithm to get U_hat and oracle
        est_res = run_algorithm(dataset=ds, c=c, phi_grid=phi_grid)
        U_hat = est_res.U_evo
        matching = est_res.matching
        theta_hat = est_res.theta_hat
        eta_hat = est_res.diagnostics["eta_hat"]
        Y, X = ds.Y, ds.X
        n_total = ds.n

        # True nuisances
        eta_true = ds.eta
        I1 = float(ds.params["I1"])
        gamma = float(ds.params["gamma"])
        theta_true = float(ds.params["theta"])
        G_true = NormalProjectionDistribution(
            mean=I1,
            std=np.sqrt(1.0 + gamma**2),
        )

        U_full_oracle = _compute_U_evo_given_components(
            phi_grid=phi_grid,
            c=c,
            Y=Y,
            X=X,
            eta_vec=eta_true,
            matching=matching,
            theta_for_R=theta_true,
            G_dist=G_true,
            n_total=n_total,  # unused internally
        )

        # distance helpers
        def sup_norm(a, b):
            return float(np.max(np.abs(a - b)))

        def l2_norm(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        rows.append({
            "n": n,
            "sup_true_vs_oracle": sup_norm(U_true, U_full_oracle),
            "sup_true_vs_hat": sup_norm(U_true, U_hat),
            "l2_true_vs_oracle": l2_norm(U_true, U_full_oracle),
            "l2_true_vs_hat": l2_norm(U_true, U_hat),
        })

    return pd.DataFrame(rows)

def sample_U_evo_direct_on_indices(
    dataset,
    c: float,
    phi: np.ndarray | float,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Direct empirical analogue of

        U_evo(phi) = E[(W - c) * 1{Q > phi}]

    but using only a subset of observations specified by `indices`.

    Args:
        dataset: Dataset instance (from data.py) with fields W, Q.
        c: cost parameter.
        phi: scalar or array of phi values.
        indices: 1D array of integer indices to keep.

    Returns:
        Array of U_hat(phi) values (same shape as phi) computed over the subset.
    """
    W = dataset.W[indices]
    Q = dataset.Q[indices]
    m = len(indices)

    phi_arr = np.asarray(phi, dtype=float)
    U_vals = np.empty_like(phi_arr, dtype=float)

    for j, phi_val in enumerate(phi_arr):
        indicator = (Q > phi_val).astype(float)
        vals = (W - c) * indicator
        U_vals[j] = float(np.mean(vals))  # 1/m sum over subset

    return U_vals






# ============================================================
# 5. Example usage (optional smoke test)
# ============================================================

if __name__ == "__main__":
    # Example: one DGP, two sample sizes, one algorithm config
    from data import DataGenConfig

    dgp_baseline = DataGenConfig(
        I0=0,
        I1=0,
        gamma=1,
        theta=1,
        rho=0.0,
        phi=0.0,  # status-quo and "true" phi for now
        dgp_id="baseline",
    )

    algo_baseline = AlgoSimConfig(
        name="c=0",
        c=0,
        phi_grid=None,  # let algorithm choose default diagnostic grid
        match_mode= "one_way",
    )

    sim_cfg = SimulationConfig(
        n_values=[200, 500, 1000, 2000, 5000],
        num_reps=1000,
        dgp_configs=[dgp_baseline],
        algo_configs=[algo_baseline],
        master_seed=2025,
    )

    df_results = run_simulation(sim_cfg)
    print(df_results.head())

    # Plot bias and variance vs n
    fig, ax_bias, ax_var = plot_mean_and_variance_vs_n(df_results)
    plt.show()

    # Plot distribution of phi_hat for n=500
    plot_phi_hat_distribution(df_results, n=2000, dgp_id="baseline", algo_name="c=1")
    plt.show()
