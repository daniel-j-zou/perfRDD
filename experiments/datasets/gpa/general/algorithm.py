# algorithm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import math

from data import Dataset  # assumes data.py is in the same folder


# ============================================================
# 1. Helper: normal CDF / PDF for G and g
# ============================================================

class NormalProjectionDistribution:
    """
    Normal approximation to the distribution of hat{gamma}^T Z.
    In this scalar setting, Z = X and we use hat_gamma * X.

    G(x) = Phi((x - mean) / std)
    barG(x) = 1 - G(x)
    g(x) = normal pdf with same mean/std
    """

    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        self.mean = float(mean)
        # avoid zero std
        self.std = float(max(std, eps))

    def _standard_normal_cdf(self, z: np.ndarray) -> np.ndarray:
        """
        Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
        We use math.erf, vectorized to handle arrays.
        """
        z = np.asarray(z, dtype=float)
        # vectorize math.erf over the array
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = (x_arr - self.mean) / self.std
        return self._standard_normal_cdf(z)

    def survival(self, x: np.ndarray | float) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = (x_arr - self.mean) / self.std
        return (
            1.0
            / (self.std * math.sqrt(2.0 * math.pi))
            * np.exp(-0.5 * z**2)
        )

# ============================================================
# X. Analytic (true) U_evo
# ============================================================

def true_U_evo(
    phi: np.ndarray | float,
    I1: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """
    Analytic expression for the true U_evo(phi):

        U_evo(phi) =
            1 / sqrt(2 * pi * (1 + gamma^2))
            * exp( - (phi - I1)^2 / (2 * (1 + gamma^2)) )
            - c * ( 1 - Phi( (phi - I1) / sqrt(1 + gamma^2) ) )

    where Phi is the standard normal CDF. This matches:

        Q = I1 + gamma * X + eta,  with  X, eta ~ N(0, 1),

    so gamma^2 * Var(X) + Var(eta) = gamma^2 + 1.

    Args:
        phi: scalar or array of phi values.
        I1: scalar I1 parameter from the DGP.
        gamma: scalar gamma parameter from the DGP.
        c: cost parameter (same c used in the estimator).

    Returns:
        U_true(phi) evaluated at phi (same shape as phi).
    """
    # std of Q's "score" part (gamma * X + eta)
    var_tot = 1.0 + gamma**2
    std = float(np.sqrt(var_tot))
    mean = float(I1)

    # Use the same NormalProjectionDistribution used in the estimator
    dist = NormalProjectionDistribution(mean=mean, std=std)

    phi_arr = np.asarray(phi, dtype=float)
    pdf_vals = dist.pdf(phi_arr)           # first term
    surv_vals = dist.survival(phi_arr)     # 1 - Phi(...)

    return pdf_vals - c * surv_vals


def true_U_evo_from_dataset(
    dataset: "Dataset",
    c: float,
    phi: np.ndarray | float,
) -> np.ndarray:
    """
    Convenience wrapper: pull I1 and gamma from dataset.params
    and evaluate true_U_evo at phi.

    Requires dataset.params to contain "I1" and "gamma".
    """
    I1 = float(dataset.params["I1"])
    gamma = float(dataset.params["gamma"])
    return true_U_evo(phi=phi, I1=I1, gamma=gamma, c=c)



# ============================================================
# 2. Matching structure and computation
# ============================================================

@dataclass
class Matching:
    """
    Matching between admitted and denied units.

    admitted_idx[i] is matched with denied_idx[i], for i = 0,...,m-1.

    mode:
      - "mutual_nn": one-to-one mutual nearest neighbors (old behavior)
      - "one_way":   each admitted t gets its nearest denied s,
                     controls can be reused multiple times
    """
    admitted_idx: np.ndarray  # indices in S_A
    denied_idx: np.ndarray    # indices in S_D (may contain repeats in "one_way")
    mode: str = "mutual_nn"

    def num_pairs(self) -> int:
        return self.admitted_idx.shape[0]

def compute_matching(
    eta_hat: np.ndarray,
    Q: np.ndarray,
    phi: float,
    mode: str = "mutual_nn",
) -> Matching:
    """
    Compute matching between admitted (S_A) and denied (S_D) groups.

    mode:
      - "mutual_nn": mutual nearest neighbors (one-to-one).
                     For each t in S_A, find nearest s in S_D;
                     for each s in S_D, find nearest t in S_A;
                     keep only pairs (t,s) that are mutual.
      - "one_way":   for each t in S_A, find nearest s in S_D.
                     No reverse matching; an s may be reused for
                     multiple t. This preserves the full S_A sample.
    """
    n = eta_hat.shape[0]
    indices = np.arange(n)

    S_A = indices[Q > phi]
    S_D = indices[Q < phi]

    if S_A.size == 0 or S_D.size == 0:
        # no possible matching
        return Matching(
            admitted_idx=np.zeros(0, dtype=int),
            denied_idx=np.zeros(0, dtype=int),
            mode=mode,
        )

    eta_A = eta_hat[S_A]
    eta_D = eta_hat[S_D]

    if mode == "one_way":
        # --------------------------------------------------------
        # New behavior: each admitted t gets its nearest denied s.
        # Controls may be reused multiple times.
        # --------------------------------------------------------
        admitted_idx = S_A.copy()
        denied_idx = np.empty_like(S_A)

        for i, t in enumerate(S_A):
            dist = np.abs(eta_D - eta_hat[t])
            j = int(np.argmin(dist))
            denied_idx[i] = int(S_D[j])

        # Optional: sort by admitted index for determinism
        order = np.argsort(admitted_idx)
        admitted_idx = admitted_idx[order]
        denied_idx = denied_idx[order]

        return Matching(
            admitted_idx=admitted_idx,
            denied_idx=denied_idx,
            mode=mode,
        )

    # ------------------------------------------------------------
    # Default / old behavior: mutual nearest neighbors (one-to-one)
    # ------------------------------------------------------------
    # For each admitted t, find nearest denied s
    best_D_for_A: Dict[int, int] = {}
    for i, t in enumerate(S_A):
        dist = np.abs(eta_D - eta_hat[t])
        j = int(np.argmin(dist))
        s = int(S_D[j])
        best_D_for_A[int(t)] = s

    # For each denied s, find nearest admitted t
    best_A_for_D: Dict[int, int] = {}
    for j, s in enumerate(S_D):
        dist = np.abs(eta_A - eta_hat[s])
        i = int(np.argmin(dist))
        t = int(S_A[i])
        best_A_for_D[int(s)] = t

    # Mutual nearest neighbors
    admitted_pairs = []
    denied_pairs = []

    for t, s in best_D_for_A.items():
        if best_A_for_D.get(s, None) == t:
            admitted_pairs.append(t)
            denied_pairs.append(s)

    if len(admitted_pairs) == 0:
        return Matching(
            admitted_idx=np.zeros(0, dtype=int),
            denied_idx=np.zeros(0, dtype=int),
            mode=mode,
        )

    admitted_idx = np.array(admitted_pairs, dtype=int)
    denied_idx = np.array(denied_pairs, dtype=int)

    # Sort pairs by admitted index for determinism
    order = np.argsort(admitted_idx)
    admitted_idx = admitted_idx[order]
    denied_idx = denied_idx[order]

    return Matching(
        admitted_idx=admitted_idx,
        denied_idx=denied_idx,
        mode=mode,
    )


# ============================================================
# 3. Estimation result container
# ============================================================

@dataclass
class EstimationResult:
    """
    Output of the algorithm for one Dataset.
    """
    phi_hat: float
    phi_grid: np.ndarray
    U_evo: np.ndarray
    u_evo: np.ndarray

    theta_hat: float
    alpha_hat: float
    gamma_intercept_hat: float
    gamma_slope_hat: float
    G_params: Dict[str, float]  # mean, std of projection

    matching: Matching
    diagnostics: Dict[str, Any]


# ============================================================
# 4. Core algorithm implementation
# ============================================================

def _fit_score_model(Q: np.ndarray, X: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Step 2: regress Q on (1, X), get LSE hat{gamma} and residuals hat{eta}_t.

    Returns:
    - intercept_hat
    - slope_hat
    - Q_hat (projection)
    - eta_hat (residuals)
    """
    n = Q.shape[0]
    X_vec = np.asarray(X).reshape(n,)  # ensure 1d

    # Design matrix with intercept and X
    design = np.column_stack([np.ones(n), X_vec])
    coef, *_ = np.linalg.lstsq(design, Q, rcond=None)
    intercept_hat = float(coef[0])
    slope_hat = float(coef[1])

    Q_hat = design @ coef
    eta_hat = Q - Q_hat

    return intercept_hat, slope_hat, Q_hat, eta_hat


def _fit_plm_theta(
    Y: np.ndarray,
    X: np.ndarray,
    eta_hat: np.ndarray,
    matching: Matching,
) -> Tuple[float, float]:
    """
    Step 5: Estimate theta and alpha from

        Y_t - Y_{s_t} = alpha * eta_t + theta * (X_t - X_{s_t}) + delta_t

    using OLS on the matched pairs.
    """
    t_idx = matching.admitted_idx
    s_idx = matching.denied_idx
    m = t_idx.shape[0]

    if m == 0:
        raise ValueError("No matched pairs available for PLM estimation.")

    Y_diff = Y[t_idx] - Y[s_idx]
    X_diff = X[t_idx] - X[s_idx]
    eta_used = eta_hat[t_idx]

    # Design matrix: [eta_t, X_t - X_{s_t}]
    H = np.column_stack([eta_used, X_diff])

    coef, *_ = np.linalg.lstsq(H, Y_diff, rcond=None)
    alpha_hat = float(coef[0])
    theta_hat = float(coef[1])

    return alpha_hat, theta_hat


def _estimate_G_from_projection(
    gamma_slope_hat: float,
    X: np.ndarray,
) -> NormalProjectionDistribution:
    """
    Step 3: estimate distribution of hat{gamma}^T Z_t = hat_gamma * X_t.

    Here Z = X is scalar, so projection is simply hat_gamma * X.

    We approximate this distribution as N(mu_hat, sigma_hat^2) by
    estimating mean and std from the sample.
    """
    proj = gamma_slope_hat * X
    mu_hat = float(np.mean(proj))
    std_hat = float(np.std(proj))
    return NormalProjectionDistribution(mean=mu_hat, std=std_hat)


def _compute_residuals(Y: np.ndarray, X: np.ndarray, theta_hat: float) -> np.ndarray:
    """
    Define R_t = Y_t - theta_hat * X_t.

    This is intended to estimate W_t 1(Q_t > phi) + beta(eta_t) + noise.
    """
    return Y - theta_hat * X

def _compute_u_evo_at_phi(
    phi: float,
    c: float,
    Y: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    eta_hat: np.ndarray,
    matching: Matching,
    theta_hat: float,
    G_dist: NormalProjectionDistribution,
    n_total: int,
) -> float:
    """
    Scalar version of u_hat_evo(phi):

      u_hat_evo(phi) =
        -(1/n) sum_{t in tilde S_A} (R_t - R_{s_t} - c) * g_hat(phi - eta_hat_t)

    where g_hat is the pdf from G_dist.
    """
    t_idx = matching.admitted_idx
    s_idx = matching.denied_idx
    m = t_idx.shape[0]

    if m == 0:
        # no pairs -> u_hat identically zero (or undefined).
        # We return 0 to avoid crashes; caller should check for num_pairs().
        return 0.0

    R = _compute_residuals(Y, X, theta_hat)

    diff_R_minus_c = R[t_idx] - R[s_idx] - c
    eta_t = eta_hat[t_idx]

    arg = phi - eta_t
    g_vals = G_dist.pdf(arg)
    denom = float(m)

    u_val = -(1.0 / denom) * float(np.sum(diff_R_minus_c * g_vals))
    return u_val


def _bracket_root_u_evo(
    phi_min: float,
    phi_max: float,
    eval_u: callable,
    num_grid: int = 101,
) -> Optional[tuple]:
    """
    Try to find a bracket [a, b] in [phi_min, phi_max] such that
    u(a) and u(b) have opposite signs (i.e., u(a) * u(b) < 0).

    We scan a coarse grid and return the first such bracket.
    If no sign change is found, return None.
    """
    phis = np.linspace(phi_min, phi_max, num_grid)
    u_vals = np.array([eval_u(phi) for phi in phis], dtype=float)

    # Look for sign changes
    signs = np.sign(u_vals)
    for i in range(len(phis) - 1):
        if signs[i] == 0:
            # Exact zero at a grid point -> we can treat it as a "root"
            return (phis[i], phis[i])
        if signs[i] * signs[i + 1] < 0:
            # sign change between phis[i] and phis[i+1]
            return (phis[i], phis[i + 1])

    # No sign change found
    return None


def _bisection_root(
    eval_u: callable,
    a: float,
    b: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Standard bisection method to find a root of eval_u in [a, b],
    assuming eval_u(a) * eval_u(b) <= 0.

    If eval_u(a) or eval_u(b) is zero, returns that endpoint.
    """
    fa = eval_u(a)
    fb = eval_u(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    if fa * fb > 0:
        # no sign change; caller should have checked
        # we just return midpoint as a fallback
        return 0.5 * (a + b)

    left, right = a, b
    f_left, f_right = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = eval_u(mid)

        if abs(f_mid) < tol:
            return mid

        # Decide which half to keep
        if f_left * f_mid < 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid

    # Max iterations: return best midpoint
    return 0.5 * (left + right)

def _compute_utility_and_score_grid(
    phi_grid: np.ndarray,
    c: float,
    Y: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    eta_hat: np.ndarray,
    matching: Matching,
    theta_hat: float,
    G_dist: NormalProjectionDistribution,
    n_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagnostic grid evaluation of U_hat_evo and u_hat_evo.

    This is *not* used to get phi_hat anymore (we use a root-finder for u_hat),
    but it's still useful if you want to visualize the function.
    """
    t_idx = matching.admitted_idx
    s_idx = matching.denied_idx
    m = t_idx.shape[0]

    if m == 0:
        return np.zeros_like(phi_grid, dtype=float), np.zeros_like(phi_grid, dtype=float)

    R = _compute_residuals(Y, X, theta_hat)
    diff_R_minus_c = R[t_idx] - R[s_idx] - c
    eta_t = eta_hat[t_idx]

    U_vals = np.empty_like(phi_grid, dtype=float)
    u_vals = np.empty_like(phi_grid, dtype=float)

    denom = float(m)

    for i, phi_prime in enumerate(phi_grid):
        arg = phi_prime - eta_t
        barG_vals = G_dist.survival(arg)
        g_vals = G_dist.pdf(arg)

        U_vals[i] = (1.0 / denom) * np.sum(diff_R_minus_c * barG_vals)
        u_vals[i] = -(1.0 / denom) * np.sum(diff_R_minus_c * g_vals)

    return U_vals, u_vals


def run_algorithm(
    dataset: Dataset,
    c: float,
    phi_grid: Optional[np.ndarray] = None,
    match_mode: str = "mutual_nn",
) -> EstimationResult:

    """
    Run the full conceptual algorithm on a given Dataset.

    Changes from previous version:
      - We now estimate phi_hat as a *root* of u_hat_evo(phi)
        using 1D root-finding (bisection), instead of grid argmin.
      - Grid-based U_evo/u_evo are kept only for diagnostics if phi_grid is provided.

    Inputs:
    - dataset: Dataset from data.py
    - c: constant appearing in (R_t - R_{s_t} - c)
    - phi_grid: optional grid for diagnostic evaluation of U_evo and u_evo.
    """
    Y = dataset.Y
    Q = dataset.Q
    X = dataset.X
    n = dataset.n

    # 1. Status-quo cutoff phi (used for S_A, S_D)
    phi_status_quo = float(dataset.params.get("phi", 0.0))

    # 2. Fit score model and get eta_hat
    gamma_intercept_hat, gamma_slope_hat, Q_hat, eta_hat = _fit_score_model(Q, X)

    # 3. Estimate G (normal approximation to projection distribution)
    G_dist = _estimate_G_from_projection(gamma_slope_hat, X)
    G_params = {"mean": G_dist.mean, "std": G_dist.std}

    # 4. Compute matching
    matching = compute_matching(
        eta_hat=eta_hat,
        Q=Q,
        phi=phi_status_quo,
        mode=match_mode,
    )

    # 5. Fit PLM to estimate theta and alpha
    alpha_hat, theta_hat = _fit_plm_theta(
        Y=Y,
        X=X,
        eta_hat=eta_hat,
        matching=matching,
    )

    # 6. Define an interval for phi where we'll search for root of u_hat
    q_min = float(np.min(Q))
    q_max = float(np.max(Q))
    span = q_max - q_min
    margin = 0.1 * (span + 1e-8)
    phi_min = q_min - margin
    phi_max = q_max + margin

    # Build scalar u_hat evaluator
    def eval_u(phi_val: float) -> float:
        return _compute_u_evo_at_phi(
            phi=phi_val,
            c=c,
            Y=Y,
            Q=Q,
            X=X,
            eta_hat=eta_hat,
            matching=matching,
            theta_hat=theta_hat,
            G_dist=G_dist,
            n_total=n,
        )

    # 7. Try to bracket a root and apply bisection
    bracket = _bracket_root_u_evo(
        phi_min=phi_min,
        phi_max=phi_max,
        eval_u=eval_u,
        num_grid=101,
    )

    if bracket is not None:
        a, b = bracket
        if a == b:
            # we hit an exact zero on the coarse grid
            phi_hat = a
        else:
            phi_hat = _bisection_root(
                eval_u=eval_u,
                a=a,
                b=b,
                tol=1e-4,
                max_iter=50,
            )
    else:
        # Fallback: no sign change over [phi_min, phi_max].
        # We choose phi_hat that minimizes |u_hat(phi)| over a coarse grid.
        coarse_phis = np.linspace(phi_min, phi_max, 201)
        u_vals_coarse = np.array([eval_u(phi) for phi in coarse_phis])
        idx_hat = int(np.argmin(np.abs(u_vals_coarse)))
        phi_hat = float(coarse_phis[idx_hat])

    # 8. Diagnostic grid for U_evo and u_evo if phi_grid is provided
    if phi_grid is None:
        # create a moderate grid around the search interval
        phi_grid = np.linspace(phi_min, phi_max, 201)

    U_evo_vals, u_evo_vals = _compute_utility_and_score_grid(
        phi_grid=phi_grid,
        c=c,
        Y=Y,
        Q=Q,
        X=X,
        eta_hat=eta_hat,
        matching=matching,
        theta_hat=theta_hat,
        G_dist=G_dist,
        n_total=n,
    )

    diagnostics: Dict[str, Any] = {
        "Q_hat": Q_hat,
        "eta_hat": eta_hat,
        "alpha_hat": alpha_hat,
        "theta_hat": theta_hat,
        "gamma_intercept_hat": gamma_intercept_hat,
        "gamma_slope_hat": gamma_slope_hat,
        "G_params": G_params,
        "matching_num_pairs": matching.num_pairs(),
        "matching_mode": matching.mode,
        "phi_status_quo": phi_status_quo,
    }

    return EstimationResult(
        phi_hat=phi_hat,
        phi_grid=phi_grid,
        U_evo=U_evo_vals,
        u_evo=u_evo_vals,
        theta_hat=theta_hat,
        alpha_hat=alpha_hat,
        gamma_intercept_hat=gamma_intercept_hat,
        gamma_slope_hat=gamma_slope_hat,
        G_params=G_params,
        matching=matching,
        diagnostics=diagnostics,
    )



# ============================================================
# 5. Tiny sanity check (optional)
# ============================================================

if __name__ == "__main__":
    from data import DataGenConfig, generate_dataset

    # Simple test DGP
    cfg = DataGenConfig(
        I0=0,
        I1=0,
        gamma=1,
        theta=1,
        rho=0.5,
        phi=0.0,  # status-quo cutoff
    )

    ds = generate_dataset(n=1000, config=cfg, seed=123)
    res = run_algorithm(dataset=ds, c=1)

    print("Estimated phi_hat:", res.phi_hat)
    print("Theta_hat:", res.theta_hat, "Alpha_hat:", res.alpha_hat)
    print("Number of matched pairs:", res.matching.num_pairs())
