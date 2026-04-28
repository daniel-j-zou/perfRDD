from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


# ============================================================
# 1. Dataset class
# ============================================================

@dataclass
class Dataset:
    """
    Container for simulated data.

    For now, tailored to the model:
        X ~ N(0, 1)
        eta ~ N(0, 1)
        eps ~ N(0, 1)
        W | eta ~ N(eta, 1)

        Q = I1 + gamma * X + eta
        Y = I0 + W * 1(Q > phi) + theta * X + rho * eta + eps

    But the structure is general enough to extend later.
    """

    # Observed variables
    Y: np.ndarray  # shape (n,)
    Q: np.ndarray  # shape (n,)
    W: np.ndarray  # shape (n,)
    X: np.ndarray  # shape (n,)

    # Latent / noise terms (useful for diagnostics, but not needed by estimator)
    eta: np.ndarray  # shape (n,)
    eps: np.ndarray  # shape (n,)

    # Parameters used to generate the data
    params: Dict[str, Any] = field(default_factory=dict)

    # Miscellaneous metadata (sample size, seed, DGP id, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Sample size."""
        return self.Y.shape[0]

    def as_observed_dict(self) -> Dict[str, np.ndarray]:
        """
        Return only the 'observed' variables as a dict.
        This is what the algorithm will typically see.
        """
        return {
            "Y": self.Y,
            "Q": self.Q,
            "W": self.W,
            "X": self.X,
        }

    def as_full_dict(self) -> Dict[str, Any]:
        """
        Return all stored variables (observed + latent + params + meta).
        Handy for debugging or saving.
        """
        return {
            "Y": self.Y,
            "Q": self.Q,
            "W": self.W,
            "X": self.X,
            "eta": self.eta,
            "eps": self.eps,
            "params": self.params,
            "meta": self.meta,
        }


# ============================================================
# 2. Data-generation configuration
# ============================================================

@dataclass
class DataGenConfig:
    """
    Configuration for the DGP

        X ~ N(0, 1)
        eta ~ N(0, 1)
        eps ~ N(0, 1)
        W | eta ~ N(eta, 1)

        Q = I1 + gamma * X + eta
        Y = I0 + W * 1(Q > phi) + theta * X + rho * eta + eps.

    We parameterize I0, I1, gamma, theta, rho, phi.

    This is easy to generalize later by:
    - allowing different variances,
    - allowing non-normal distributions,
    - making X vector-valued, etc.
    """

    I0: float = 0.0
    I1: float = 0.0
    gamma: float = 1.0
    theta: float = 1.0
    rho: float = 0.0
    phi: float = 0.0

    # For generalization: you can add noise scales, alternative distributions, etc.
    sigma_X: float = 1.0
    sigma_eta: float = 1.0
    sigma_eps: float = 1.0
    sigma_W_cond: float = 1.0  # std dev of W | eta

    dgp_id: str = "baseline_scalar_normal"


# ============================================================
# 3. Data-generation function
# ============================================================

def generate_dataset(
        n: int,
        config: DataGenConfig,
        seed: Optional[int] = None,
) -> Dataset:
    """
    Generate a Dataset object with n observations from the specified DGP.

    Model:
        X ~ N(0, sigma_X^2)
        eta ~ N(0, sigma_eta^2)
        eps ~ N(0, sigma_eps^2)
        W | eta ~ N(eta, sigma_W_cond^2)

        Q = I1 + gamma * X + eta
        Y = I0 + W * 1(Q > phi) + theta * X + rho * eta + eps
    """
    rng = np.random.default_rng(seed)

    # 1. Draw basic random variables
    X = rng.normal(loc=0.0, scale=config.sigma_X, size=n)
    eta = rng.normal(loc=0.0, scale=config.sigma_eta, size=n)
    eps = rng.normal(loc=0.0, scale=config.sigma_eps, size=n)

    # 2. W given eta: W_i ~ N(eta_i, sigma_W_cond^2)
    W = rng.normal(loc=eta, scale=config.sigma_W_cond, size=n)

    # 3. Score Q and outcome Y
    Q = config.I1 + config.gamma * X + eta
    treatment = (Q > config.phi).astype(float)
    Y = (
            config.I0
            + W * treatment
            + config.theta * X
            + config.rho * eta
            + eps
    )

    # 4. Pack parameters and meta info
    params = {
        "I0": config.I0,
        "I1": config.I1,
        "gamma": config.gamma,
        "theta": config.theta,
        "rho": config.rho,
        "phi": config.phi,
        "sigma_X": config.sigma_X,
        "sigma_eta": config.sigma_eta,
        "sigma_eps": config.sigma_eps,
        "sigma_W_cond": config.sigma_W_cond,
    }

    meta = {
        "n": n,
        "dgp_id": config.dgp_id,
        "seed": seed,
    }

    # 5. Construct Dataset
    return Dataset(
        Y=Y,
        Q=Q,
        W=W,
        X=X,
        eta=eta,
        eps=eps,
        params=params,
        meta=meta,
    )


# ============================================================
# 4. Tiny sanity check (optional)
# ============================================================

if __name__ == "__main__":
    cfg = DataGenConfig(
        phi=-9,
    )
    data = generate_dataset(n=1000, config=cfg, seed=42)
    print("n =", data.n)
    print("Y mean ~", np.mean(data.Y))
    print("Q mean ~", np.mean(data.Q))
    print("Proportion treated ~", np.mean(data.Q > cfg.phi))
