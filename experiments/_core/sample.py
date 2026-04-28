"""The single contract every dataset adapter must satisfy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

Threshold = Union[float, Tuple[float, ...]]


@dataclass
class RDDSample:
    """A regression-discontinuity sample.

    The shape conventions handle both single-threshold and
    multi-threshold designs:
      single:  Q has shape (n,);   threshold is a scalar
      multi:   Q has shape (n, k); threshold is a length-k tuple

    Default treatment rule: D = 1 when Q is strictly above the
    threshold on every running variable. Override via
    `treatment_rule(Q) -> ndarray` for non-AND rules.
    """

    Q: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    threshold: Threshold
    name: str
    feature_names: List[str]
    description: str = ""
    citation: str = ""
    treatment_rule: Optional[Callable[[np.ndarray], np.ndarray]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.Y)
        if len(self.Q) != n:
            raise ValueError(f"Q has length {len(self.Q)} but Y has length {n}")
        if len(self.X) != n:
            raise ValueError(f"X has length {len(self.X)} but Y has length {n}")
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {self.X.shape}")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {self.X.shape[1]} columns but feature_names has "
                f"{len(self.feature_names)} entries"
            )

    @property
    def n(self) -> int:
        return len(self.Y)

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def k(self) -> int:
        return 1 if self.Q.ndim == 1 else self.Q.shape[1]

    @property
    def D(self) -> np.ndarray:
        """Treatment indicator. AND rule across running variables by default."""
        if self.treatment_rule is not None:
            return np.asarray(self.treatment_rule(self.Q)).astype(int)
        Q = self.Q[:, None] if self.Q.ndim == 1 else self.Q
        thr = np.atleast_1d(np.asarray(self.threshold, dtype=float))
        if thr.shape[0] != Q.shape[1]:
            raise ValueError(
                f"threshold has length {thr.shape[0]} but Q has {Q.shape[1]} columns"
            )
        return (Q > thr[None, :]).all(axis=1).astype(int)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "p": self.p,
            "k": self.k,
            "threshold": self.threshold,
            "n_treated": int(self.D.sum()),
            "n_control": int(self.n - self.D.sum()),
        }
