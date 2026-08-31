"""Policy-oriented outcome definitions for the academic-probation data.

The published ``nextGPA`` outcome is only observed when a student has a GPA
at a subsequent evaluation.  Because academic probation affects persistence,
silently dropping rows with a missing ``nextGPA`` conditions on a
post-treatment variable.  This module therefore exposes three distinct types
of outcomes instead of conflating them:

* full-population persistence outcomes;
* ``nextGPA`` among students for whom it is recorded (diagnostic only); and
* a full-population composite outcome whose value for "no subsequent GPA" is
  explicit and varied in sensitivity analysis.

``nextGPA`` is already measured as subsequent absolute GPA minus the
student's campus-specific probation cutoff.  If ``a`` is the absolute GPA
assigned to the no-record state, the corresponding composite value is
``a - gpacutoff``.  The original replication do-file mentions sensitivity
variables based on absolute GPAs 0.0, 0.8, 0.9, and 1.1; those values are the
defaults here.  This construction applies to every missing ``nextGPA`` row,
not just rows marked ``left_school``.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.gpa.adapter import (
    DATA_PATH,
    Q_COL,
    X_COLS,
    Y_COL,
    _below_zero,
)


DEFAULT_NO_GRADE_ABSOLUTE_GPAS = (0.0, 0.8, 0.9, 1.1, 1.5)
DEFAULT_NO_GRADE_PENALTIES = (2.0, 4.0, 5.0, 6.0, 8.0)


def load_frame() -> pd.DataFrame:
    """Load and validate the common full analysis frame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. See experiments/datasets/gpa/README.md."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(X_COLS) | {
        Q_COL,
        Y_COL,
        "gpacutoff",
        "fallreg_year2",
        "left_school",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"GPA data are missing required columns: {missing}")
    if df[[Q_COL, "gpacutoff", *X_COLS]].isna().any().any():
        raise ValueError("GPA score, cutoff, or baseline covariates contain missing values")
    for column in ("fallreg_year2", "left_school"):
        values = set(df[column].dropna().unique())
        if values != {0, 1}:
            raise ValueError(f"{column} is not a complete binary variable: {values}")
    return df


def composite_next_gpa(
    df: pd.DataFrame,
    no_grade_absolute_gpa: float,
    no_grade_penalty: float = 0.0,
) -> np.ndarray:
    """Define subsequent-GPA value on the full population.

    Observed values retain the published outcome.  A row without a subsequent
    evaluation receives
    ``no_grade_absolute_gpa - gpacutoff - no_grade_penalty`` so that both
    states are expressed in GPA-equivalent units.  The penalty is a welfare
    valuation of failing to produce a subsequent academic record, not a
    literal GPA and not an estimate recovered from the data.
    """
    assumed = float(no_grade_absolute_gpa)
    if not np.isfinite(assumed) or assumed < 0.0 or assumed > 4.3:
        raise ValueError("no_grade_absolute_gpa must be finite and in [0, 4.3]")
    penalty = float(no_grade_penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("no_grade_penalty must be finite and nonnegative")
    outcome = df[Y_COL].to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(outcome)
    outcome[missing] = (
        assumed
        - df.loc[missing, "gpacutoff"].to_numpy(dtype=float)
        - penalty
    )
    return outcome


def _make_sample(
    df: pd.DataFrame,
    outcome: np.ndarray,
    *,
    name: str,
    description: str,
    extras: Dict[str, object] | None = None,
) -> RDDSample:
    return RDDSample(
        Q=df[Q_COL].to_numpy(dtype=float),
        X=df[X_COLS].to_numpy(dtype=float),
        Y=np.asarray(outcome, dtype=float),
        threshold=0.0,
        name=name,
        feature_names=list(X_COLS),
        description=description,
        citation="Lindo, Sanders, Oreopoulos (2010), AEJ: Applied Economics",
        treatment_rule=_below_zero,
        extras={} if extras is None else extras,
    )


def load_fall_return(df: pd.DataFrame | None = None) -> RDDSample:
    """Whether the student registered in the fall of year two (full sample)."""
    frame = load_frame() if df is None else df
    return _make_sample(
        frame,
        frame["fallreg_year2"].to_numpy(dtype=float),
        name="gpa_fall_return",
        description=(
            "Academic probation RDD; full-population outcome equals one when "
            "the student registered in the fall of year two."
        ),
        extras={"outcome_population": "full", "outcome_kind": "persistence"},
    )


def load_not_left_voluntarily(df: pd.DataFrame | None = None) -> RDDSample:
    """One minus the replication file's voluntary-leaving indicator."""
    frame = load_frame() if df is None else df
    outcome = 1.0 - frame["left_school"].to_numpy(dtype=float)
    return _make_sample(
        frame,
        outcome,
        name="gpa_not_left_voluntarily",
        description=(
            "Academic probation RDD; full-population outcome equals one unless "
            "the student is recorded as voluntarily leaving the university."
        ),
        extras={"outcome_population": "full", "outcome_kind": "persistence"},
    )


def load_next_gpa_recorded(df: pd.DataFrame | None = None) -> RDDSample:
    """Whether a subsequent GPA is recorded (full sample)."""
    frame = load_frame() if df is None else df
    outcome = frame[Y_COL].notna().to_numpy(dtype=float)
    return _make_sample(
        frame,
        outcome,
        name="gpa_next_gpa_recorded",
        description=(
            "Academic probation RDD; full-population outcome equals one when a "
            "GPA at a subsequent evaluation is recorded."
        ),
        extras={"outcome_population": "full", "outcome_kind": "observation_or_progress"},
    )


def load_observed_next_gpa(df: pd.DataFrame | None = None) -> RDDSample:
    """Published subsequent-GPA outcome, explicitly labeled as selected."""
    frame = load_frame() if df is None else df
    observed = frame[Y_COL].notna()
    selected = frame.loc[observed].copy()
    return _make_sample(
        selected,
        selected[Y_COL].to_numpy(dtype=float),
        name="gpa_observed_next_gpa_diagnostic",
        description=(
            "Academic probation RDD; subsequent GPA minus the applicable cutoff "
            "among students with a recorded subsequent evaluation. This is a "
            "post-treatment-selected diagnostic, not a full-population effect."
        ),
        extras={
            "outcome_population": "recorded_next_gpa_only",
            "post_treatment_selected": True,
        },
    )


def load_composite_next_gpa(
    no_grade_absolute_gpa: float,
    df: pd.DataFrame | None = None,
    no_grade_penalty: float = 0.0,
) -> RDDSample:
    """Full-population subsequent-GPA composite for a stated no-grade value."""
    frame = load_frame() if df is None else df
    assumed = float(no_grade_absolute_gpa)
    penalty = float(no_grade_penalty)
    outcome = composite_next_gpa(frame, assumed, penalty)
    assumed_label = f"{assumed:.2f}".replace(".", "p")
    penalty_label = f"{penalty:.2f}".replace(".", "p")
    return _make_sample(
        frame,
        outcome,
        name=f"gpa_composite_no_grade_{assumed_label}_penalty_{penalty_label}",
        description=(
            "Academic probation RDD; observed subsequent GPA minus cutoff when "
            "recorded, and assumed absolute GPA "
            f"{assumed:.2f} minus the student's cutoff and a GPA-equivalent "
            f"no-record penalty of {penalty:.2f} otherwise."
        ),
        extras={
            "outcome_population": "full",
            "outcome_kind": "composite",
            "no_grade_absolute_gpa": assumed,
            "no_grade_penalty": penalty,
            "n_no_grade": int(frame[Y_COL].isna().sum()),
        },
    )


def load_redesign_bundle(
    no_grade_absolute_gpas: Iterable[float] = DEFAULT_NO_GRADE_ABSOLUTE_GPAS,
    no_grade_penalties: Iterable[float] = DEFAULT_NO_GRADE_PENALTIES,
) -> Dict[str, RDDSample]:
    """Return all primary, sensitivity, and diagnostic outcome samples."""
    df = load_frame()
    samples: Dict[str, RDDSample] = {
        "fall_return": load_fall_return(df),
        "not_left_voluntarily": load_not_left_voluntarily(df),
        "next_gpa_recorded": load_next_gpa_recorded(df),
    }
    for assumed in no_grade_absolute_gpas:
        sample = load_composite_next_gpa(float(assumed), df)
        samples[f"composite_no_grade_{float(assumed):.2f}"] = sample
    for penalty in no_grade_penalties:
        sample = load_composite_next_gpa(
            no_grade_absolute_gpa=0.0,
            no_grade_penalty=float(penalty),
            df=df,
        )
        samples[f"composite_zero_gpa_penalty_{float(penalty):.2f}"] = sample
    samples["observed_next_gpa_diagnostic"] = load_observed_next_gpa(df)
    return samples
