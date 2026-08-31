"""Prespecified welfare-outcome menu for the GPA probation application.

The menu keeps three concepts separate:

* directly observed full-population outcomes;
* inherited missing-GPA sensitivity values; and
* small, explicitly labeled status-adjusted stress tests.

No scenario is selected because it produces a preferred policy threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.gpa.redesign import (
    _make_sample,
    composite_next_gpa,
    load_frame,
)


PHYSICAL_NO_RECORD_GPAS = (0.0, 0.8, 0.9, 1.1, 1.5)


@dataclass(frozen=True)
class WelfareOutcome:
    """One prespecified outcome plus its interpretation metadata."""

    key: str
    label: str
    category: str
    status: str
    units: str
    formula: str
    values: np.ndarray


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"GPA welfare columns are missing: {missing_columns}")


def build_welfare_outcomes(frame: pd.DataFrame) -> Dict[str, WelfareOutcome]:
    """Construct the complete 16-outcome menu without selecting a winner."""
    required = (
        "nextGPA",
        "gpacutoff",
        "fallreg_year2",
        "left_school",
        "credits_earned2",
        "goodstanding_year2",
    )
    _require_columns(
        frame,
        required,
    )
    complete_columns = tuple(column for column in required if column != "nextGPA")
    if frame[list(complete_columns)].isna().any().any():
        raise ValueError(
            f"GPA welfare status columns contain missing values: {complete_columns}"
        )
    fall_return = frame["fallreg_year2"].to_numpy(dtype=float)
    not_left = 1.0 - frame["left_school"].to_numpy(dtype=float)
    recorded = frame["nextGPA"].notna().to_numpy(dtype=float)
    credits = frame["credits_earned2"].to_numpy(dtype=float)
    good_standing = frame["goodstanding_year2"].to_numpy(dtype=float)
    left = frame["left_school"].to_numpy(dtype=float)

    outcomes: Dict[str, WelfareOutcome] = {}

    def add(
        key: str,
        label: str,
        category: str,
        status: str,
        units: str,
        formula: str,
        values: np.ndarray,
    ) -> None:
        array = np.asarray(values, dtype=float)
        if len(array) != len(frame) or not np.isfinite(array).all():
            raise ValueError(f"welfare outcome {key} is incomplete or non-finite")
        outcomes[key] = WelfareOutcome(
            key=key,
            label=label,
            category=category,
            status=status,
            units=units,
            formula=formula,
            values=array,
        )

    add(
        "fall_return",
        "Fall-year-2 enrollment",
        "direct",
        "primary",
        "probability",
        "fallreg_year2",
        fall_return,
    )
    add(
        "not_left_voluntarily",
        "Not voluntarily leaving",
        "direct",
        "primary",
        "probability",
        "1 - left_school",
        not_left,
    )
    add(
        "next_gpa_recorded",
        "Any subsequent GPA record",
        "direct",
        "primary",
        "probability",
        "1{nextGPA is recorded}",
        recorded,
    )
    add(
        "credits_earned_year2",
        "Year-2 credits earned",
        "direct",
        "primary",
        "course-credit units",
        "credits_earned2",
        credits,
    )
    add(
        "good_standing_year2",
        "Good standing in year 2",
        "direct",
        "secondary_pending_coding_provenance",
        "probability",
        "goodstanding_year2",
        good_standing,
    )

    for assumed in PHYSICAL_NO_RECORD_GPAS:
        label_value = f"{assumed:.1f}"
        key_value = label_value.replace(".", "p")
        status = "inherited_sensitivity"
        if assumed == 0.0:
            status = "physical_lower_bound_sensitivity"
        elif assumed == 1.5:
            status = "institutional_cutoff_benchmark"
        add(
            f"composite_no_record_gpa_{key_value}",
            f"Composite GPA; no record = {label_value}",
            "missing_gpa_sensitivity",
            status,
            "GPA points relative to cutoff",
            f"observed nextGPA; otherwise {label_value} - gpacutoff",
            composite_next_gpa(frame, assumed),
        )

    for assumed in (0.8, 1.1):
        base = composite_next_gpa(frame, assumed)
        for penalty in (0.10, 0.25):
            a_key = f"{assumed:.1f}".replace(".", "p")
            p_key = f"{penalty:.2f}".replace(".", "p")
            add(
                f"composite_a{a_key}_leave_penalty_{p_key}",
                f"Composite a={assumed:.1f}; leave penalty {penalty:.2f}",
                "status_adjusted_stress",
                "stress_test",
                "GPA-equivalent welfare points",
                (
                    f"composite(a={assumed:.1f}) - {penalty:.2f} * "
                    "left_school"
                ),
                base - penalty * left,
            )

    base = composite_next_gpa(frame, 0.8)
    for bonus in (0.10, 0.25):
        b_key = f"{bonus:.2f}".replace(".", "p")
        add(
            f"composite_a0p8_return_bonus_{b_key}",
            f"Composite a=0.8; return bonus {bonus:.2f}",
            "status_adjusted_stress",
            "stress_test",
            "GPA-equivalent welfare points",
            f"composite(a=0.8) + {bonus:.2f} * fallreg_year2",
            base + bonus * fall_return,
        )

    if len(outcomes) != 16:
        raise AssertionError(f"expected 16 welfare outcomes, constructed {len(outcomes)}")
    return outcomes


def load_welfare_menu(frame: pd.DataFrame | None = None) -> Dict[str, RDDSample]:
    """Return all welfare outcomes as full-population RDD samples."""
    data = load_frame() if frame is None else frame
    outcomes = build_welfare_outcomes(data)
    return {
        key: _make_sample(
            data,
            outcome.values,
            name=f"gpa_welfare_{key}",
            description=outcome.label,
            extras={
                "welfare_label": outcome.label,
                "welfare_category": outcome.category,
                "welfare_status": outcome.status,
                "welfare_units": outcome.units,
                "welfare_formula": outcome.formula,
            },
        )
        for key, outcome in outcomes.items()
    }
