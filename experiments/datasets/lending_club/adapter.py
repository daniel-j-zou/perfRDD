"""Lending Club loan-level data — STUB.

Score Q = FICO score (composite of credit history); rich relative to X
(borrower demographics + loan terms). LC used hard FICO floors (e.g. 660,
later 600) for eligibility, plus discrete grade-bucket cutoffs.
"""
from __future__ import annotations

from pathlib import Path

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "loans.csv"


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. See "
            "experiments/datasets/lending_club/README.md for download "
            "instructions."
        )
    raise NotImplementedError(
        "lending_club adapter not yet wired up; choose Q (e.g. fico_range_low) "
        "and the threshold (660 / 600) once the loan archive is downloaded."
    )
