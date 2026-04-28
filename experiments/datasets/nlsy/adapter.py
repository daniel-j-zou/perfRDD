"""NLSY — AFQT score with downstream labor outcomes — STUB.

Score Q = AFQT percentile (composite of four ASVAB subtests).
X = parental education, race, region, family income, schooling.
Y = log hourly wage (or annual earnings) at a chosen reference age.
"""
from __future__ import annotations

from pathlib import Path

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "nlsy.csv"


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. See "
            "experiments/datasets/nlsy/README.md for download instructions."
        )
    raise NotImplementedError(
        "nlsy adapter not yet wired up; pick a threshold (e.g. AFQT 30 for "
        "AFQT-III/Cat-IV split) once the extract is downloaded."
    )
