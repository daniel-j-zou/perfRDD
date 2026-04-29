"""MIMIC-IV ICU severity-score RDD — STUB (DUA-restricted).

Q = ICU severity score (SOFA / SAPS-II / APACHE-IV);
treatment threshold = e.g. SOFA >= 10 (high mortality risk);
Y = ICU length of stay or ventilator-free days.

MIMIC-IV requires PhysioNet credentialing (CITI training + DUA).
Cannot be auto-downloaded.
"""
from __future__ import annotations

from pathlib import Path

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "mimic.csv"


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. MIMIC-IV requires PhysioNet credentialing "
            "(CITI training + DUA). See "
            "experiments/datasets/mimic/README.md for the access process."
        )
    raise NotImplementedError(
        "mimic adapter not yet wired up; pick Q (e.g. SOFA score on day 1) "
        "and the threshold once your extract is downloaded."
    )
