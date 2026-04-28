"""Taxi cab "Default Tips" — STUB.

Score Q is the default-tip-suggestion threshold; running variable typically
the fare. Outcome Y is the chosen tip. See README for details and
download instructions.
"""
from __future__ import annotations

from pathlib import Path

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "trips.csv"


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. See "
            "experiments/datasets/taxi/README.md for download instructions."
        )
    raise NotImplementedError(
        "taxi adapter not yet wired up; see Haggag & Paci (2014) data."
    )
