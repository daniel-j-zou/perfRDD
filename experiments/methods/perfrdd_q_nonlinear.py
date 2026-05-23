"""perfrdd with a nonparametric Q ~ X first stage (Y still linear in X)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import perfrdd


def run(sample: RDDSample) -> Dict[str, Any]:
    out = Path(__file__).resolve().parent.parent / "runs" / "perfrdd_q_nonlinear" / sample.name
    return perfrdd(sample, out, first_stage="q_nonlinear")
