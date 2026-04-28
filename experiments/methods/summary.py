"""A trivial reference method: report the sample summary.

Useful as a smoke test for the runner and as a template for new methods.
A method is just a function `RDDSample -> Any`.
"""
from __future__ import annotations

from typing import Any, Dict

from experiments._core.sample import RDDSample


def summary(sample: RDDSample) -> Dict[str, Any]:
    return sample.summary()
