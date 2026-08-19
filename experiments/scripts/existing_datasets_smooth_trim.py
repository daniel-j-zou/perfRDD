"""Run symmetric smooth support trimming on every available 1-D dataset.

Outputs are written below ``experiments/runs/smooth_trim_existing``.  Missing
restricted datasets are recorded as skips rather than silently invented or
downloaded.  The adapter registry is the source of truth for what is already
present locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from experiments._core.registry import list_datasets, load
from experiments.methods.perfrdd_smooth_trim import perfrdd_smooth_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "smooth_trim_existing"
OUT_JSON = OUT_ROOT / "summary.json"


def main() -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for name in list_datasets():
        try:
            sample = load(name)
        except FileNotFoundError as exc:
            skipped[name] = str(exc)
            print(f"[skip] {name}: data not present")
            continue
        if sample.k != 1:
            skipped[name] = "smooth-trim implementation currently requires a one-dimensional running variable"
            print(f"[skip] {name}: running variable is {sample.k}-dimensional")
            continue
        print(f"[run] {name}: n={sample.n:,}, p={sample.p}")
        try:
            results[name] = perfrdd_smooth_trim(
                sample,
                OUT_ROOT / name,
                eps=0.1,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"[skip] {name}: {exc}")

    payload = {
        "description": "Symmetric smooth implementation of the hard support-trimmed PerfRDD estimand",
        "eps": 0.1,
        "delta_rate": "window_width * n^(-1/3)",
        "knot_rate": "n^(11/60)",
        "results": results,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[wrote] {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
