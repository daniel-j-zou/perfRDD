#!/usr/bin/env python
"""Run a method across every dataset whose data is present.

Usage (from the perfrdd repo root):
    python -m experiments.scripts.run_all                   # default: summary method
    python -m experiments.scripts.run_all --method summary  # explicit
    python -m experiments.scripts.run_all --only gpa hmda   # restrict
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from experiments._core.runner import run_all


def _resolve_method(name: str):
    mod = importlib.import_module(f"experiments.methods.{name}")
    fn = getattr(mod, "run", None) or getattr(mod, name, None)
    if fn is None:
        raise AttributeError(
            f"experiments.methods.{name} must define `run()` or `{name}()`"
        )
    return fn


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="summary", help="module name in experiments/methods/")
    p.add_argument("--only", nargs="*", default=None, help="dataset names")
    p.add_argument("--out", default=None, help="path to JSON output")
    args = p.parse_args(argv)

    method = _resolve_method(args.method)
    results = run_all(method, only=args.only)

    payload = json.dumps(results, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload)
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
