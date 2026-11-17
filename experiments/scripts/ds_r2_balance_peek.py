"""Interim summary of an unfinished ``ds_r2_balance_coverage`` run, from its shard rows.

Reads ``<run>/shards/<design>_n<n>_r*/<design>_n<n>.jsonl``, summarizes every
(design, n) cell with the replications finished so far, and prints the overview of
``ds_r2_balance_tables``. For monitoring only; the paper uses the merged run.

Run (from code/, on a compute node):
    python -m experiments.scripts.ds_r2_balance_peek experiments/runs/ds_r2_balance_coverage_999
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from experiments.scripts import ds_r2_balance_tables as TB
from experiments.scripts.ds_r2_balance_coverage import SPECS, _summarize, population


def main(argv=None):
    run = Path((argv or sys.argv[1:])[0])
    cells = defaultdict(list)
    for f in sorted((run / "shards").glob("*/*.jsonl")):
        key, n = f.stem.rsplit("_n", 1)
        for line in f.read_text().splitlines():
            if line.strip():
                cells[(key, int(n))].append(json.loads(line))
    res = {}
    for (key, n), rows in sorted(cells.items()):
        if len(rows) < 20:
            continue
        if key not in res:
            res[key] = {"population": population(SPECS[key]), "cells": {}}
        rows = sorted(rows, key=lambda r: r["rep"])
        cell = _summarize(rows, n, res[key]["population"])
        cell["edge"] = {
            "rotation_thresholds_at_edge": sum(r["edge_rotations"] for r in rows),
            "pooled_at_edge": sum(r["edge_pooled"] for r in rows),
            "boot_draws_at_edge": sum(r["boot_draws_at_edge"] for r in rows),
            "boot_draws_total": sum(len(r["boot"]["avg"]) for r in rows)}
        res[key]["cells"][str(n)] = cell
    print(TB.markdown(res).replace("| design | n |", "| design | n |", 1))
    print("\nreplications per cell: " + ", ".join(
        f"{k}@{n}:{len(v)}" for (k, n), v in sorted(cells.items())))


if __name__ == "__main__":
    main()
