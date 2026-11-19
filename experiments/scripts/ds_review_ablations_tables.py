"""Compact overview of the review-ablation summaries (``ds_review_ablations.py``).

Prints, for every design and n, the rotated average's coverage (percentile / normal), its
bias in Monte Carlo standard deviations, the ratio of the mean bootstrap variance to the
Monte Carlo variance, its mean against the target, and the share of rotated thresholds
on the edge of Phi; the group bootstrap for ``cluster_g31``; the best-cell check for
``grid_h27``.

Run (from code/): python -m experiments.scripts.ds_review_ablations_tables <run dir>
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(out: Path) -> None:
    print("| design | n | B | coverage pct / normal | bias / sd | boot var / MC var | mean (target) "
          "| edge share |")
    print("|---|---:|---:|---|---:|---:|---|---:|")
    for path in sorted(out.glob("*_summary.json")):
        res = json.loads(path.read_text())
        d, phi = res["design"], res["population"]["phi_star"]
        for n, cell in sorted(res["cells"].items(), key=lambda kv: int(kv[0])):
            n = int(n)
            rows = [("units", cell)]
            if "group_bootstrap" in cell:
                rows.append(("groups", cell["group_bootstrap"]))
            for label, c in rows:
                a = c["arms"]["avg"]
                sd = np.sqrt(a["n_var"] / n)
                edge = cell["edge"]["rotation_thresholds_at_edge"] / (4 * cell["reps"])
                tag = d if label == "units" else f"{d} (groups)"
                print(f"| {tag} | {n:,} | {c['boot_draws_per_rep']} | "
                      f"{a['coverage_percentile']:.3f} / {a['coverage_normal']:.3f} | "
                      f"{a['bias'] / sd:+.2f} | {a['boot_var_ratio']:.2f} | "
                      f"{phi + a['bias']:.4f} ({phi:.4f}) | {edge:.3f} |")
            if "coverage_best_cell_percentile" in cell:
                print(f"| {d} best cell | {n:,} | | covers {cell['coverage_best_cell_percentile']:.3f}"
                      f"; picks it {cell['share_best_cell']:.3f}; mean regret "
                      f"{cell['mean_regret_share']:.4f} of the gain | | | | |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
