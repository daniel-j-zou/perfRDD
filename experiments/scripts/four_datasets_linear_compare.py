"""Compose a four-dataset comparison figure for the LINEAR perfrdd setting.

Per-row layout (one row per dataset):
    [ stats text ][ alpha.png ][ utility.png ]

Reads existing per-dataset PNGs from experiments/runs/perfrdd/<name>/ and
summary stats from experiments/runs/perfrdd/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "perfrdd"
SUMMARY = RUNS / "summary.json"

DATASETS = [
    ("gpa",          "GPA — academic probation",       "treated when Q < 0"),
    ("nhanes",       "NHANES — HbA1c diabetic cutoff", "treated when Q ≥ 6.5"),
    ("oulad",        "OULAD — first-TMA pass mark",    "treated when Q ≥ 40"),
    ("lending_club", "Lending Club — DTI trigger",     "treated when Q ≥ 30"),
]

OUT = ROOT / "runs" / "four_datasets_linear_compare.png"


def _stats_block(s: dict, treatment_txt: str) -> str:
    n_used = s["n_used"]
    n_tr = s["n_treated"]
    r2 = s["first_stage_R2"]
    avg_alpha = s["avg_alpha"]
    phi_stars = s["phi_star"]
    # Costs are sorted strings ("0.0", ...) — keep insertion order.
    phi_lines = "\n".join(
        f"  c={float(c):.3g}:  φ*={v:.3g}" for c, v in phi_stars.items()
    )
    return (
        f"n = {n_used:,}    treated = {n_tr:,}\n"
        f"{treatment_txt}\n"
        f"\n"
        f"first-stage R²  = {r2:.3f}\n"
        f"avg α(η)        = {avg_alpha:.3g}\n"
        f"\n"
        f"{phi_lines}"
    )


def main() -> None:
    summary = json.loads(SUMMARY.read_text())

    n_rows = len(DATASETS)
    fig = plt.figure(figsize=(20, 3.6 * n_rows))
    # 3 columns: stats text, alpha image, utility image.
    # Make the image columns wider than the text column.
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=3,
        width_ratios=[1.0, 3.0, 2.2],
        hspace=0.18, wspace=0.05,
        left=0.02, right=0.99, top=0.94, bottom=0.02,
    )

    fig.suptitle(
        "Performative RDD — Linear setting, four datasets\n"
        r"η = Q − γ̂ᵀX,   α(η) = treatment-effect curve,   U(φ) = utility under cost grid",
        fontsize=15, fontweight="bold", y=0.985,
    )

    for i, (name, title, treatment_txt) in enumerate(DATASETS):
        s = summary[name]

        # Stats panel.
        ax_txt = fig.add_subplot(gs[i, 0])
        ax_txt.axis("off")
        ax_txt.text(
            0.02, 0.95, title,
            transform=ax_txt.transAxes,
            fontsize=12, fontweight="bold", va="top",
        )
        ax_txt.text(
            0.02, 0.78, _stats_block(s, treatment_txt),
            transform=ax_txt.transAxes,
            fontsize=9.5, family="monospace", va="top",
        )

        # Alpha image.
        ax_a = fig.add_subplot(gs[i, 1])
        ax_a.imshow(mpimg.imread(RUNS / name / "alpha.png"))
        ax_a.axis("off")

        # Utility image.
        ax_u = fig.add_subplot(gs[i, 2])
        ax_u.imshow(mpimg.imread(RUNS / name / "utility.png"))
        ax_u.axis("off")

    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
