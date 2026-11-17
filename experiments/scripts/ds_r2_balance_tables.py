"""Tables of the R^2 / treated-share coverage run (``ds_r2_balance_coverage.py``).

Reads ``<run>/<design>_summary.json`` and prints, for the rotated average (and the
oracle), a markdown overview and the LaTeX rows of the two Appendix 7 tables:
coverage of the 95% percentile and normal intervals, and the bias in standard
deviations with the ratio of the mean bootstrap variance to the Monte Carlo variance.

Run (from code/):
    python -m experiments.scripts.ds_r2_balance_tables experiments/runs/ds_r2_balance_coverage_999
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ORDER = [f"r2_{r}_{b}" for b in ("bal", "unbal") for r in ("10", "30", "50", "70", "90")]
ORDER.append("taxi_like")


def load(run: Path) -> dict:
    out = {}
    for key in ORDER:
        p = run / f"{key}_summary.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def rows(res: dict):
    for key in ORDER:
        if key not in res:
            continue
        r = res[key]
        pop = r["population"]
        for n, cell in sorted(r["cells"].items(), key=lambda kv: int(kv[0])):
            a, o = cell["arms"]["avg"], cell["arms"]["oracle_avg"]
            e = cell["edge"]
            yield {
                "key": key, "r2": pop["r2"], "balance": pop["balance"], "n": int(n),
                "reps": cell["reps"], "bias_sd": a["sqrt_n_bias"] / np.sqrt(a["n_var"]),
                "bias": a["sqrt_n_bias"], "bias_se": a["sqrt_n_bias_se"], "n_var": a["n_var"],
                "n_boot_var": a["mean_n_boot_var"], "ratio": a["boot_var_ratio"],
                "V": pop["V_rotated"], "pct": a["coverage_percentile"],
                "nrm": a["coverage_normal"], "pct_pooled": cell["arms"]["pooled"]["coverage_percentile"],
                "nrm_pooled": cell["arms"]["pooled"]["coverage_normal"],
                "o_pct": o["coverage_percentile"], "o_nrm": o["coverage_normal"],
                "o_bias_sd": o["sqrt_n_bias"] / np.sqrt(o["n_var"]),
                "edge_rot": e["rotation_thresholds_at_edge"] / (4 * cell["reps"]),
                "edge_draws": e["boot_draws_at_edge"] / max(e["boot_draws_total"], 1),
                "failures": cell["boot_failures"],
            }


def markdown(res: dict) -> str:
    lines = ["| design | n | bias/sd | n Var | n Var* | ratio | V | pct | normal | pooled pct/nrm "
             "| oracle pct/nrm | edge rot / draws | fail |", "|---|---:|" + "---:|" * 11]
    for r in rows(res):
        V = "-" if r["V"] is None else f"{r['V']:.1f}"
        lines.append(
            f"| {r['key']} | {r['n']:,} | {r['bias_sd']:+.2f} | {r['n_var']:.1f} | {r['n_boot_var']:.1f} "
            f"| {r['ratio']:.2f} | {V} | {r['pct']:.3f} | {r['nrm']:.3f} | {r['pct_pooled']:.3f}/"
            f"{r['nrm_pooled']:.3f} | {r['o_pct']:.3f}/{r['o_nrm']:.3f} | {r['edge_rot']:.3f} / "
            f"{r['edge_draws']:.3f} | {r['failures']} |")
    return "\n".join(lines)


def latex(res: dict, ns=(4000, 16000, 64000, 256000)) -> str:
    by = {}
    for r in rows(res):
        by[(r["key"], r["n"])] = r
    out = ["% coverage: percentile / normal, rotated average"]
    for key in ORDER[:-1]:
        cells = [by.get((key, n)) for n in ns]
        r2 = key.split("_")[1]
        out.append(f"0.{r2[0]} & " + " & ".join(
            "--" if c is None else f"{c['pct']:.3f} / {c['nrm']:.3f}" for c in cells) + r"\\")
    out.append("% bias in standard deviations; mean bootstrap variance over Monte Carlo variance")
    for key in ORDER[:-1]:
        cells = [by.get((key, n)) for n in ns]
        r2 = key.split("_")[1]
        out.append(f"0.{r2[0]} & " + " & ".join(
            "--" if c is None else f"${c['bias_sd']:+.2f}$ / {c['ratio']:.2f}" for c in cells) + r"\\")
    return "\n".join(out)


def main(argv=None):
    run = Path((argv or sys.argv[1:])[0])
    res = load(run)
    print(markdown(res))
    print()
    print(latex(res))


if __name__ == "__main__":
    main()
