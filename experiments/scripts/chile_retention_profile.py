"""Profile the MINEDUC student-performance files, one row per year.

Reports record counts, final-status counts (P promoted, R retained, Y withdrawn),
missing MRUN and off-grid averages, counts at averages 4.4/4.5/4.6 and 4.9/5.0/5.1,
heaping ratios n(c) / sqrt(n(c-0.1) * n(c+0.1)), and the retained share at each side
of the 4.5 and 5.0 cutoffs. Sample: SIT_FIN in {P, R} with PROM_GRAL > 0.

    python experiments/scripts/chile_retention_profile.py OUT.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "datasets" / "chile_retention" / "data" / "raw"
WANT = {"MRUN", "PROM_GRAL", "ASISTENCIA", "SIT_FIN", "COD_ENSE", "COD_GRADO"}


def norm(col: str) -> str:
    # Files mix encodings (ASCII, UTF-8 with/without BOM, Latin-1) and header case.
    return col.replace("﻿", "").replace("ï»¿", "").strip().upper()


def profile(year: int) -> dict:
    csv = next((RAW / str(year)).glob("*.csv"))
    df = pd.read_csv(csv, sep=";", encoding="latin-1", dtype=str,
                     usecols=lambda c: norm(c) in WANT)
    df.columns = [norm(c) for c in df.columns]
    gpa = pd.to_numeric(df.PROM_GRAL.str.replace(",", ".", regex=False), errors="coerce")
    sit = df.SIT_FIN.fillna("").str.strip()
    keep = sit.isin(["P", "R"]) & (gpa > 0)
    g, retained = gpa[keep].round(1), sit[keep] == "R"

    def n(v):
        return int((g == v).sum())

    def r(v):
        m = g == v
        return float(retained[m].mean()) if m.any() else float("nan")

    return dict(
        year=year, rows=len(df), P=int((sit == "P").sum()), R=int((sit == "R").sum()),
        Y=int((sit == "Y").sum()),
        mrun_missing=float(pd.to_numeric(df.MRUN, errors="coerce").isna().mean()),
        offgrid=float((np.abs(gpa[keep] * 10 - np.round(gpa[keep] * 10)) > 1e-6).mean()),
        n44=n(4.4), n45=n(4.5), n46=n(4.6), n49=n(4.9), n50=n(5.0), n51=n(5.1),
        r44=r(4.4), r45=r(4.5), r49=r(4.9), r50=r(5.0),
    )


def main(out: str) -> None:
    years = sorted(int(p.name) for p in RAW.iterdir() if p.is_dir() and p.name.isdigit())
    t = pd.DataFrame([profile(y) for y in years]).set_index("year")
    t["heap45"] = t.n45 / np.sqrt(t.n44 * t.n46)
    t["heap50"] = t.n50 / np.sqrt(t.n49 * t.n51)
    t.to_csv(out)
    cols = ["rows", "P", "R", "Y", "n44", "n45", "n46", "heap45", "r44", "r45",
            "heap50", "r49", "r50"]
    print(t[cols].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
