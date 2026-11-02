"""Screen of the Freddie Mac sample files for the dose-schedule design (2026-11-02).

    python -m experiments.datasets.freddie_mac.screen

Two checks. (1) The fee grid shows up in rates: mean rate residual (rate minus its first-payment-
month x LTV-band mean) just above minus just below each credit-score cutoff of the grid, 30-year
fixed-rate owner-occupied one-unit purchases, against placebo cutoffs. (2) How well the loan
covariates predict the credit score (R^2), which bounds how far a fee can be transported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
# Origination file layout of the sample files (31 fields; no servicer name).
ORIG = ["fico", "first_pay", "fthb", "maturity", "msa", "mi_pct", "units", "occ", "cltv", "dti", "upb", "ltv", "rate",
        "channel", "ppm", "amort", "state", "ptype", "zip", "seq", "purpose", "term", "nborr", "seller", "superconf",
        "preharp", "program", "harp", "valuation", "io", "mi_cancel"]


def origination() -> pd.DataFrame:
    fs = sorted(RAW.glob("sample_orig_*.txt"))
    if not fs:
        raise FileNotFoundError(f"no sample_orig_*.txt in {RAW}; see README.md")
    df = pd.concat([pd.read_csv(f, sep="|", header=None, names=ORIG, low_memory=False) for f in fs], ignore_index=True)
    df["oy"] = df.first_pay // 100
    return df


def rate_jumps(df: pd.DataFrame, width: int = 5) -> pd.DataFrame:
    d = df[df.fico.between(300, 850) & df.ltv.between(1, 105) & (df.term == 360) & (df.amort == "FRM")
           & (df.occ == "P") & (df.purpose == "P") & (df.units == 1)].copy()
    d["ltvb"] = pd.cut(d.ltv, [0, 60, 70, 75, 80, 85, 90, 95, 97, 105], labels=False)
    d["res"] = d.rate - d.groupby(["first_pay", "ltvb"]).rate.transform("mean")
    rows = []
    for era, (lo, hi) in {"2009-22": (2009, 2022), "2023-26": (2023, 2026)}.items():
        for ltvb, lab in [(3, "75-80"), (6, "90-95")]:
            g = d[d.oy.between(lo, hi) & (d.ltvb == ltvb)]
            for c in range(620, 800, 10):
                jump = (g[g.fico.between(c, c + width - 1)].res.mean()
                        - g[g.fico.between(c - width, c - 1)].res.mean())
                rows.append(dict(era=era, ltv=lab, cutoff=c, grid=c % 20 == 0, jump_bp=100 * jump))
    return pd.DataFrame(rows)


def score_r2(df: pd.DataFrame) -> float:
    d = df[df.fico.between(300, 850) & df.ltv.between(1, 105) & df.dti.between(0, 65)]
    num = pd.DataFrame({"ltv": d.ltv, "ltv2": d.ltv ** 2 / 100, "dti": d.dti, "dti2": d.dti ** 2 / 100,
                        "lupb": np.log(d.upb), "lupb2": np.log(d.upb) ** 2, "nb2": (d.nborr == 2).astype(float),
                        "fthb": (d.fthb == "Y").astype(float), "mi": d.mi_pct.clip(0, 60),
                        "cltv_gap": (d.cltv - d.ltv).clip(0, 50)})
    cat = pd.get_dummies(d[["occ", "purpose", "channel", "ptype", "state", "term", "units", "first_pay"]].astype(str),
                         drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), num.to_numpy(float), cat.to_numpy(float)])
    y = d.fico.to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - (y - X @ b).var() / y.var())


def main():
    df = origination()
    print(f"{len(df):,} loans, origination years {df.oy.min()}-{df.oy.max()}")
    j = rate_jumps(df)
    for (era, ltv), g in j.groupby(["era", "ltv"]):
        grid = " ".join(f"{r.cutoff}:{r.jump_bp:+.1f}" for r in g[g.grid].itertuples())
        plac = " ".join(f"{r.cutoff}:{r.jump_bp:+.1f}" for r in g[~g.grid].itertuples())
        print(f"{era} LTV {ltv}: grid cutoffs {grid}\n    placebo {plac}")
    print(f"R^2 of the credit score on loan covariates: {score_r2(df):.3f}")


if __name__ == "__main__":
    main()
