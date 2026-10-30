"""Hospital eligibility for the 340B Drug Pricing Program: a group cutoff on the Medicare
disproportionate share (DSH) adjustment percentage, i.e. on the sum of two fractions.

42 U.S.C. 256b(a)(4) makes a hospital a 340B covered entity if it meets an ownership test
((L)(i): owned or operated by a state or local government, a public or private nonprofit
formally granted governmental powers, or a private nonprofit with a state or local
government contract to serve low-income patients) and a DSH test on its most recent cost
report:

* (L)(ii) subsection (d) (IPPS) hospitals: DSH adjustment percentage > 11.75 (or a "Pickle"
  hospital); (M) children's and free-standing cancer hospitals: the same test;
* (O) rural referral centers (RRC) and sole community hospitals (SCH): >= 8;
* (N) critical access hospitals: no DSH test.

HRSA's registration screens take the DSH adjustment percentage and the control type from
HCRIS: Worksheet E, Part A, line 33 ("allowable disproportionate share percentage") of the
most recently filed Medicare cost report (form CMS-2552-10); proprietary control types 3-6
are ineligible. RRC status is Worksheet S-2 line 116 and SCH status S-2 line 35.

The adjustment percentage is a piecewise-linear increasing function of the
disproportionate patient percentage (DPP, line 32 = line 30 + line 31):

    DPP = SSI fraction (line 30: Medicare SSI days / Medicare Part A days, from CMS)
        + Medicaid fraction (line 31: Medicaid, non-Medicare days / total days),
    adj = 0 (DPP < 15), 2.5 + 0.65 (DPP - 15) (15 <= DPP < 20.2),
          5.88 + 0.825 (DPP - 20.2) (DPP >= 20.2), capped at 12 for small urban and most
          rural hospitals (42 CFR 412.106(d); the cap is above both cutoffs).

Both cutoffs lie on the top segment, so in DPP units (percent):

    adj > 11.75  <=>  DPP > 20.2 + 5.87 / 0.825 = 27.3152 (DSH group);
    adj >= 8     <=>  DPP >= 20.2 + 2.12 / 0.825 = 22.7697 (RRC/SCH group).

The cost report stores the fractions and line 33 rounded to 4 decimals (0.01 point of
DPP); with that rounding the rule is DPP >= 27.33 and DPP >= 22.77 on the 0.01 grid, so
the adapter uses the midpoints CUT = 27.325 and 22.765 (``rule_check()``).

Panel: hospital (CMS certification number) x cost report, indexed by t = the federal
fiscal year in which the cost reporting period begins (the HCRIS file year), t = 2011-2022.
Sample: short-term acute IPPS hospitals in the IPPS Impact File for FY t (50 states and
DC; Maryland is not IPPS, Puerto Rico has no SSI), nonprofit (control 1-2) or state/local
government (8-13), not Pickle hospitals, with the two fractions on report t and Y.

  Q   ``load()``: DPP minus the group cutoff (percent); D = 1{Q >= 0} (eligible side
      above). ``load_two_score()``: (SSI %, Medicaid %), weights (1, 1).
  Y   (default) log outpatient charges in the "drugs charged to patients" cost center
      (Worksheet C Part I line 73 column 7) on cost report t+2. Other outcomes:
      ``outcome=`` "drug_ccr", "log_drug_cost", "log_op_revenue", "op_margin",
      "uc_share", "drug_share_op".
  X   Impact File FY t (published before period t begins): log beds, resident-to-bed
      ratio and a teaching flag, large-urban / other-urban flags, case-mix index, census
      region; HCRIS control type (church nonprofit, government subtypes); RRC/SCH; FY
      indicators. ``lagged_y=True`` adds Y on report t-1 (with a missing flag); it is
      off by default because it jumps at the cutoff as much as Y does.

First stage: HRSA's covered-entity list (OPAIS) has no plain-request download. CMS's
340B remedy file (Addendum AAA to CMS-1793-F) lists, by CCN, the 2018-2022 Medicare
underpayment on 340B-acquired drugs billed under OPPS; it is > 0 only for hospitals that
billed 340B-acquired drugs (rural SCHs, children's and cancer hospitals were exempt).

    from experiments.datasets.hospital_340b import adapter
    s = adapter.load()
    adapter.rule_check(); adapter.local_jump(s)
"""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.hospital_340b.download import RAW, hcris_zip
# Impact Files (IPPS hospitals, beds, teaching, urban, region, CMI): read through the
# medicare_dsh folder's reader and cache, not copied.
from experiments.datasets.medicare_dsh.adapter import build_panel as impact_panel

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "hcris_340b_panel.csv.gz"
PANEL_VERSION = 1
HCRIS_YEARS = range(2011, 2025)
SCORE_YEARS = (2011, 2022)

# DPP cutoffs (percent) implied by the statutory cutoffs on the adjustment percentage.
CUT_DSH_EXACT = 20.2 + (11.75 - 5.88) / 0.825      # 27.3152, adj > 11.75
CUT_RS_EXACT = 20.2 + (8.0 - 5.88) / 0.825         # 22.7697, adj >= 8
CUT_DSH, CUT_RS = 27.325, 22.765                   # midpoints of the 0.01 grid
GROUP_CUTOFFS = {"dsh": CUT_DSH, "rrc_sch": CUT_RS}
ADJ_CUTOFFS = {"dsh": 11.75, "rrc_sch": 8.0}

NONPROFIT = {1, 2}
GOVERNMENT = {8, 9, 10, 11, 12, 13}
CONTROL = {1: "nonprofit_church", 2: "nonprofit_other", 3: "proprietary_individual",
           4: "proprietary_corporation", 5: "proprietary_partnership", 6: "proprietary_other",
           7: "gov_federal", 8: "gov_city_county", 9: "gov_county", 10: "gov_state",
           11: "gov_hospital_district", 12: "gov_city", 13: "gov_other"}

# (worksheet, line, column) -> name, CMS-2552-10.
NMRC = {
    ("E00A18A", "03000", "00100"): "ssi",            # SSI fraction (from CMS)
    ("E00A18A", "03100", "00100"): "mcd",            # Medicaid fraction
    ("E00A18A", "03200", "00100"): "dpp_cr",         # sum of lines 30 and 31
    ("E00A18A", "03300", "00100"): "adj_cr",         # allowable DSH percentage
    ("E00A18A", "03300", "00101"): "adj_cr2",        # second column (split periods)
    ("S200001", "03500", "00100"): "sch_periods",    # periods SCH status in effect
    ("C000001", "07300", "00100"): "drug_cost",      # drugs charged to patients: cost
    ("C000001", "07300", "00600"): "drug_ip_chg",    #   inpatient charges
    ("C000001", "07300", "00700"): "drug_op_chg",    #   outpatient charges
    ("C000001", "07300", "00800"): "drug_chg",       #   total charges
    ("C000001", "20200", "00100"): "tot_cost",       # all cost centers (line 202)
    ("C000001", "20200", "00700"): "tot_op_chg",
    ("C000001", "20200", "00800"): "tot_chg",
    ("G200000", "02800", "00100"): "ip_rev",         # G-2 line 28: gross patient revenue
    ("G200000", "02800", "00200"): "op_rev",
    ("G300000", "00300", "00100"): "net_pat_rev",    # G-3 line 3
    ("G300000", "00400", "00100"): "opex",           # G-3 line 4
    ("G300000", "00500", "00100"): "net_inc_pat",    # G-3 line 5
    ("S100000", "03000", "00100"): "uc_cost",        # S-10 line 30: uncompensated care cost
    ("S100000", "03100", "00100"): "uc_unreimb",     # S-10 line 31
}
ALPHA = {("S200001", "11600", "00100"): "rrc_flag"}  # "Y" = referral center
RPT_COLS = {0: "rec", 1: "ctrl", 2: "ccn", 4: "status", 5: "fyb", 6: "fye", 7: "proc_dt"}

OUTCOMES = {
    "log_drug_op_chg": "log outpatient charges, drugs charged to patients (C-I line 73 col 7)",
    "drug_ccr": "drug cost / drug charges (C-I line 73 col 1 / col 8)",
    "log_drug_cost": "log cost of drugs charged to patients (C-I line 73 col 1)",
    "drug_share_op": "drug share of outpatient charges, percent (line 73 / line 202, col 7)",
    "log_op_revenue": "log gross outpatient patient revenue (G-2 line 28 col 2)",
    "op_margin": "operating margin, percent: G-3 line 5 / line 3",
    "uc_share": "uncompensated care cost (S-10 line 30), percent of operating expenses",
}
MAIN_OUTCOME = "log_drug_op_chg"


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.csv.gz")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, compression="gzip")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ------------------------------------------------------------------------- HCRIS

def read_hcris(year: int, force: bool = False) -> pd.DataFrame:
    """One HCRIS hospital file (CMS-2552-10): one row per report with the NMRC/ALPHA items
    above and the report header. Cached in data/processed/hcris_<year>.csv.gz."""
    cache = PROCESSED / f"hcris_{year}.csv.gz"
    if cache.exists() and not force:
        return pd.read_csv(cache, dtype={"ccn": str, "rrc_flag": str})
    z = hcris_zip(year)
    if not z.exists():
        raise FileNotFoundError(f"{z} missing. Run: python -m experiments.datasets.hospital_340b.download")
    with zipfile.ZipFile(z) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        rpt = pd.read_csv(zf.open(names[f"hosp10_{year}_rpt.csv"]), header=None, dtype=str)
        rpt = rpt[list(RPT_COLS)].rename(columns=RPT_COLS)
        rpt["rec"] = rpt.rec.astype(int)
        out = rpt.set_index("rec")
        for member, items, numeric in ((f"hosp10_{year}_nmrc.csv", NMRC, True),
                                       (f"hosp10_{year}_alpha.csv", ALPHA, False)):
            sheets = {k[0] for k in items}
            keys = {"|".join(k): v for k, v in items.items()}
            parts = []
            dtype = {"rec": int, "w": str, "l": str, "c": str, "v": float if numeric else str}
            for ch in pd.read_csv(zf.open(names[member]), header=None,
                                  names=["rec", "w", "l", "c", "v"], dtype=dtype,
                                  chunksize=4_000_000):
                ch = ch[ch.w.isin(sheets)]
                key = ch.w + "|" + ch.l + "|" + ch.c
                ch = ch.assign(item=key.map(keys)).dropna(subset=["item"])
                parts.append(ch[["rec", "item", "v"]])
            long = pd.concat(parts, ignore_index=True)
            wide = long.pivot_table(index="rec", columns="item", values="v", aggfunc="first")
            out = out.join(wide.reindex(columns=list(items.values())))
    out = out.reset_index()
    out["year"] = year
    out["ccn"] = out.ccn.astype(str).str.strip().str.zfill(6)
    _atomic_csv(out, cache)
    return out


def _reports(years=HCRIS_YEARS) -> pd.DataFrame:
    """One report per hospital and file year: the longest period, then the latest
    processed (short reports arise from changes of ownership and new providers)."""
    r = pd.concat([read_hcris(y) for y in years], ignore_index=True)
    r["fyb"] = pd.to_datetime(r.fyb, format="%m/%d/%Y", errors="coerce")
    r["fye"] = pd.to_datetime(r.fye, format="%m/%d/%Y", errors="coerce")
    r["proc_dt"] = pd.to_datetime(r.proc_dt, format="%m/%d/%Y", errors="coerce")
    r["days"] = (r.fye - r.fyb).dt.days + 1
    r["n_reports"] = r.groupby(["ccn", "year"]).rec.transform("size")
    r = r.sort_values(["ccn", "year", "days", "proc_dt"]).drop_duplicates(["ccn", "year"], keep="last")
    return r.reset_index(drop=True)


def _derive(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    pos = lambda s: s.where(s > 0)                                   # noqa: E731
    r["log_drug_op_chg"] = np.log(pos(r.drug_op_chg))
    r["drug_ccr"] = r.drug_cost / pos(r.drug_chg)
    r["drug_ccr"] = r.drug_ccr.where(r.drug_ccr.between(0, 5))
    r["log_drug_cost"] = np.log(pos(r.drug_cost))
    r["drug_share_op"] = 100 * r.drug_op_chg / pos(r.tot_op_chg)
    r["drug_share_op"] = r.drug_share_op.where(r.drug_share_op.between(0, 100))
    r["log_op_revenue"] = np.log(pos(r.op_rev))
    m = 100 * r.net_inc_pat / pos(r.net_pat_rev)
    r["op_margin"] = m.where(m.between(-100, 100))
    u = 100 * r.uc_cost / pos(r.opex)
    r["uc_share"] = u.where(u.between(0, 100))
    return r


# ------------------------------------------------------------------------- panel

def _formula(dpp):
    d = np.asarray(dpp, float)
    return np.where(d < 15, 0.0, np.where(d < 20.2, 2.5 + 0.65 * (d - 15), 5.88 + 0.825 * (d - 20.2)))


def build(force: bool = False) -> pd.DataFrame:
    """Report-level panel (all hospitals and control types, file years 2011-2024) with the
    scores, the rule, Impact File covariates for FY t, lags, leads and 340B remedy amounts."""
    if PANEL.exists() and not force:
        p = pd.read_csv(PANEL, dtype={"ccn": str, "rrc_flag": str, "urgeo": str})
        if int(p.panel_version.iloc[0]) == PANEL_VERSION:
            return p
    r = _derive(_reports())
    r["ssi_pct"] = 100 * r.ssi
    r["mcd_pct"] = 100 * r.mcd
    r["dpp"] = r.ssi_pct.fillna(0) + r.mcd_pct.fillna(0)
    r.loc[r.ssi.isna() & r.mcd.isna(), "dpp"] = np.nan
    r["dpp_line32"] = 100 * r.dpp_cr
    r["adj_line33"] = 100 * r.adj_cr.fillna(0.0)
    r["adj_formula"] = _formula(r.dpp)
    r["rrc"] = r.rrc_flag.astype(str).str.strip().str.upper().eq("Y").astype(int)
    r["sch"] = (r.sch_periods.fillna(0) > 0).astype(int)
    r["group"] = np.where((r.rrc == 1) | (r.sch == 1), "rrc_sch", "dsh")
    r["cutoff"] = r.group.map(GROUP_CUTOFFS)
    r["Q"] = r.dpp - r.cutoff
    r["D"] = (r.Q >= 0).astype(int)
    # The cost report's own allowable DSH % against the statutory cutoffs (HRSA prefill).
    r["eligible_line33"] = np.where(r.group == "rrc_sch", r.adj_cr.fillna(0) >= 0.08 - 1e-9,
                                    r.adj_cr.fillna(0) > 0.1175 + 1e-9).astype(int)
    r["pickle_cr"] = (np.abs(r.adj_cr - 0.35) < 1e-6).astype(int)
    r["ctrl"] = pd.to_numeric(r.ctrl, errors="coerce")

    # Lags and leads by CCN and file year.
    key = r.set_index(["ccn", "year"])
    for k, tag in ((-1, "_lag1"), (1, "_lead1"), (2, "_lead2")):
        idx = pd.MultiIndex.from_arrays([r.ccn, r.year + k])
        cols = list(OUTCOMES) + (["dpp", "D", "ctrl"] if k < 0 else ["D", "ctrl"])
        for c in cols:
            r[c + tag] = key[c].reindex(idx).to_numpy()

    # Impact File FY t (CMS, published with the IPPS final rule in August of t-1).
    imp = impact_panel()
    imp = imp.rename(columns={"provider": "ccn"})
    imp["ccn"] = imp.ccn.astype(str).str.zfill(6)
    keep = ["ccn", "year", "beds", "resbed", "cmi", "region", "urgeo", "urspa", "ptype",
            "pickle", "dpp"]
    imp = imp[keep].rename(columns={"dpp": "dpp_impact", "pickle": "pickle_impact",
                                    "ptype": "ptype_impact"})
    r = r.merge(imp, on=["ccn", "year"], how="left")
    r["in_impact"] = r.beds.notna().astype(int)

    # CMS 340B remedy (Addendum AAA): 340B-acquired drug underpayment by calendar year.
    rem = read_remedy()
    r = r.merge(rem, on="ccn", how="left")
    r["panel_version"] = PANEL_VERSION
    _atomic_csv(r, PANEL)
    return pd.read_csv(PANEL, dtype={"ccn": str, "rrc_flag": str, "urgeo": str})


def read_remedy() -> pd.DataFrame:
    """Addendum AAA (CMS-1793-F, final rule 2023-11): per CCN, '<CY> 340B Drugs Impact'
    for CY 2018-2022, the Medicare OPPS underpayment on drugs billed as 340B-acquired."""
    z = RAW / "cms_340b" / "addendum_aaa.zip"
    if not z.exists():
        raise FileNotFoundError(f"{z} missing. Run download.py")
    with zipfile.ZipFile(z) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".txt") and "Remedy Payment" in n)
        lines = zf.read(member).decode("latin-1").splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Provider Number"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep="\t", dtype=str)
    df = df[df["Provider Number"].fillna("").str.strip().str.fullmatch(r"[0-9A-Z]{6}")]
    out = pd.DataFrame({"ccn": df["Provider Number"].str.strip()})
    for y in range(2018, 2023):
        col = next(c for c in df.columns if c.startswith(f"{y} 340B Drugs Impact"))
        out[f"remedy_{y}"] = pd.to_numeric(df[col].str.replace(r"[$,\"*]", "", regex=True),
                                           errors="coerce").to_numpy()
    return out.drop_duplicates("ccn")


# ------------------------------------------------------------------------- samples

def _frame(outcome: str = MAIN_OUTCOME, lead: int = 2, years=SCORE_YEARS,
           groups=("dsh", "rrc_sch")) -> pd.DataFrame:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    p = build()
    st = p.ccn.str[:2]
    ycol = f"{outcome}_lead{lead}"
    df = p[p.year.between(*years)
           & p.ccn.str[2:].str.fullmatch(r"0[0-8]\d\d")             # short-term acute
           & ~st.isin(["21", "40"]) & (p.in_impact == 1)
           & p.ctrl.isin(NONPROFIT | GOVERNMENT)
           & (p.pickle_cr == 0) & (p.pickle_impact.fillna(0) == 0)
           & p.ssi.notna() & p.mcd.notna() & p.dpp.between(0, 100)
           & p.group.isin(groups) & p[ycol].notna()]
    df = df[df.beds > 0].copy()
    df["Y"] = df[ycol]
    return df.reset_index(drop=True)


def _design(df: pd.DataFrame, outcome: str, lagged_y: bool = False) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    X["log_beds"] = np.log(df.beds)
    X["resbed"] = df.resbed.fillna(0.0)
    X["teaching"] = (X.resbed > 0).astype(float)
    cmi = df.cmi
    X["cmi"] = cmi.fillna(cmi.median())
    X["cmi_missing"] = cmi.isna().astype(float)
    ug = df.urgeo.fillna("").astype(str).str.upper()
    X["large_urban"] = ug.eq("LURBAN").astype(float)
    X["other_urban"] = ug.eq("OURBAN").astype(float)
    X["rrc_sch"] = (df.group == "rrc_sch").astype(float)
    for c in sorted(set(df.ctrl.astype(int)) - {2}):                 # base: nonprofit, other
        X[f"ctrl_{CONTROL[c]}"] = (df.ctrl == c).astype(float)
    if lagged_y:      # report t-1: itself shifted by (persistent) eligibility, see README
        lag = df[f"{outcome}_lag1"]
        X["y_lag1"] = lag.fillna(lag.median())
        X["y_lag1_missing"] = lag.isna().astype(float)
    reg = df.region.fillna(0).astype(int)
    for g in sorted(reg.unique())[1:]:
        X[f"region_{g}"] = (reg == g).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"fy_{y}"] = (df.year == y).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _extras(df: pd.DataFrame, lead: int) -> dict:
    cy = df.year + lead          # calendar year mostly covered by report t + lead
    rem = np.full(len(df), np.nan)
    for y in range(2018, 2023):
        m = (cy == y).to_numpy()
        rem[m] = df[f"remedy_{y}"].to_numpy(float)[m]
    return {
        "unit_id": df.ccn.to_numpy(str), "year": df.year.to_numpy(int),
        "group": df.group.to_numpy(str), "group_cutoffs": dict(GROUP_CUTOFFS),
        "adj_cutoffs": dict(ADJ_CUTOFFS),
        "cutoff_dpp": df.cutoff.to_numpy(float), "dpp": df.dpp.to_numpy(float),
        "ssi_pct": df.ssi_pct.to_numpy(float), "mcd_pct": df.mcd_pct.to_numpy(float),
        "adj_line33": df.adj_line33.to_numpy(float), "adj_formula": df.adj_formula.to_numpy(float),
        # HRSA pre-fills its registration screen with line 33 of the latest cost report:
        # the published value the rule is checked against (OPAIS enrollment is not
        # downloadable; see README).
        "eligible_published": df.eligible_line33.to_numpy(float),
        "control_type": df.ctrl.to_numpy(int), "rrc": df.rrc.to_numpy(int),
        "sch": df.sch.to_numpy(int), "beds": df.beds.to_numpy(float),
        "dpp_lag1": df.dpp_lag1.to_numpy(float), "D_lag1": df.D_lag1.to_numpy(float),
        "dpp_impact": df.dpp_impact.to_numpy(float),
        "remedy_cy": rem, "remedy_year": np.where((cy >= 2018) & (cy <= 2022), cy, -1),
        "fyb": df.fyb.astype(str).to_numpy(), "fye": df.fye.astype(str).to_numpy(),
        "lead": lead, "y_lag1": df[[c for c in df.columns if c.endswith("_lag1")
                                    and c[:-5] in OUTCOMES]].to_numpy(float),
        "y_lag1_names": [c for c in df.columns if c.endswith("_lag1") and c[:-5] in OUTCOMES],
    }


def load(outcome: str = MAIN_OUTCOME, lead: int = 2, years=SCORE_YEARS,
         lagged_y: bool = False) -> RDDSample:
    """Q = DPP - group cutoff (27.325 DSH; 22.765 RRC/SCH), percent; D = 1{Q >= 0}
    (340B-eligible above); Y = outcome on cost report t + lead (default log outpatient
    drug charges two reports later)."""
    df = _frame(outcome, lead, years)
    X, names = _design(df, outcome, lagged_y)
    return RDDSample(
        Q=df.Q.to_numpy(float), X=X, Y=df.Y.to_numpy(float), threshold=0.0,
        name="hospital_340b" if (outcome == MAIN_OUTCOME and lead == 2) else f"hospital_340b_{outcome}_t{lead}",
        feature_names=names,
        description=("340B eligibility of nonprofit and public IPPS hospitals, cost reports "
                     f"FY {years[0]}-{years[1]}. Q = DSH patient percentage (SSI + Medicaid "
                     "fraction, percent) minus the group cutoff (27.325 = adjustment "
                     "> 11.75; 22.765 for RRC/SCH = adjustment >= 8); D = 1{Q >= 0}; "
                     f"Y = {OUTCOMES[outcome]}, cost report t+{lead}."),
        citation=("42 U.S.C. 256b(a)(4)(L)-(O); HRSA 340B OPAIS registration help; CMS HCRIS "
                  "hospital cost reports (CMS-2552-10), FY 2011-2024; CMS IPPS Impact Files; "
                  "Desai & McWilliams (2018), NEJM, doi:10.1056/NEJMsa1706475"),
        treatment_rule=lambda q: (np.asarray(q, float) >= 0).astype(int),
        extras={**_extras(df, lead), "outcome": outcome, "cutoff": 0.0},
    )


def load_two_score(group: str = "dsh", outcome: str = MAIN_OUTCOME, lead: int = 2,
                   years=SCORE_YEARS, lagged_y: bool = False) -> RDDSample:
    """Q = (SSI fraction, Medicaid fraction), percent, for one cutoff group; D =
    1{Q1 + Q2 >= c} with c = 27.325 ("dsh": DSH adjustment > 11.75) or 22.765
    ("rrc_sch": >= 8). Same X and Y as ``load()``."""
    cut = GROUP_CUTOFFS[group]
    df = _frame(outcome, lead, years, groups=(group,))
    X, names = _design(df, outcome, lagged_y)
    w = np.array([1.0, 1.0])
    return RDDSample(
        Q=df[["ssi_pct", "mcd_pct"]].to_numpy(float), X=X, Y=df.Y.to_numpy(float),
        threshold=(cut, cut), name=f"hospital_340b_two_score_{group}", feature_names=names,
        description=(f"340B eligibility ({group} group), nonprofit and public IPPS hospitals, "
                     f"cost reports FY {years[0]}-{years[1]}. Q = (SSI fraction, Medicaid "
                     f"fraction) in percent; treatment = 1{{Q1 + Q2 >= {cut}}}; Y = "
                     f"{OUTCOMES[outcome]}, cost report t+{lead}."),
        citation="42 U.S.C. 256b(a)(4); CMS HCRIS (CMS-2552-10 Worksheet E Part A lines 30-33)",
        treatment_rule=lambda Q: (np.asarray(Q, float) @ w >= cut).astype(int),
        extras={**_extras(df, lead), "weights": (1.0, 1.0), "cutoff": cut, "outcome": outcome,
                "cutoff_exact": CUT_DSH_EXACT if group == "dsh" else CUT_RS_EXACT},
    )


def load_rrc_sch(**kw) -> RDDSample:
    return load_two_score("rrc_sch", **kw)


# ------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """By file year, on all short-term acute reports with Worksheet E Part A (50 states and
    DC, every control type): line 32 = line 30 + line 31; line 33 = the formula on line 32
    (to 0.0001 of a fraction, cap 12 allowed); and eligibility from Q (DPP vs the group
    cutoff) against eligibility from line 33 (the value HRSA pre-fills)."""
    p = build()
    p = p[p.ccn.str[2:].str.fullmatch(r"0[0-8]\d\d") & ~p.ccn.str[:2].isin(["21", "40"])
          & p.ssi.notna() & p.mcd.notna() & (p.pickle_cr == 0)]
    rows = []
    for y, g in p.groupby("year"):
        f = np.round(g.adj_formula / 100, 4)
        f_ok = (np.abs(f - g.adj_cr.fillna(0)) < 1.5e-4) | (np.abs(g.adj_cr - 0.12) < 1e-9) & (f >= 0.12)
        e = g.eligible_line33 == g.D
        rows.append({
            "fy": y, "reports": len(g), "sum_ok": float((np.abs(g.dpp - g.dpp_line32) < 0.011).mean()),
            "line33_formula_ok": float(f_ok.mean()), "line33_capped": int((np.abs(g.adj_cr - 0.12) < 1e-9).sum()),
            "rule_agree": float(e.mean()), "disagree": int((~e).sum()),
            "eligible_share_np_gov": float(g.D[g.ctrl.isin(NONPROFIT | GOVERNMENT)].mean()),
            "rrc_sch": int((g.group == "rrc_sch").sum()),
        })
    return pd.DataFrame(rows)


def rule_exceptions() -> pd.DataFrame:
    p = build()
    p = p[p.ccn.str[2:].str.fullmatch(r"0[0-8]\d\d") & ~p.ccn.str[:2].isin(["21", "40"])
          & p.ssi.notna() & p.mcd.notna() & (p.pickle_cr == 0)]
    bad = p[p.eligible_line33 != p.D]
    return bad[["ccn", "year", "group", "ssi_pct", "mcd_pct", "dpp", "dpp_line32",
                "adj_line33", "adj_formula", "eligible_line33", "D"]]


def group_check() -> pd.DataFrame:
    """RRC/SCH from the cost report (S-2 lines 116 and 35) against the Impact File provider
    type for FY t (7, 15, 17, 22 = RRC; 16, 17, 21, 22 = SCH or EACH)."""
    p = build()
    p = p[p.in_impact == 1]
    pt = p.ptype_impact
    imp_rs = pt.isin([7, 15, 16, 17, 21, 22]).astype(int)
    cr_rs = (p.group == "rrc_sch").astype(int)
    return pd.crosstab([p.year], [cr_rs.rename("cost_report_rrc_sch"), imp_rs.rename("impact_rrc_sch")])


def local_jump(s: RDDSample | None = None, h: float = 5.0, y: np.ndarray | None = None) -> dict:
    """Local-linear jump in Y at Q = 0 (triangular kernel, bandwidth h in DPP points,
    separate slopes), standard error clustered by hospital."""
    s = s or load()
    q = s.Q if s.Q.ndim == 1 else s.Q.sum(1) - s.extras["cutoff"]
    yv = s.Y if y is None else np.asarray(y, float)
    m = (np.abs(q) < h) & np.isfinite(yv)
    q, yv, cl = q[m], yv[m], np.asarray(s.extras["unit_id"])[m]
    d = (q >= 0).astype(float)
    w = 1 - np.abs(q) / h
    Z = np.column_stack([np.ones_like(q), d, q, d * q])
    Wz = Z * w[:, None]
    A = np.linalg.inv(Z.T @ Wz)
    b = A @ (Wz.T @ yv)
    e = yv - Z @ b
    g = pd.Series(range(len(cl))).groupby(cl).indices
    meat = np.zeros((4, 4))
    for idx in g.values():
        sc = (Wz[idx] * e[idx, None]).sum(0)
        meat += np.outer(sc, sc)
    G = len(g)
    V = A @ meat @ A * G / max(G - 1, 1)
    return {"jump": float(b[1]), "se": float(np.sqrt(V[1, 1])), "n": int(m.sum()),
            "units": G, "h": h, "below_mean": float(b[0])}


def readiness(s: RDDSample | None = None) -> dict:
    s = s or load()
    Xt = np.column_stack([np.ones(s.n), s.X])
    out = {"n": s.n, "units": len(np.unique(s.extras["unit_id"])), "p": s.p,
           "treated": int(s.D.sum())}
    Qm = s.Q[:, None] if s.Q.ndim == 1 else s.Q
    for j in range(Qm.shape[1]):
        T = Xt @ np.linalg.lstsq(Xt, Qm[:, j], rcond=None)[0]
        out[f"R2_Q{j + 1}"] = float(1 - np.var(Qm[:, j] - T) / np.var(Qm[:, j]))
    T = Xt @ np.linalg.lstsq(Xt, s.Y, rcond=None)[0]
    out["R2_Y"] = float(1 - np.var(s.Y - T) / np.var(s.Y))
    q = s.Q if s.Q.ndim == 1 else s.Q.sum(1) - s.extras["cutoff"]
    out["bunching_0.5"] = (int(((q >= 0) & (q < 0.5)).sum()), int(((q >= -0.5) & (q < 0)).sum()))
    return out
