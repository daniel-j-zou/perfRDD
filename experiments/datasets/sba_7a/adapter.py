"""SBA 7(a) loans at the $150,000 guarantee notch (FOIA loan-level files, data.sba.gov).

Q = gross approval amount in dollars (``GrossApproval``). The SBA guarantees 85 percent of a
standard 7(a) loan of at most $150,000 and 75 percent of a larger loan, so the treatment
(the higher guarantee) applies AT OR BELOW the line: ``treatment_rule = Q <= 150000``.
Y = 1 if the loan was charged off (``LoanStatus == "CHGOFF"``), 0 if paid in full
(``"P I F"``); loans still outstanding (``EXEMPT``), cancelled (``CANCLD``) or undisbursed
(``COMMIT``) are excluded. X = covariates that explain the loan amount: term, jobs
supported, NAICS sector, business type and age, Census division, processing method,
revolver, collateral and franchise flags, approval fiscal year and lender size (number of
7(a) loans by the lender in the whole file); the initial interest rate is in the file only
from FY2009 and is an optional covariate (``include_rate=True``).

The notch is a program parameter that applies to the "standard" 7(a) products only: SBA
Express loans carry a flat 50 percent guarantee at any size, export and EWCP products 90
percent, so those processing methods are dropped (``EXCLUDED_METHODS``). It also switches
off in three stimulus windows when both sides were raised to 90 percent (ARRA from
2009-03 to 2010-05, the Small Business Jobs Act from 2010-10 to 2010-12, the Economic Aid
Act from 2021-01 to 2021-09; windows verified in the data, see README.md), and the
$150,000 / 85 percent schedule only starts in January 2001 (FY2001 Q2); earlier years used
80 percent at or below $100,000 and older schedules. ``NOTCH_WINDOWS`` lists the kept
approval-date ranges.

Expected effect: Bachas, Kim and Yannelis (JFE 2021) document bunching just below $150,000
and charge-off rates that trend smoothly through the line, so the effect of the guarantee on
default at the line is expected to be null; the score is manipulated near the line.

Files: the four FOIA 7(a) csv files under ``data/raw/`` (see download.py). ``build()`` reads
them once (about 2.0 million rows) and caches ``data/processed/sba_7a_all.parquet``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "sba_7a_all.parquet"
THRESHOLD = 150000.0

RAW_FILES = [
    "FOIA_7a_FY1991_FY1999_asof_260930.csv",
    "FOIA_7a_FY2000_FY2009_asof_260930.csv",
    "FOIA_7a_FY2010_FY2019_asof_260930.csv",
    "FOIA_7a_FY2020_Present_asof_260930.csv",
]
USECOLS = [
    "LocationID", "BankFDICNumber", "GrossApproval", "SBAGuaranteedApproval", "ApprovalDate",
    "ApprovalFY", "ProcessingMethod", "InitialInterestRate", "FixedorVariableInterestInd",
    "TermInMonths", "NaicsCode", "FranchiseCode", "ProjectState", "BorrState", "BusinessType",
    "BusinessAge", "LoanStatus", "ChargeOffDate", "GrossChargeOffAmount", "RevolverStatus",
    "JobsSupported", "CollateralInd", "SoldSecMrktInd",
]
NUMERIC = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFY", "InitialInterestRate",
           "TermInMonths", "JobsSupported", "GrossChargeOffAmount"]

# Processing methods whose guarantee does not follow the 85/75 notch (flat 50 or 90 percent).
EXCLUDED_METHODS = {
    "SBA Express Program", "SBA Express Bridge Loan", "Export Express", "7a with EWCP",
    "Preferred Lenders with EWCP", "Certified Lenders Program with EWCP",
    "Community Advantage International Trade", "International Trade Loans",
    "Export Import Harmonization  (EXIM)", "Dealer Floor Plan Loans",
    "Domestic Revolving Line of Credit",
}
# Approval-date windows (inclusive) in which the 85/75 notch at $150,000 is in force.
NOTCH_WINDOWS = [("2001-01-01", "2009-02-28"), ("2010-06-01", "2010-09-30"),
                 ("2011-01-01", "2020-12-31"), ("2021-10-01", "2099-12-31")]

METHOD_GROUP = {
    "7a General": "general", "Preferred Lenders Program": "plp",
    "Certified Lenders Program": "clp", "Low Documentation Program": "lowdoc",
    "Community Express": "community", "Community Advantage Initiative": "community",
    "Community Advantage Recovery Loan": "community", "Community Advantage RLOC": "community",
    "Small Loan Advantage Initiative": "initiative", "Patriot Express Loans": "initiative",
    "Rural Loan Initiative": "initiative", "Gulf Opportunity Pilot Loan Program": "initiative",
    "PLP MARC": "plp", "MARC 7a General": "general", "7(a) WCP": "capline",
    "Preferred Lenders with WCP": "capline", "Working Capital CAPLine": "capline",
    "Contract CAPLine": "capline", "Builders CAPLine": "capline", "Seasonal CAPLine": "capline",
    "Small Asset Base Line of Credit (CAPLine)": "capline",
}
METHOD_INDICATORS = ["plp", "clp", "lowdoc", "community", "initiative", "capline"]  # base: general

NAICS_SECTORS = ["11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51", "52", "53",
                 "54", "55", "56", "61", "62", "71", "72", "81", "92"]
NAICS_BASE = "72"  # accommodation and food services, the largest sector in 7(a)

DIVISION = {}
for _div, _states in {
    "new_england": "CT ME MA NH RI VT", "mid_atlantic": "NJ NY PA",
    "east_north_central": "IL IN MI OH WI", "west_north_central": "IA KS MN MO NE ND SD",
    "south_atlantic": "DE DC FL GA MD NC SC VA WV", "east_south_central": "AL KY MS TN",
    "west_south_central": "AR LA OK TX", "mountain": "AZ CO ID MT NV NM UT WY",
    "pacific": "AK CA HI OR WA", "territory": "PR VI GU AS MP",
}.items():
    for _s in _states.split():
        DIVISION[_s] = _div
DIVISION_INDICATORS = ["new_england", "mid_atlantic", "east_north_central",
                       "west_north_central", "south_atlantic", "east_south_central",
                       "west_south_central", "mountain", "territory"]  # base: pacific

NEW_BUSINESS = {"New, Less than 1 Year old", "Startup, Loan Funds will Open Business",
                "New Business or 2 years or less"}

X_COLS = (["term_months", "log_jobs", "jobs_supported", "approval_fy", "individual",
           "partnership", "new_business", "change_of_ownership", "age_unanswered",
           "revolver", "collateral", "franchise", "log_lender_loans"]
          + [f"method_{m}" for m in METHOD_INDICATORS]
          + [f"naics_{s}" for s in NAICS_SECTORS if s != NAICS_BASE]
          + [f"div_{d}" for d in DIVISION_INDICATORS])


def _require() -> None:
    missing = [f for f in RAW_FILES if not (RAW / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} missing under {RAW}. Run "
            "`python -m experiments.datasets.sba_7a.download` (see README.md).")


def build(force: bool = False) -> pd.DataFrame:
    """Read the four FOIA csv files into one typed frame of all 7(a) loans (cached)."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    _require()
    frames = []
    for f in RAW_FILES:
        d = pd.read_csv(RAW / f, usecols=USECOLS, dtype=str, encoding="latin-1")
        d["source_file"] = f
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ApprovalDate"] = pd.to_datetime(df["ApprovalDate"], errors="coerce")
    # lender size: number of 7(a) loans by SBA lender ID in the whole file (all programs)
    df["lender_loans"] = df.groupby("LocationID")["LocationID"].transform("size").astype(float)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def _naics_sector(code: pd.Series) -> pd.Series:
    two = code.astype(str).str.strip().str[:2]
    two = two.where(two.str.match(r"^\d\d$"), None)
    return (two.replace({"31": "31-33", "32": "31-33", "33": "31-33", "44": "44-45",
                         "45": "44-45", "48": "48-49", "49": "48-49"})
               .where(lambda s: s.isin(NAICS_SECTORS), None))


def design_frame(fiscal_years: Tuple[int, int] = (2001, 2016)) -> pd.DataFrame:
    """Standard-program 7(a) loans approved inside the notch windows and the fiscal-year
    range (resolved and unresolved alike), with the covariate columns built."""
    df = build()
    keep = (~df.ProcessingMethod.isin(EXCLUDED_METHODS)) & df.ApprovalDate.notna()
    win = np.zeros(len(df), dtype=bool)
    for a, b in NOTCH_WINDOWS:
        win |= (df.ApprovalDate >= a) & (df.ApprovalDate <= b)
    keep &= win & df.ApprovalFY.between(*fiscal_years)
    d = df.loc[keep].copy()
    d["guarantee_share"] = d.SBAGuaranteedApproval / d.GrossApproval
    d["resolved"] = d.LoanStatus.isin(["P I F", "CHGOFF"])
    d["charge_off"] = (d.LoanStatus == "CHGOFF").astype(float)
    d["term_months"] = d.TermInMonths
    d["jobs_supported"] = d.JobsSupported
    d["log_jobs"] = np.log1p(d.JobsSupported)
    d["approval_fy"] = d.ApprovalFY
    d["individual"] = (d.BusinessType == "INDIVIDUAL").astype(float)
    d["partnership"] = (d.BusinessType == "PARTNERSHIP").astype(float)
    d["new_business"] = d.BusinessAge.isin(NEW_BUSINESS).astype(float)
    d["change_of_ownership"] = (d.BusinessAge == "Change of Ownership").astype(float)
    d["age_unanswered"] = (d.BusinessAge.isna() | (d.BusinessAge == "Unanswered")).astype(float)
    d["revolver"] = (d.RevolverStatus == "Y").astype(float)
    d["collateral"] = (d.CollateralInd == "Y").astype(float)
    d["franchise"] = d.FranchiseCode.notna().astype(float)
    d["log_lender_loans"] = np.log(d.lender_loans)
    d["method_group"] = d.ProcessingMethod.map(METHOD_GROUP)
    for m in METHOD_INDICATORS:
        d[f"method_{m}"] = (d.method_group == m).astype(float)
    d["naics_sector"] = _naics_sector(d.NaicsCode)
    for s in NAICS_SECTORS:
        if s != NAICS_BASE:
            d[f"naics_{s}"] = (d.naics_sector == s).astype(float)
    d["division"] = d.ProjectState.str.strip().map(DIVISION)
    for v in DIVISION_INDICATORS:
        d[f"div_{v}"] = (d.division == v).astype(float)
    d["interest_rate"] = d.InitialInterestRate
    return d


def sample_frame(fiscal_years: Tuple[int, int] = (2001, 2016),
                 include_rate: bool = False) -> pd.DataFrame:
    """Resolved loans (paid in full or charged off) with complete Q, Y and X."""
    d = design_frame(fiscal_years)
    d = d[d.resolved & d.GrossApproval.gt(0) & d.BusinessType.notna()
          & d.naics_sector.notna() & d.division.notna() & d.method_group.notna()
          & d.CollateralInd.notna() & d.TermInMonths.gt(0) & d.JobsSupported.ge(0)
          & d.lender_loans.notna()]
    cols = list(X_COLS) + (["interest_rate"] if include_rate else [])
    d = d.dropna(subset=cols + ["GrossApproval", "charge_off"])
    return d.copy()


def design_matrix(d: pd.DataFrame, include_rate: bool = False):
    """(X, names, dropped): the X_COLS block (plus the rate when asked) without the
    columns that are constant in ``d`` (e.g. ``change_of_ownership`` exists only from
    FY2018, and some processing methods only in some years)."""
    cols = list(X_COLS) + (["interest_rate"] if include_rate else [])
    X = d[cols].to_numpy(float)
    keep = X.std(axis=0) > 0
    dropped = [c for c, k in zip(cols, keep) if not k]
    return X[:, keep], [c for c, k in zip(cols, keep) if k], dropped


def load(fiscal_years: Tuple[int, int] = (2001, 2016),
         include_rate: bool = False) -> RDDSample:
    d = sample_frame(fiscal_years, include_rate)
    X, cols, dropped = design_matrix(d, include_rate)
    return RDDSample(
        Q=d.GrossApproval.to_numpy(float), X=X,
        Y=d.charge_off.to_numpy(float), threshold=THRESHOLD, name="sba_7a",
        feature_names=cols,
        description=("SBA 7(a) FOIA loans, standard programs, approved in the notch windows "
                     f"FY{fiscal_years[0]}-{fiscal_years[1]}. Q = gross approval ($); "
                     "treatment = 1{Q <= 150000} (85% guarantee at or below the line, 75% "
                     "above); Y = charged off (vs paid in full)."),
        citation=("U.S. Small Business Administration, 7(a) & 504 FOIA, data.sba.gov "
                  "(public domain); Bachas, Kim and Yannelis (2021), Loan guarantees and "
                  "credit supply, Journal of Financial Economics 139(3), 872-894."),
        treatment_rule=lambda Q: (np.asarray(Q) <= THRESHOLD).astype(int),
        extras={"dropped_constant_columns": dropped,
                "guarantee_share": d.guarantee_share.to_numpy(float),
                "guaranteed_amount": d.SBAGuaranteedApproval.to_numpy(float),
                "charge_off_amount": d.GrossChargeOffAmount.to_numpy(float),
                "interest_rate": d.interest_rate.to_numpy(float),
                "approval_date": d.ApprovalDate.to_numpy(),
                "state": d.ProjectState.to_numpy(), "lender_id": d.LocationID.to_numpy(),
                "processing_method": d.ProcessingMethod.to_numpy(),
                "naics_code": d.NaicsCode.to_numpy()},
    )


def mirrored(fiscal_years: Tuple[int, int] = (2001, 2016), include_rate: bool = False):
    """(Q, X, Y, phi0, frame) with the score mirrored so that treatment is Q >= phi0:
    Q = -GrossApproval, phi0 = -150000 (treated iff amount <= 150000)."""
    d = sample_frame(fiscal_years, include_rate)
    X, _, _ = design_matrix(d, include_rate)
    return -d.GrossApproval.to_numpy(float), X, d.charge_off.to_numpy(float), -THRESHOLD, d
