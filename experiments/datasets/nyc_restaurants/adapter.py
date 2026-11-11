"""NYC restaurant inspections: the A-grade line at 13 points (NYC Open Data 43nn-pn8j).

Unit: a restaurant x cycle initial inspection. Q = the inspection score (violation points)
at a "Cycle Inspection / Initial Inspection"; the grade card is A for 0-13 points, so the
threshold is 13.5 and treatment is 1{Q <= 13.5} (received an A on the spot). A restaurant
scoring 14 or more is not graded at the initial inspection; it is re-inspected, and the
posted card (A, B or C, or "grade pending") comes from the re-inspection. The initial
inspection is the graded one here because its score alone decides whether the restaurant
gets an A immediately or enters the re-inspection cycle (see README.md).

Y, from the same file, measured after the initial inspection:
  * ``outcome="next_score_14"`` (``load``): the restaurant's next cycle inspection at least
    60 days later (initial or re-inspection; ``next_kind="initial"`` restricts to the next
    cycle's initial inspection) scores 14 or more, i.e. it loses or fails to get an A;
  * ``outcome="no_inspection_18m"`` (``load_exit``): no inspection of any kind within 18
    months (548 days) after the initial inspection. The file holds only restaurants that
    are active on the record date, so closures are NOT in it; this outcome is a gap in the
    inspection record of a survivor, not an exit (README.md).

X = cuisine indicators (the 15 most common cuisines plus "other"), borough indicators,
the number of prior inspections in the file, the prior inspection's score, the year, and
(``with_critical=True``) the count of critical violations cited at the inspection. The
critical count is a component of the score, not separate from it; it is off by default.

Rows with any missing field are dropped: no score, no prior scored inspection in the
file (the rolling three-year window cuts the first inspection of every restaurant), no
later inspection to score the outcome, or a window too short to observe it.

``build()`` collapses the violation-level CSV to one row per inspection and caches
``data/processed/inspections.parquet``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CSV = RAW / "current.csv"
CACHE = PROCESSED / "inspections.parquet"
THRESHOLD = 13.5
GAP_DAYS = 60          # a later inspection counts as "next graded" only after this many days
EXIT_DAYS = 548        # 18 months
CYCLE_INITIAL = "Cycle Inspection / Initial Inspection"
CYCLE_RE = "Cycle Inspection / Re-inspection"
GRADED_TYPES = (CYCLE_INITIAL, CYCLE_RE)
BOROS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
TOP_CUISINES = 15
NOT_INSPECTED = pd.Timestamp("1900-01-01")

X_BASE = ["n_prior", "prior_score", "year"]


def _require() -> None:
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} missing. Run python -m experiments.datasets.nyc_restaurants.download "
            "(see README.md).")


def build(force: bool = False) -> pd.DataFrame:
    """One row per (restaurant, inspection date, inspection type), cached."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    _require()
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["inspection_date"] = pd.to_datetime(df.inspection_date, format="%m/%d/%Y", errors="coerce")
    df["record_date"] = pd.to_datetime(df.record_date, format="%m/%d/%Y", errors="coerce")
    df["score"] = pd.to_numeric(df.score, errors="coerce")
    df = df[df.inspection_date.notna() & (df.inspection_date > NOT_INSPECTED)]
    key = ["camis", "inspection_date", "inspection_type"]
    first = df.groupby(key).agg(
        boro=("boro", "first"), cuisine=("cuisine_description", "first"),
        zipcode=("zipcode", "first"), score=("score", "max"), grade=("grade", "first"),
        action=("action", "first"), record_date=("record_date", "max"),
        n_viol=("violation_code", lambda s: int((s != "").sum())),
        n_critical=("critical_flag", lambda s: int((s == "Critical").sum())),
        n_not_critical=("critical_flag", lambda s: int((s == "Not Critical").sum())),
    ).reset_index()
    first["closed"] = first.action.str.contains("Closed by DOHMH", case=False).astype(int)
    first = first.sort_values(["camis", "inspection_date", "inspection_type"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    first.to_parquet(CACHE, index=False)
    return first


def _attach_history(insp: pd.DataFrame) -> pd.DataFrame:
    """Prior-inspection count and prior score for every inspection (any type): the count
    of earlier rows of the restaurant in the file, and the score of its latest scored
    inspection on a strictly earlier date."""
    insp = insp.sort_values(["camis", "inspection_date", "inspection_type"]).reset_index(drop=True)
    insp["n_prior"] = insp.groupby("camis").cumcount()
    scored = (insp.loc[insp.score.notna(), ["camis", "inspection_date", "score"]]
              .rename(columns={"inspection_date": "_d", "score": "prior_score"})
              .sort_values("_d"))
    insp["_i"] = np.arange(len(insp))
    m = pd.merge_asof(insp.sort_values("inspection_date"), scored, by="camis",
                      left_on="inspection_date", right_on="_d", direction="backward",
                      allow_exact_matches=False)
    return m.sort_values("_i").drop(columns=["_i", "_d"]).reset_index(drop=True)


def _next_inspection(base: pd.DataFrame, pool: pd.DataFrame, min_days: int,
                     cols: tuple = ("inspection_date", "score", "inspection_type")) -> pd.DataFrame:
    """For each row of ``base``, the first row of ``pool`` (same restaurant) dated at least
    ``min_days`` later. Returns the pool columns prefixed ``next_``."""
    pool = pool.sort_values("inspection_date")[["camis", *cols]].rename(
        columns={c: f"next_{c}" for c in cols})
    b = base[["camis", "inspection_date"]].copy()
    b["_cut"] = b.inspection_date + pd.Timedelta(days=min_days)
    b["_i"] = np.arange(len(b))
    m = pd.merge_asof(b.sort_values("_cut"), pool, by="camis", left_on="_cut",
                      right_on="next_inspection_date", direction="forward")
    return m.sort_values("_i").drop(columns=["_cut", "_i", "camis", "inspection_date"]).reset_index(drop=True)


def sample_frame(outcome: str = "next_score_14", next_kind: str = "any",
                 with_critical: bool = False, min_year: int = 2010) -> pd.DataFrame:
    """Cycle initial inspections with Q, X and the requested Y; rows with a missing field
    dropped. ``next_kind`` is "any" (next cycle initial or re-inspection) or "initial"."""
    insp = _attach_history(build())
    base = insp[(insp.inspection_type == CYCLE_INITIAL) & insp.score.notna()
                & (insp.inspection_date.dt.year >= min_year)].copy()
    record = insp.record_date.max()
    if outcome == "next_score_14":
        types = GRADED_TYPES if next_kind == "any" else (CYCLE_INITIAL,)
        pool = insp[insp.inspection_type.isin(types) & insp.score.notna()]
        nxt = _next_inspection(base, pool, GAP_DAYS)
        base = pd.concat([base.reset_index(drop=True), nxt], axis=1)
        base["next_days"] = (base.next_inspection_date - base.inspection_date).dt.days
        base["Y"] = (base.next_score >= 14).astype(float).where(base.next_score.notna())
    elif outcome == "no_inspection_18m":
        nxt = _next_inspection(base, insp, 1)
        base = pd.concat([base.reset_index(drop=True), nxt], axis=1)
        base["next_days"] = (base.next_inspection_date - base.inspection_date).dt.days
        # the window must be long enough to see 18 months of follow-up
        base = base[base.inspection_date + pd.Timedelta(days=EXIT_DAYS) <= record]
        base["Y"] = (base.next_days.isna() | (base.next_days > EXIT_DAYS)).astype(float)
    else:
        raise ValueError(f"unknown outcome {outcome!r}")

    base = base[base.boro.isin(BOROS) & (base.cuisine != "")]
    top = (build().drop_duplicates("camis").cuisine.value_counts()
           .drop("", errors="ignore").drop("Other", errors="ignore").head(TOP_CUISINES).index)
    cuisine_cols = []
    for c in top:
        col = "cuisine_" + c.lower().replace("/", "_").replace(",", "").replace(" ", "_")
        base[col] = (base.cuisine == c).astype(float)
        cuisine_cols.append(col)
    base["cuisine_other"] = (~base.cuisine.isin(top)).astype(float)
    cuisine_cols.append("cuisine_other")
    boro_cols = []
    for b in BOROS:
        col = "boro_" + b.lower().replace(" ", "_")
        base[col] = (base.boro == b).astype(float)
        boro_cols.append(col)
    base["year"] = base.inspection_date.dt.year.astype(float)
    base["n_prior"] = base.n_prior.astype(float)
    x_cols = cuisine_cols + boro_cols + X_BASE + (["n_critical"] if with_critical else [])
    base = base.dropna(subset=["score", "Y", *x_cols]).reset_index(drop=True)
    base.attrs["x_cols"] = x_cols
    return base


def _sample(d: pd.DataFrame, outcome: str, description: str) -> RDDSample:
    x_cols = d.attrs["x_cols"]
    return RDDSample(
        Q=d.score.to_numpy(float), X=d[x_cols].to_numpy(float), Y=d.Y.to_numpy(float),
        threshold=THRESHOLD, name=f"nyc_restaurants_{outcome}", feature_names=list(x_cols),
        description=description,
        citation=("NYC DOHMH, New York City Restaurant Inspection Results, NYC Open Data "
                  "43nn-pn8j (pull of 2026-11-10)"),
        treatment_rule=lambda Q: (np.asarray(Q) <= THRESHOLD).astype(int),
        extras={"camis": d.camis.to_numpy(), "inspection_date": d.inspection_date.to_numpy(),
                "grade": d.grade.to_numpy(), "closed_at_inspection": d.closed.to_numpy(float),
                "n_critical": d.n_critical.to_numpy(float), "n_viol": d.n_viol.to_numpy(float),
                "next_days": d.next_days.to_numpy(float),
                "next_inspection_type": d.get("next_inspection_type", pd.Series(dtype=object)).to_numpy(),
                "next_score": d.get("next_score", pd.Series(dtype=float)).to_numpy(float)},
    )


def load(next_kind: str = "any", with_critical: bool = False) -> RDDSample:
    """Y = the next cycle inspection at least 60 days later scores 14 or more."""
    d = sample_frame("next_score_14", next_kind=next_kind, with_critical=with_critical)
    return _sample(d, "next_score_14", (
        "NYC restaurant cycle initial inspections 2022-2026. Q = inspection score (points); "
        "treatment = 1{Q <= 13.5} (graded A on the spot); Y = the next cycle "
        f"{'initial inspection' if next_kind == 'initial' else 'inspection (initial or re-inspection)'}"
        " at least 60 days later scores 14 or more."))


def load_exit(with_critical: bool = False) -> RDDSample:
    """Y = no inspection of any kind within 18 months (survivors only; see README)."""
    d = sample_frame("no_inspection_18m", with_critical=with_critical)
    return _sample(d, "no_inspection_18m", (
        "NYC restaurant cycle initial inspections 2022-2025. Q = inspection score (points); "
        "treatment = 1{Q <= 13.5} (graded A on the spot); Y = no inspection of any kind "
        "within 18 months. Only restaurants active at the record date are in the file."))


def mirrored(**kw):
    """(Q, X, Y, phi0, frame) with the score mirrored so that treatment is Q >= phi0:
    Q = -score, phi0 = -13.5."""
    d = sample_frame("next_score_14", **kw)
    return (-d.score.to_numpy(float), d[d.attrs["x_cols"]].to_numpy(float),
            d.Y.to_numpy(float), -THRESHOLD, d)
