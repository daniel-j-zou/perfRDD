"""E-Rate (USAC): the discount matrix as several cutoffs on the NSLP share.

The E-Rate program pays a share (the discount) of eligible telecom, internet and internal
connections costs for schools and libraries. 47 CFR 54.505(c) sets the discount from a
matrix in the share of students eligible for the National School Lunch Program (NSLP)
and the entity's urban/rural status:

    NSLP %      <1   1-19   20-34   35-49   50-74   75-100
    C1 urban    20    40     50      60      80      90
    C1 rural    25    50     60      70      80      90
    C2 (both)   as C1, except 85 in the top band

For a school district the share is district-wide (total NSLP-eligible / total students,
54.505(b)(1)), and the district is rural when more than 50% of its schools are rural.

Q = 100 * NSLP-eligible students / full-time students of the billed entity (school
    district or independent school), from its FCC Form 471 for funding year t.
Effective cutoffs on Q, found in the data (``rule_check``): 0.995, 19.5, 34.5, 49.5,
    74.5, with ties going up. That is, the published discount equals the matrix applied
    to round-half-up(Q) at the 20/35/50/75 edges, and to Q rounded to two decimals at
    the 1% edge.
D = load(): 1{Q >= 49.5}, the 80% band (C1 urban 60 -> 80, rural 70 -> 80).
    load_c20 / load_c35 / load_c75: the other edges.
X = log enrolment, log number of schools, rural, independent school, private / charter /
    tribal / ESA flags, billed-entity latitude and longitude, state and year indicators.
Y = load(): log(1 + pre-discount eligible dollars requested on all of the entity's
    FY t+1 Forms 471, as filed); 0 when it files none.
    load_committed(): log(1 + dollars committed on its FY t+1 funding requests).

Units are entity x funding year, t = 2022-2025 (2022-2024 for load_committed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "erate_panel.csv.gz"
APP_CACHE = PROCESSED / "erate_applications.csv.gz"
YEARS = (2022, 2023, 2024, 2025, 2026)
TYPES = ("School District", "School")

# Effective cutoffs on Q (percent), lowest band first; see rule_check().
CUTS = (0.995, 19.5, 34.5, 49.5, 74.5)
C1_URBAN = (20, 40, 50, 60, 80, 90)
C1_RURAL = (25, 50, 60, 70, 80, 90)
C2_URBAN = (20, 40, 50, 60, 80, 85)
C2_RURAL = (25, 50, 60, 70, 80, 85)

BI_COLS = ["application_number", "funding_year", "org_state", "form_version",
           "chosen_category_of_service", "epc_organization_id",
           "organization_entity_type_name", "certified_datetime", "full_time_students",
           "total_eligible_nslp_students", "nslp_percentage", "is_urban", "c1_discount",
           "c2_discount", "pre_discount_eligible_amount", "funding_request_amount",
           "latitude", "longitude"]
DC_COLS = ["application_number", "form_version", "par_entity_ben", "par_subtype",
           "par_alternative_dis", "entity_cep_percentage", "child_entity_type_name",
           "child_entity_ben", "child_entity_is_urban_or_rural",
           "child_entity_cep_percentage", "child_alternative_dis"]
FRN_COLS = ["funding_year", "form_version", "ben", "form_471_frn_status_name",
            "funding_commitment_request", "total_pre_discount_costs"]


def band(q: np.ndarray) -> np.ndarray:
    """Matrix band 0-5 of the NSLP share q (percent), using the effective cutoffs."""
    return np.searchsorted(np.asarray(CUTS), np.asarray(q, float), side="right")


def matrix(q: np.ndarray, rural: np.ndarray, category: int = 1) -> np.ndarray:
    tabs = {1: (C1_URBAN, C1_RURAL), 2: (C2_URBAN, C2_RURAL)}[category]
    b = band(q)
    return np.where(np.asarray(rural, bool), np.asarray(tabs[1])[b], np.asarray(tabs[0])[b])


def _check_raw() -> None:
    need = [RAW / f"basic_info_fy{y}.csv" for y in YEARS]
    need += [RAW / f"frn_status_fy{y}.csv" for y in YEARS]
    need += [RAW / f"discount_calc_fy{y}.csv" for y in YEARS]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{RAW} is missing {missing[:3]}... Run "
            "`python -m experiments.datasets.erate.download` (see README).")


def _applications() -> pd.DataFrame:
    """Form 471 Basic Information rows of school districts and independent schools."""
    frames = []
    for y in YEARS:
        df = pd.read_csv(RAW / f"basic_info_fy{y}.csv", usecols=BI_COLS,
                         dtype={"application_number": str, "epc_organization_id": str})
        frames.append(df[df["organization_entity_type_name"].isin(TYPES)])
    a = pd.concat(frames, ignore_index=True).rename(columns={"epc_organization_id": "ben"})
    a["q"] = 100 * a["total_eligible_nslp_students"] / a["full_time_students"]
    a["rural"] = (a["is_urban"] == "Rural").astype(int)
    a["c1"] = (100 * a["c1_discount"]).round().astype(int)
    a["c2"] = (100 * a["c2_discount"]).round().astype(int)
    return a


def _pick(a: pd.DataFrame, first: str) -> pd.DataFrame:
    """One version per application: `first` ('Current' or 'Original') when present."""
    a = a.assign(_v=(a["form_version"] != first).astype(int))
    return a.sort_values(["application_number", "_v"]).drop_duplicates(
        "application_number").drop(columns="_v")


def _discount_calc(apps: pd.DataFrame) -> pd.DataFrame:
    """Per chosen application: subtype flags and the billed entity's schools."""
    keep = set(zip(apps["application_number"], apps["form_version"]))
    out = []
    for y in YEARS:
        for chunk in pd.read_csv(RAW / f"discount_calc_fy{y}.csv", usecols=DC_COLS,
                                 dtype=str, chunksize=400_000):
            m = [k in keep for k in zip(chunk["application_number"], chunk["form_version"])]
            out.append(chunk[m])
    dc = pd.concat(out, ignore_index=True)
    dc = dc.merge(apps[["application_number", "ben"]], on="application_number")
    dc = dc[dc["par_entity_ben"] == dc["ben"]]
    sch = dc[dc["child_entity_type_name"] == "School"]
    cep = pd.to_numeric(sch["child_entity_cep_percentage"], errors="coerce").fillna(0)
    per = sch.assign(rural_child=(sch["child_entity_is_urban_or_rural"] == "Rural"),
                     cep_child=cep > 0,
                     alt_child=sch["child_alternative_dis"].notna()).groupby(
        "application_number").agg(n_schools=("child_entity_ben", "nunique"),
                                  share_rural_schools=("rural_child", "mean"),
                                  share_cep_schools=("cep_child", "mean"),
                                  share_alt_schools=("alt_child", "mean"))
    nif = dc[dc["child_entity_type_name"].fillna("").str.startswith("Non-Instructional")]
    per = per.join(nif.groupby("application_number")["child_entity_ben"].nunique()
                   .rename("n_nifs"), how="outer")
    par = dc.drop_duplicates("application_number").set_index("application_number")
    sub = par["par_subtype"].fillna("")
    par_cep = pd.to_numeric(par["entity_cep_percentage"], errors="coerce").fillna(0)
    flags = pd.DataFrame({
        "private": sub.str.contains("Private").astype(int),
        "charter": sub.str.contains("Charter").astype(int),
        "tribal": sub.str.contains("Tribal|BIE", regex=True).astype(int),
        "esa": sub.str.contains("ESA").astype(int),
        "par_cep": (par_cep > 0).astype(int),
        "par_alt": par["par_alternative_dis"].notna().astype(int),
        "subtype": sub,
    })
    return flags.join(per, how="left").reset_index()


def _build() -> pd.DataFrame:
    _check_raw()
    a = _applications()
    cur = _pick(a, "Current")      # post-review version: the discount actually applied
    orig = _pick(a, "Original")    # as filed: the dollars the applicant requested
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cur.to_csv(APP_CACHE, index=False)

    # Entity-year score: a Category 1 application first, then the earliest certified.
    cur = cur.assign(_c2=(cur["chosen_category_of_service"] != "Category1").astype(int))
    ey = cur.sort_values(["ben", "funding_year", "_c2", "certified_datetime",
                          "application_number"]).drop_duplicates(["ben", "funding_year"])
    ey = ey.drop(columns="_c2")
    nq = cur.groupby(["ben", "funding_year"])["q"].nunique().rename("n_q_values")
    ey = ey.merge(nq.reset_index(), on=["ben", "funding_year"])
    ey = ey.merge(_discount_calc(ey), on="application_number", how="left")
    ey.loc[ey["organization_entity_type_name"] == "School", "n_schools"] = 1

    req = orig.groupby(["ben", "funding_year"]).agg(
        req_pre=("pre_discount_eligible_amount", "sum"),
        req_frn=("funding_request_amount", "sum"))
    req_c1 = orig[orig["chosen_category_of_service"] == "Category1"].groupby(
        ["ben", "funding_year"])["pre_discount_eligible_amount"].sum().rename("req_pre_c1")
    frames = [pd.read_csv(RAW / f"frn_status_fy{y}.csv", usecols=FRN_COLS,
                          dtype={"ben": str}) for y in YEARS]
    frn = pd.concat(frames, ignore_index=True)
    frn = frn[(frn["form_version"] == "Current")
              & (frn["form_471_frn_status_name"] == "Funded")]
    com = frn.groupby(["ben", "funding_year"]).agg(
        committed=("funding_commitment_request", "sum"),
        committed_pre=("total_pre_discount_costs", "sum"))
    out = pd.concat([req, req_c1, com], axis=1).fillna(0.0).reset_index()

    nxt = out.assign(funding_year=out["funding_year"] - 1).rename(
        columns={c: f"{c}_next" for c in out.columns if c not in ("ben", "funding_year")})
    same = out.rename(columns={c: f"{c}_same" for c in out.columns
                               if c not in ("ben", "funding_year")})
    ey = ey.merge(same, on=["ben", "funding_year"], how="left")
    ey = ey.merge(nxt, on=["ben", "funding_year"], how="left")
    filed = set(zip(a["ben"], a["funding_year"]))
    ey["files_next"] = [int((b, y + 1) in filed) for b, y in zip(ey["ben"], ey["funding_year"])]
    for c in [c for c in ey.columns if c.endswith("_next")]:
        ey[c] = ey[c].fillna(0.0)
    ey.to_csv(CACHE, index=False)
    return ey


def panel(rebuild: bool = False) -> pd.DataFrame:
    """Entity x funding-year panel, FY 2022-2026 (cached in data/processed/)."""
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, dtype={"ben": str, "application_number": str})
    return _build()


def rule_check() -> pd.DataFrame:
    """Published C1/C2 discount vs the matrix applied to Q, every application version."""
    _check_raw()
    a = _applications()
    a = a.assign(m1=matrix(a["q"], a["rural"], 1), m2=matrix(a["q"], a["rural"], 2))
    a["ok_c1"] = a["m1"] == a["c1"]
    a["ok_c2"] = a["m2"] == a["c2"]
    return a.groupby(["funding_year", "form_version"]).agg(
        rows=("q", "size"), agree_c1=("ok_c1", "mean"), agree_c2=("ok_c2", "mean"),
        mismatches_c1=("ok_c1", lambda s: int((~s).sum())))


def _x(df: pd.DataFrame):
    lat = df["latitude"].where(df["latitude"].between(-15, 72))
    lon = df["longitude"].where(df["longitude"].between(-180, 146))
    miss_geo = (lat.isna() | lon.isna()).astype(int)
    st_mean = df.assign(lat=lat, lon=lon).groupby("org_state")[["lat", "lon"]].transform("mean")
    base = pd.DataFrame({
        "log_students": np.log(df["full_time_students"]),
        "log_schools": np.log(df["n_schools"].fillna(1).clip(lower=1)),
        "rural": df["rural"],
        "independent_school": (df["organization_entity_type_name"] == "School").astype(int),
        "private": df["private"].fillna(0),
        "charter": df["charter"].fillna(0),
        "tribal": df["tribal"].fillna(0),
        "esa": df["esa"].fillna(0),
        "latitude": lat.fillna(st_mean["lat"]).fillna(0),
        "longitude": lon.fillna(st_mean["lon"]).fillna(0),
        "geo_missing": miss_geo,
    })
    states = sorted(df["org_state"].unique())
    years = sorted(df["funding_year"].unique())
    st = pd.get_dummies(df["org_state"]).reindex(columns=states[1:], fill_value=0)
    st.columns = [f"state_{s}" for s in st.columns]
    yr = pd.get_dummies(df["funding_year"]).reindex(columns=years[1:], fill_value=0)
    yr.columns = [f"fy_{y}" for y in yr.columns]
    X = pd.concat([base, st, yr], axis=1).astype(float)
    return X.to_numpy(), list(X.columns)


def _sample(cut: float, y_col: str, years: tuple, label: str, y_text: str) -> RDDSample:
    df = panel()
    df = df[df["funding_year"].isin(years) & df["q"].between(0, 100)].reset_index(drop=True)
    X, names = _x(df)
    Q = df["q"].to_numpy(float)
    k = CUTS.index(cut)
    rural = df["rural"].to_numpy(bool)
    above = (k + 1)
    pub_band_c1 = np.where(rural, np.searchsorted(C1_RURAL, df["c1"], side="left"),
                           np.searchsorted(C1_URBAN, df["c1"], side="left"))
    group = np.where(rural, "rural", "urban")
    lo_u, hi_u = C1_URBAN[k], C1_URBAN[k + 1]
    lo_r, hi_r = C1_RURAL[k], C1_RURAL[k + 1]
    return RDDSample(
        Q=Q,
        X=X,
        Y=np.log1p(df[y_col].clip(lower=0).to_numpy(float)),
        threshold=cut,
        name=label,
        feature_names=names,
        description=(
            f"E-Rate discount matrix, FY {years[0]}-{years[-1]} school districts and "
            f"independent schools. Q = NSLP-eligible share of students (%), cutoff {cut} "
            f"(C1 discount urban {lo_u}->{hi_u}, rural {lo_r}->{hi_r}). Y = {y_text}."),
        citation=("USAC Open Data (opendata.usac.org): FCC Form 471 Basic Information "
                  "9s6i-myen, FRN Status qdmp-ygft, Discount Calculations upfy-khtr; "
                  "47 CFR 54.505."),
        treatment_rule=lambda Q, c=cut: (np.asarray(Q) >= c).astype(int),
        extras={
            "ben": df["ben"].to_numpy(),
            "year": df["funding_year"].to_numpy(),
            "group": group,
            "group_cutoffs": {"urban": cut, "rural": cut},
            "cutoff": cut,
            "cutoffs": CUTS,
            "discount_c1": df["c1"].to_numpy(),
            "discount_c2": df["c2"].to_numpy(),
            "eligible_published": (pub_band_c1 >= above).astype(int),
            "cep_parent": df["par_cep"].fillna(0).to_numpy(),
            "share_cep_schools": df["share_cep_schools"].to_numpy(),
            "alt_discount": df["par_alt"].fillna(0).to_numpy(),
            "n_q_values": df["n_q_values"].to_numpy(),
            "files_next": df["files_next"].to_numpy(),
            "raw": df,
        },
    )


def load() -> RDDSample:
    """80% band edge (Q >= 49.5); Y = log(1 + FY t+1 pre-discount dollars requested)."""
    return _sample(49.5, "req_pre_next", (2022, 2023, 2024, 2025), "erate",
                   "log(1 + pre-discount eligible dollars requested on FY t+1 Forms 471)")


def load_c20() -> RDDSample:
    return _sample(19.5, "req_pre_next", (2022, 2023, 2024, 2025), "erate_c20",
                   "log(1 + pre-discount eligible dollars requested on FY t+1 Forms 471)")


def load_c35() -> RDDSample:
    return _sample(34.5, "req_pre_next", (2022, 2023, 2024, 2025), "erate_c35",
                   "log(1 + pre-discount eligible dollars requested on FY t+1 Forms 471)")


def load_c75() -> RDDSample:
    return _sample(74.5, "req_pre_next", (2022, 2023, 2024, 2025), "erate_c75",
                   "log(1 + pre-discount eligible dollars requested on FY t+1 Forms 471)")


def load_committed() -> RDDSample:
    """80% band edge; Y = log(1 + FY t+1 dollars committed on funded FRNs), t <= 2024."""
    return _sample(49.5, "committed_next", (2022, 2023, 2024), "erate_committed",
                   "log(1 + dollars committed on FY t+1 funded FRNs)")
