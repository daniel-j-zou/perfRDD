"""US census tracts: the low-income-community (LIC) test as an either-cutoff rule.

The New Markets Tax Credit (26 USC 45D(e)) and the 2018 Opportunity Zone (OZ)
designations use the same tract eligibility test:

    LIC = 1{poverty rate >= 20%  or  MFI ratio <= 80%},

where the MFI ratio is tract median family income over a benchmark: the greater of
statewide and metro-area MFI for metro tracts, statewide MFI for non-metro tracts
(possession-wide MFI in Puerto Rico). The 2018 OZ round used ACS 2011-2015 5-year
estimates, published by the CDFI Fund per tract.

Q = (poverty rate %, MFI ratio %), CDFI's published 2011-2015 values, natural units.
D = 1{Q1 >= 20 or Q2 <= 80}. Applied to the published scores it reproduces CDFI's own
    LIC flag; see ``rule_check()``.
X = 2010 decennial tract counts (Census Planning Database 2019, ``*_CEN_2010`` columns
    only) and the OMB metro flag, plus state indicators. No ACS estimate enters X.
Y = load(): change in log(1 + LODES7 workplace jobs), 2017 to 2019.
    load_placebo(): the same change 2015 to 2017 (before designation).
    load_lihtc(): any HUD LIHTC property placed in service 2019 or later.

OZ designation (governors picked about a quarter of eligible tracts) is fuzzy and is
stored in extras; eligibility D is the sharp, intent-to-treat treatment.
"""
from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.us_tracts_lic._excel import read_xlsb, read_xlsx
from experiments.datasets.us_tracts_lic.download import LODES_YEARS, STATES

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "us_tracts_lic.csv.gz"
NMTC = RAW / "nmtc-2011-2015-lic-nov2-2017-4pm.xlsx"
OZ_RESOURCE = RAW / "ozone-information-resource.2.27.18-locked2.xlsb"
OZ_DESIGNATED = RAW / "designated-qozs.12.14.18.xlsx"
ACS_DIR = RAW / "acs2015"
PDB = RAW / "pdb2019trv6_us.zip"
LODES_DIR = RAW / "lodes7"
LIHTC = RAW / "lihtc_hud.csv"

POV_CUT, MFI_CUT = 20.0, 80.0
THRESHOLD = (POV_CUT, MFI_CUT)
BASE_X = ["log_pop", "log_density", "urbanized", "urban_cluster", "age_under5", "age_5_17",
          "age_18_24", "age_45_64", "age_65plus", "hispanic", "nh_black", "nh_asian",
          "nh_other", "inst_gq", "noninst_gq", "family_hh", "married_hh", "female_head_hh",
          "single_person_hh", "hh_with_kids", "hh_size", "renter", "vacant",
          "mail_return_rate", "mail_return_missing", "metro"]
STATE_FIPS = [f for f, st, _ in STATES if st != "pr"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing[0]} missing. Run `python -m experiments.datasets.us_tracts_lic.download`.")


# --------------------------------------------------------------------------- CDFI files
@lru_cache(maxsize=1)
def _tract_renumbering() -> dict:
    """Post-2010 tract number -> 2010 tract number (CDFI's 'Tract Number Changes' tab:
    26 tracts in AK, AZ, CA, NY, SD and VA renumbered without boundary changes)."""
    rows = read_xlsb(OZ_RESOURCE, "Tract Number Changes")[2:]
    return {str(r[2]).strip(): str(r[1]).strip() for r in rows if len(r) > 2 and r[1] and r[2]}


def _to2010(geoid: pd.Series) -> pd.Series:
    """Put every source on 2010 tract numbers (the numbering of LODES7 blocks)."""
    renum = _tract_renumbering()
    return geoid.map(lambda g: renum.get(g, g))


def _cdfi_scores() -> pd.DataFrame:
    rows = read_xlsx(NMTC, "NMTC LICs 2011-2015 ACS")[1:]
    df = pd.DataFrame([r[:15] + [None] * (15 - len(r)) for r in rows], columns=[
        "geoid", "metro_cdfi", "lic_published", "pov", "pov_ok_published", "mfi_ratio",
        "mfi_ok_published", "unemp", "county", "st", "state_name", "county_name",
        "unemp_ratio", "unemp_flag", "pov_universe_cdfi"])
    for c in ["pov", "mfi_ratio", "unemp", "pov_universe_cdfi"]:
        df[c] = _num(df[c])
    for c in ["lic_published", "pov_ok_published", "mfi_ok_published"]:
        df[c] = (df[c].str.strip() == "Yes").astype(float)
    df["geoid"] = _to2010(df.geoid.str.strip())
    df["metro"] = df.metro_cdfi.str.strip().str.lower().eq("metropolitan").astype(float)
    return df.drop(columns=["unemp_ratio", "unemp_flag", "county_name"])


def _oz_lists() -> pd.DataFrame:
    """OZ eligibility (CDFI information resource, Feb 27 2018) and designation (Dec 14 2018)."""
    renum = _tract_renumbering()
    fix = lambda g: renum.get(g, g)                                         # noqa: E731
    lic = {fix(str(r[1]).strip()) for r in read_xlsb(OZ_RESOURCE, "1)LICs")[1:] if r[1]}
    contig = {fix(str(r[1]).strip()) for r in read_xlsb(OZ_RESOURCE, "2)Contiguous")[1:]
              if len(r) > 1 and r[1]}
    contig |= {fix(str(r[1]).strip()) for r in read_xlsb(OZ_RESOURCE, "3)Cross-State")[2:]
               if len(r) > 1 and r[1]}
    rows = read_xlsx(OZ_DESIGNATED, _first_sheet(OZ_DESIGNATED))[5:]
    des = pd.DataFrame([(fix(r[2].strip()), r[3], r[4]) for r in rows if len(r) > 4 and r[2]],
                       columns=["geoid", "oz_tract_type", "oz_acs_source"])
    out = pd.DataFrame({"geoid": sorted(lic | contig | set(des.geoid))})
    out["oz_lic_listed"] = out.geoid.isin(lic).astype(float)
    out["oz_contiguous_listed"] = out.geoid.isin(contig).astype(float)
    out = out.merge(des, on="geoid", how="left")
    out["oz_designated"] = out.oz_tract_type.notna().astype(float)
    return out


def _first_sheet(path: Path) -> str:
    wb = zipfile.ZipFile(path).read("xl/workbook.xml").decode("utf-8")
    return re.search(r'<sheet [^>]*name="([^"]+)"', wb).group(1)


# --------------------------------------------------------------------------- ACS 2011-2015
def _geo(st: str) -> pd.DataFrame:
    g = pd.read_csv(ACS_DIR / f"g20155{st}.csv", header=None, dtype=str, encoding="latin-1",
                    usecols=[2, 3, 4, 9, 10, 13, 22, 48, 49], keep_default_na=False)
    g.columns = ["sumlev", "comp", "logrecno", "state", "county", "tract", "cbsa", "geoid",
                 "name"]
    return g[g.comp == "00"]


def _seq(st: str, seq: str, cols: dict, allgeo: bool = False) -> pd.DataFrame:
    folder = ACS_DIR / "allgeo" if allgeo else ACS_DIR
    with zipfile.ZipFile(folder / f"20155{st}{seq}000.zip") as z:
        raw = z.read(f"e20155{st}{seq}000.txt")
    e = pd.read_csv(io.BytesIO(raw), header=None, dtype=str, usecols=[5] + list(cols))
    e.columns = ["logrecno"] + list(cols.values())
    for c in cols.values():
        e[c] = _num(e[c])
    return e


def _acs() -> pd.DataFrame:
    """Recompute both scores from the ACS 2011-2015 summary file: poverty from B17001,
    MFI from B19113 for the tract, the state and the CBSA (county -> CBSA from the
    summary-file geography, summary level 322)."""
    g = _geo("us")
    cbs = [g[g.sumlev == "310"].merge(_seq("us", "0064", {74: "mfi"}), on="logrecno")]
    parts, st_mfi = [], []
    for _, st, _ in STATES:
        g = _geo(st)
        e = _seq(st, "0064", {74: "mfi"}, allgeo=True)
        st_mfi.append(g[g.sumlev == "040"][["logrecno", "state"]].merge(e, on="logrecno")
                      .rename(columns={"mfi": "state_mfi"}))
        if st == "pr":      # Puerto Rico's CBSAs are not in the US file; all lie within PR
            cbs.append(g[g.sumlev == "320"].merge(e, on="logrecno"))
        t = g[g.sumlev == "140"][["logrecno", "geoid"]]
        t = t.merge(_seq(st, "0048", {6: "pov_universe", 7: "pov_count"}), on="logrecno")
        t = t.merge(_seq(st, "0064", {74: "mfi_tract"}), on="logrecno")
        cty = g[g.sumlev == "322"][["state", "county", "cbsa"]].drop_duplicates(["state", "county"])
        t["geoid"] = _to2010(t.geoid.str[-11:])
        t["state"], t["county"] = t.geoid.str[:2], t.geoid.str[2:5]
        parts.append(t.merge(cty, on=["state", "county"], how="left").drop(columns="logrecno"))
    t = pd.concat(parts, ignore_index=True)
    cb = pd.concat(cbs)
    cb = pd.DataFrame({"cbsa": cb.cbsa, "cbsa_mfi": cb.mfi,
                       "cbsa_is_metro": cb.name.str.contains("Metro Area").astype(float)})
    st_mfi = pd.concat(st_mfi)[["state", "state_mfi"]]
    t = t.merge(st_mfi, on="state", how="left")
    t = t.merge(cb, on="cbsa", how="left")
    # published to one decimal, rounded half up (S1701)
    t["pov_acs"] = np.floor(1000 * t.pov_count / t.pov_universe.where(t.pov_universe > 0)
                            + 0.5 + 1e-9) / 10
    return t.drop(columns=["county"])


def _benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """The benchmark: max(state, metro) MFI for metro tracts, state MFI otherwise.
    Metro status is CDFI's (OMB 15-01); the metro MFI is that of the county's CBSA in
    the ACS geography. CDFI treats Puerto Rico like a state (its San Juan tracts are
    benchmarked on the metro MFI), so this does too."""
    metro_mfi = df.cbsa_mfi.where((df.metro == 1) & (df.cbsa_is_metro == 1))
    use_metro = metro_mfi > df.state_mfi
    df["metro_mfi"] = metro_mfi
    df["benchmark_mfi"] = np.where(use_metro, metro_mfi, df.state_mfi)
    df["benchmark_type"] = np.where(df.state_mfi.isna(), "", np.where(use_metro, "metro", "state"))
    df["mfi_ratio_acs"] = 100 * df.mfi_tract / df.benchmark_mfi
    df["benchmark_implied_cdfi"] = 100 * df.mfi_tract / df.mfi_ratio
    return df


# --------------------------------------------------------------------------- covariates
def _pdb() -> pd.DataFrame:
    """2010 decennial counts from the Census Planning Database 2019 tract file."""
    cols = ["GIDTR", "LAND_AREA", "URBANIZED_AREA_POP_CEN_2010", "URBAN_CLUSTER_POP_CEN_2010",
            "Tot_Population_CEN_2010", "Pop_under_5_CEN_2010", "Pop_5_17_CEN_2010",
            "Pop_18_24_CEN_2010", "Pop_45_64_CEN_2010", "Pop_65plus_CEN_2010", "Inst_GQ_CEN_2010",
            "Non_Inst_GQ_CEN_2010", "Hispanic_CEN_2010", "NH_White_alone_CEN_2010",
            "NH_Blk_alone_CEN_2010", "NH_Asian_alone_CEN_2010", "Rel_Family_HHD_CEN_2010",
            "MrdCple_Fmly_HHD_CEN_2010", "Female_No_HB_CEN_2010", "Sngl_Prns_HHD_CEN_2010",
            "HHD_PPL_Und_18_CEN_2010", "Tot_Prns_in_HHD_CEN_2010", "Tot_Housing_Units_CEN_2010",
            "Tot_Occp_Units_CEN_2010", "Tot_Vacant_Units_CEN_2010", "Renter_Occp_HU_CEN_2010",
            "Mail_Return_Rate_CEN_2010"]
    with zipfile.ZipFile(PDB) as z:
        with z.open(z.namelist()[0]) as fh:
            p = pd.read_csv(fh, usecols=cols, dtype={"GIDTR": str}, encoding="latin-1")
    p["geoid"] = _to2010(p.GIDTR.str.zfill(11))
    c = {k: _num(p[k]) for k in cols[1:]}
    pop = c["Tot_Population_CEN_2010"]
    hh = c["Tot_Occp_Units_CEN_2010"]
    pp, ph = pop.where(pop > 0), hh.where(hh > 0)
    hu = c["Tot_Housing_Units_CEN_2010"].where(c["Tot_Housing_Units_CEN_2010"] > 0)
    out = pd.DataFrame({
        "geoid": p.geoid, "pop2010": pop,
        "log_pop": np.log1p(pop),
        "log_density": np.log1p(pop / c["LAND_AREA"].where(c["LAND_AREA"] > 0)),
        "urbanized": c["URBANIZED_AREA_POP_CEN_2010"] / pp,
        "urban_cluster": c["URBAN_CLUSTER_POP_CEN_2010"] / pp,
        "age_under5": c["Pop_under_5_CEN_2010"] / pp, "age_5_17": c["Pop_5_17_CEN_2010"] / pp,
        "age_18_24": c["Pop_18_24_CEN_2010"] / pp, "age_45_64": c["Pop_45_64_CEN_2010"] / pp,
        "age_65plus": c["Pop_65plus_CEN_2010"] / pp,
        "hispanic": c["Hispanic_CEN_2010"] / pp, "nh_black": c["NH_Blk_alone_CEN_2010"] / pp,
        "nh_asian": c["NH_Asian_alone_CEN_2010"] / pp,
        "nh_other": 1 - (c["Hispanic_CEN_2010"] + c["NH_White_alone_CEN_2010"]
                         + c["NH_Blk_alone_CEN_2010"] + c["NH_Asian_alone_CEN_2010"]) / pp,
        "inst_gq": c["Inst_GQ_CEN_2010"] / pp, "noninst_gq": c["Non_Inst_GQ_CEN_2010"] / pp,
        "family_hh": c["Rel_Family_HHD_CEN_2010"] / ph,
        "married_hh": c["MrdCple_Fmly_HHD_CEN_2010"] / ph,
        "female_head_hh": c["Female_No_HB_CEN_2010"] / ph,
        "single_person_hh": c["Sngl_Prns_HHD_CEN_2010"] / ph,
        "hh_with_kids": c["HHD_PPL_Und_18_CEN_2010"] / ph,
        "hh_size": c["Tot_Prns_in_HHD_CEN_2010"] / ph,
        "renter": c["Renter_Occp_HU_CEN_2010"] / ph,
        "vacant": c["Tot_Vacant_Units_CEN_2010"] / hu,
        "mail_return_rate": c["Mail_Return_Rate_CEN_2010"],
    })
    # no mail-back rate where the 2010 census used update/leave or enumeration only
    out["mail_return_missing"] = out.mail_return_rate.isna().astype(float)
    out["mail_return_rate"] = out.mail_return_rate.fillna(out.mail_return_rate.median())
    # The 2019 file also lists post-2010 tracts (no 2010 counts) next to the 2010 ones,
    # and both numbers of a renumbered tract; keep the 2010 record.
    out = out[out.pop2010.notna()].sort_values("pop2010", ascending=False)
    return out.drop_duplicates("geoid").sort_index()


# --------------------------------------------------------------------------- outcomes
def _lodes() -> pd.DataFrame:
    """Workplace jobs (C000) per 2010 tract. A tract absent from a state-year file has
    zero jobs; a state-year file that LODES7 does not publish gives NaN."""
    frames = {}
    for _, st, _ in STATES:
        if st == "pr":
            continue
        for y in LODES_YEARS:
            f = LODES_DIR / f"{st}_wac_S000_JT00_{y}.csv.gz"
            if f.exists():
                w = pd.read_csv(f, usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
                s = w.groupby(w.w_geocode.str[:11]).C000.sum()
                frames.setdefault(y, []).append(s)
    out = pd.concat({y: pd.concat(v) for y, v in frames.items()}, axis=1)
    out.columns = [f"jobs_{y}" for y in out.columns]
    out.index.name = "geoid"
    return out.reset_index()


def _lodes_states() -> dict:
    return {y: {st for _, st, _ in STATES
                if (LODES_DIR / f"{st}_wac_S000_JT00_{y}.csv.gz").exists()} for y in LODES_YEARS}


def _lihtc() -> pd.DataFrame:
    """HUD LIHTC projects per tract by year placed in service (YR_PIS)."""
    h = pd.read_csv(LIHTC, dtype=str)
    h = h[h.STATE2KX.notna() & h.CNTY2KX.notna() & h.TRACT2KX.notna()]
    tr = h.TRACT2KX.str.replace(".", "", regex=False).str.zfill(6)
    h["geoid"] = _to2010(h.STATE2KX.str.zfill(2) + h.CNTY2KX.str.zfill(3) + tr)
    yr = _num(h.YR_PIS)
    units = _num(h.N_UNITS).fillna(0)
    post, pre = yr.between(2019, 2030), yr.between(2015, 2018)
    g = pd.DataFrame({"geoid": h.geoid, "post": post.astype(float), "pre": pre.astype(float),
                      "post_units": units.where(post, 0), "pre_units": units.where(pre, 0)})
    g = g.groupby("geoid").sum().reset_index()
    return g.rename(columns={"post": "lihtc_proj_2019on", "pre": "lihtc_proj_2015_18",
                             "post_units": "lihtc_units_2019on", "pre_units": "lihtc_units_2015_18"})


# --------------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    """One row per 2010 tract in CDFI's NMTC file, with the published and recomputed
    scores, the benchmark, OZ status, 2010 covariates and the outcomes."""
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"geoid": str, "state": str, "county": str, "cbsa": str})
    _require(NMTC, OZ_RESOURCE, OZ_DESIGNATED, ACS_DIR / "g20155us.csv", PDB, LODES_DIR, LIHTC)
    df = _cdfi_scores()
    df["state"] = df.geoid.str[:2]
    df = df.merge(_acs().drop(columns=["state"]), on="geoid", how="left")
    df = _benchmark(df)
    df = df.merge(_oz_lists(), on="geoid", how="left")
    for c in ["oz_lic_listed", "oz_contiguous_listed", "oz_designated"]:
        df[c] = df[c].fillna(0.0)
    df = df.merge(_pdb(), on="geoid", how="left")
    jobs = _lodes()
    df = df.merge(jobs, on="geoid", how="left")
    have = _lodes_states()
    abbr = {f: st for f, st, _ in STATES}
    for y in LODES_YEARS:
        in_file = df.state.map(abbr).isin(have[y])
        df[f"jobs_{y}"] = np.where(in_file, df[f"jobs_{y}"].fillna(0.0), np.nan)
    df["dlog_jobs_17_19"] = np.log1p(df.jobs_2019) - np.log1p(df.jobs_2017)
    df["dlog_jobs_15_17"] = np.log1p(df.jobs_2017) - np.log1p(df.jobs_2015)
    df = df.merge(_lihtc(), on="geoid", how="left")
    lihtc_cols = ["lihtc_proj_2019on", "lihtc_proj_2015_18", "lihtc_units_2019on",
                  "lihtc_units_2015_18"]
    df[lihtc_cols] = df[lihtc_cols].fillna(0.0)
    df["any_lihtc_2019on"] = (df.lihtc_proj_2019on > 0).astype(float)
    df["any_lihtc_2015_18"] = (df.lihtc_proj_2015_18 > 0).astype(float)
    df["D"] = rule(df.pov.to_numpy(), df.mfi_ratio.to_numpy())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def rule(pov: np.ndarray, mfi_ratio: np.ndarray) -> np.ndarray:
    """The LIC test; a missing score never qualifies (CDFI's convention for N/A)."""
    pov, mfi_ratio = np.asarray(pov, float), np.asarray(mfi_ratio, float)
    with np.errstate(invalid="ignore"):
        return ((pov >= POV_CUT) | (mfi_ratio <= MFI_CUT)).astype(int)


def _either_rule(Q: np.ndarray) -> np.ndarray:
    return rule(Q[:, 0], Q[:, 1])


def rule_check() -> dict:
    """Agreement of the rule with CDFI's flags, and of the recomputed ACS scores with
    CDFI's published scores, over all tracts in CDFI's file."""
    df = build()
    d = rule(df.pov, df.mfi_ratio)
    acs = df.state.isin([f for f, _, _ in STATES])          # 50 states, DC, Puerto Rico
    both = df.mfi_ratio.notna() & df.mfi_ratio_acs.notna()
    d_acs = rule(df.pov_acs, df.mfi_ratio_acs)
    return {
        "tracts": int(len(df)),
        "rule_vs_cdfi_lic_flag": float(np.mean(d == df.lic_published)),
        "rule_vs_cdfi_lic_flag_mismatches": int(np.sum(d != df.lic_published)),
        "rule_vs_oz_lic_list": float(np.mean(d == df.oz_lic_listed)),
        "rule_eligible_not_on_oz_list": int(np.sum((d == 1) & (df.oz_lic_listed == 0))),
        "oz_listed_not_rule_eligible": int(np.sum((d == 0) & (df.oz_lic_listed == 1))),
        "acs_tracts": int(acs.sum()),
        "acs_recomputed_rule_vs_cdfi_flag": float(np.mean(d_acs[acs] == df.lic_published[acs])),
        "acs_pov_equal_to_cdfi": float(np.mean(
            (np.abs(df.pov_acs - df.pov) < 0.05)[df.pov.notna() & acs])),
        "acs_mfi_ratio_within_0.01": float(np.mean(
            (np.abs(df.mfi_ratio_acs - df.mfi_ratio) < 0.01)[both])),
        "benchmark_type_counts": df.loc[df.mfi_ratio.notna(), "benchmark_type"]
                                   .value_counts().to_dict(),
        "oz_designated": int(df.oz_designated.sum()),
        "oz_designated_rule_eligible": int(((df.oz_designated == 1) & (d == 1)).sum()),
    }


# --------------------------------------------------------------------------- samples
def _sample(y_col: str, label: str) -> RDDSample:
    df = build()
    df = df[df.state.isin(STATE_FIPS)]
    for f in STATE_FIPS[1:]:
        df = df.assign(**{f"st_{f}": (df.state == f).astype(float)})
    x_cols = BASE_X + [f"st_{f}" for f in STATE_FIPS[1:]]
    keep = df.pov.notna() & df.mfi_ratio.notna() & df[BASE_X].notna().all(axis=1) \
        & df[y_col].notna()
    df = df[keep]
    return RDDSample(
        Q=df[["pov", "mfi_ratio"]].to_numpy(float), X=df[x_cols].to_numpy(float),
        Y=df[y_col].to_numpy(float), threshold=THRESHOLD,
        name=f"us_tracts_lic_{y_col}", feature_names=x_cols,
        description=("US 2010 census tracts (50 states + DC). Q = (poverty rate %, MFI "
                     "ratio %), ACS 2011-2015 as published by CDFI; treatment = "
                     "1{Q1 >= 20 or Q2 <= 80} (NMTC low-income community, OZ-eligible "
                     f"2018); X = 2010 decennial; Y = {label}."),
        citation=("CDFI Fund NMTC 2011-2015 LIC data and OZ resources; Census ACS 2011-2015 "
                  "summary file; Census Planning Database 2019; LEHD LODES7; HUD LIHTC"),
        treatment_rule=_either_rule,
        extras={
            "geoid": df.geoid.to_numpy(), "state": df.state.to_numpy(),
            "eligible_published": df.lic_published.to_numpy(float),
            "oz_designated": df.oz_designated.to_numpy(float),
            "oz_tract_type": df.oz_tract_type.fillna("").to_numpy(),
            "oz_lic_listed": df.oz_lic_listed.to_numpy(float),
            "oz_contiguous_listed": df.oz_contiguous_listed.to_numpy(float),
            "metro": df.metro.to_numpy(float),
            "benchmark_type": df.benchmark_type.to_numpy(),
            "benchmark_mfi": df.benchmark_mfi.to_numpy(float),
            "state_mfi": df.state_mfi.to_numpy(float),
            "metro_mfi": df.metro_mfi.to_numpy(float),
            "mfi_tract": df.mfi_tract.to_numpy(float),
            "pov_universe": df.pov_universe.to_numpy(float),
            "jobs": df[[f"jobs_{y}" for y in LODES_YEARS]].to_numpy(float),
            "jobs_years": LODES_YEARS,
        },
    )


def load() -> RDDSample:
    return _sample("dlog_jobs_17_19", "change in log(1 + workplace jobs), LODES7 2017 to 2019")


def load_placebo() -> RDDSample:
    return _sample("dlog_jobs_15_17", "change in log(1 + workplace jobs), 2015 to 2017 (placebo)")


def load_lihtc() -> RDDSample:
    return _sample("any_lihtc_2019on", "any LIHTC property placed in service 2019 or later")
