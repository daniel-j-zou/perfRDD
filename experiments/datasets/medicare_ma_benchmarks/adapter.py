"""Medicare Advantage county benchmarks: the ACA applicable percentage as a dose.

Since 2012 a county's MA benchmark is a percentage of its per-capita fee-for-service (FFS)
rate. The percentage is set by the quartile of the county's FFS rate among the counties
of the 50 states and DC (Social Security Act 1853(n)): 115% (lowest-cost quartile),
107.5%, 100%, 95% (highest). See README.md for the rule as reproduced from the CMS rate
calculation files.

Panel: county (SSA code) x payment year t.
  Q        percentile rank (0-100) of the ranking FFS rate, i.e. the county's FFS rate of
           year t-1 (excluding the phased-out IME), among the ranked counties of year t.
           ``score="logratio"``: 100 log(FFS_{t-1} / 75th-percentile cutoff of year t).
  A        the applicable percentage given by the rule (``extras["A"]``), a fraction:
           the mean of the quartile percentages of years t and t-1 (one level in 2012).
  Y        MA penetration (%) in July of payment year t (``outcome="pen"``); also
           ``"dpen"`` (change from July t-1) and ``"pen_lead"`` (July t+1).
  X        lagged penetration and eligibles (July t-1), Census population, age 65+/85+
           shares, median age, SAIPE median income and poverty, density, the county's
           average FFS risk score, the 2004 urban-floor flag, year and state indicators.

    from experiments.datasets.medicare_ma_benchmarks import adapter
    s = adapter.load_dose()             # payment years 2017-2026 (no ACA transition)
    adapter.rule_check()                # formula vs published applicable percentages
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
EXTRACTED = RAW / "extracted"
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "ma_panel.csv.gz"
PANEL_VERSION = 2

PCT = np.array([1.15, 1.075, 1.0, 0.95])            # pure quartile 0 (lowest cost) .. 3
LEVELS = np.unique(np.round([(a + b) / 2 for a in PCT for b in PCT], 6))
A_REF = 0.95
RATE_YEARS = range(2012, 2028)
PEN_YEARS = range(2012, 2027)
STATES51 = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
               "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
# New Alaska county codes (Valdez-Cordova split) first ranked in 2024, before they had a
# prior-year FFS rate: CMS's 2024 cutoffs are reproduced when they carry the parent's rate.
PARENT = {"02063": "02261", "02066": "02261"}

CONT = ["pen_lag", "log_elig_lag", "log_pop", "share65", "share85", "median_age",
        "log_income", "poverty", "log_density", "ffs_risk"]
Z_NAMES = ["pen_lag", "log_income", "log_density", "urban_floor_2004"]


# --------------------------------------------------------------------------- extraction
def _extract(zpath: Path) -> Path:
    """Extract a downloaded archive into extracted/<stem>/ (idempotent, no path escapes)."""
    dest = EXTRACTED / zpath.stem
    if dest.exists() and any(dest.rglob("*")):
        return dest
    if not zpath.exists():
        raise FileNotFoundError(f"{zpath} missing. Run: python -m "
                                "experiments.datasets.medicare_ma_benchmarks.download")
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe member {m.filename!r} in {zpath}")
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
    return dest


def _num(s) -> float:
    s = str(s).replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- CMS rate files
def _risk_files(year: int):
    d = _extract(RAW / "ratebooks" / f"calc_{year}.zip")
    main = [p for p in d.rglob("*.csv") if p.name.lower() == f"risk{year}.csv"]
    if len(main) != 1:
        raise RuntimeError(f"risk{year}.csv not found uniquely in {d}")
    extra = [p for p in d.rglob("*.csv") if p.name.lower() == f"risk{year} revised pr.csv"]
    return main[0], extra


def read_risk(year: int) -> pd.DataFrame:
    """One row per county of CMS's risk<year>.csv (non-PACE rate calculation)."""
    def parse(path):
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="latin-1"))))
        i = next(k for k, r in enumerate(rows) if r and r[0].strip() == "Code")
        hdr = [re.sub(r"\s+", " ", h).strip() for h in rows[i]]
        data = [r + [""] * (len(hdr) - len(r)) for r in rows[i + 1:]
                if r and re.fullmatch(r"\d{4,5}|\d\dXXX", r[0].strip())]
        df = pd.DataFrame([r[:len(hdr)] for r in data], columns=range(len(hdr)))

        def col(pat, required=True):
            c = [k for k, h in enumerate(hdr) if re.search(pat, h, re.I)]
            if not c and required:
                raise KeyError(f"{path.name}: no column matching {pat!r}")
            return df[c[0]] if c else pd.Series([""] * len(df))

        y = year
        out = pd.DataFrame({"ssa": df[0].str.strip().str.zfill(5),
                            "state": df[1].str.strip(), "county": df[2].str.strip()})
        out["ffs_prev_col"] = df[3].map(_num)          # rate of year t-1 as printed in file t
        out["ffs"] = col(rf"^{y} FFS Rate (excluding Phase-out IME|\(excludes phased-out IME)").map(_num)
        out["preaca"] = col(rf"^{y} Pre-ACA Rate \(excludes phased-out IME").map(_num)
        out["trans_years"] = col(r"^Transition Years").map(_num)
        out["qpct"] = col(r"^Quartile Percent").map(_num)
        out["qualifying"] = col(r"^Qualifying County").str.strip().str.upper().eq("YES").astype(int)
        out["urban_floor_2004"] = col(r"^Urban Floor in 2004").str.strip().str.upper().eq("YES").astype(int)
        r0 = col(rf"^0% Bonus {y} Rate", required=False)
        if (r0 == "").all():
            r0 = col(rf"^<= ?2\.5 Star {y} Rate")
        out["rate0"] = r0.map(_num)
        out["capped0_pub"] = col(r"^0% Bonus Rate Capped", required=False).str.strip().str.upper().eq("YES").astype(int)
        return out

    main, extra = _risk_files(year)
    out = parse(main)
    for p in extra:                                   # 2014: revised Puerto Rico rows
        pr = parse(p)
        out = pd.concat([out[~out.ssa.isin(pr.ssa)], pr], ignore_index=True)
    out["year"] = year
    out["territory"] = (~out.state.isin(STATES51)).astype(int)
    return out


def read_ffs_risk_score(year: int) -> pd.Series:
    """County average FFS risk score (latest year) from the Non-PACE table posted with year's rates."""
    d = _extract(RAW / "ratebooks" / f"calc_{year}.zip")
    f = [p for p in d.rglob("*.csv") if "non-pace" in p.name.lower() and "dialysis" not in p.name.lower()]
    rows = list(csv.reader(io.StringIO(f[0].read_text(encoding="latin-1"))))
    vals = {}
    for r in rows:                      # year columns ascend, so the last number is the latest year
        if r and re.fullmatch(r"\d{4,5}", r[0].strip()):
            nums = [_num(c) for c in r[3:] if c.strip()]
            nums = [v for v in nums if v == v]
            if nums:
                vals[r[0].strip().zfill(5)] = nums[-1]
    return pd.Series(vals, name="ffs_risk", dtype=float)


def ratebook_panel() -> pd.DataFrame:
    """All payment years with the rule applied: ranking rate, Q, cutoffs, pure quartiles, A."""
    rb = pd.concat([read_risk(y) for y in RATE_YEARS], ignore_index=True)
    W = rb.pivot(index="ssa", columns="year", values="ffs")
    pure = {}
    rows = []
    for y in RATE_YEARS:
        g = rb[rb.year == y].set_index("ssa").copy()
        if y == 2012:                                 # 2012 ranks the "Estimated 2010 FFS Rate"
            v = g["ffs_prev_col"].copy()
        else:
            v = W[y - 1].reindex(g.index)
            for k, par in PARENT.items():
                if k in v.index and pd.isna(v[k]) and par in W.index:
                    v[k] = W.loc[par, y - 1]
        ranked = (g.territory == 0) & v.notna()
        rv = np.sort(v[ranked].to_numpy())
        n = len(rv)
        cuts = np.quantile(rv, [0.25, 0.5, 0.75])
        below = np.searchsorted(rv, v.fillna(np.inf).to_numpy(), side="left")   # strictly below
        g["rank_rate"] = v
        g["n_ranked"] = n
        g["Q_rank"] = np.where(v.notna(), 100.0 * below / (n - 1), np.nan)
        k = np.where(v.notna(), np.digitize(v.fillna(0).to_numpy(), cuts, right=True), -1)
        g["pure_q"] = k
        g["pure_pct"] = np.where(k >= 0, PCT[np.clip(k, 0, 3)], np.nan)
        g["cut25"], g["cut50"], g["cut75"] = cuts
        pure[y] = g["pure_pct"]
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    prev = []
    for y in RATE_YEARS:
        g = out[out.year == y]
        prev.append(g.ssa.map(pure[y - 1]) if y - 1 in pure else pd.Series(np.nan, index=g.index))
    out["prev_pct"] = pd.concat(prev)
    out["A_formula"] = np.where(out.prev_pct.notna(), (out.pure_pct + out.prev_pct) / 2, out.pure_pct)
    # ACA transition (1853(n)(3)): share of the pre-ACA amount in the blended benchmark
    k = out.year - 2011
    out["w_preaca"] = np.clip(1 - k / out.trans_years, 0, 1).where(out.trans_years.notna(), 0.0)
    blend = out.w_preaca * out.preaca + (1 - out.w_preaca) * out.A_formula * out.ffs
    out["rate0_formula"] = np.minimum(blend, out.preaca)
    out["capped"] = (blend >= out.preaca - 0.005).astype(int)     # CMS flags ties as capped
    return out


def schedule(q, group, score: str = "rank", cuts: dict | None = None):
    """Dose from the score: mean of this year's quartile percentage and last year's.

    ``group`` labels are "<year>|<prev_pct>" ("<year>|none" when there is no prior year).
    For ``score="rank"`` the cutoffs are 25/50/75 in every year (exact: see README); for
    ``score="logratio"`` they are 100 log(c/c75) for the year's cutoffs ``cuts[year]``.
    Ties go down: a county at a cutoff gets the higher percentage.
    """
    q = np.asarray(q, float)
    group = np.asarray(group).astype(str)
    out = np.full(len(q), np.nan)
    for lab in np.unique(group):
        m = group == lab
        year, prev = lab.split("|")
        if score == "rank":
            c = np.array([25.0, 50.0, 75.0])
        else:
            c25, c50, c75 = cuts[int(year)]
            c = 100 * np.log(np.array([c25, c50, c75]) / c75)
        k = (q[m, None] > c[None, :] + 1e-9).sum(axis=1)
        p = PCT[k]
        out[m] = p if prev == "none" else (p + float(prev)) / 2
    return out


# --------------------------------------------------------------------------- enrolment, Census
def read_penetration(year: int) -> pd.DataFrame:
    d = _extract(RAW / "penetration" / f"pen_{year}_07.zip")
    f = [p for p in d.rglob("*.csv")][0]
    df = pd.read_csv(f, dtype=str, encoding="latin-1")
    df = df[df["SSA"].fillna("").str.fullmatch(r"\d{5}")].copy()
    out = pd.DataFrame({"ssa": df["SSA"], "fips": df["FIPS"].str.zfill(5)})
    out["eligibles"] = df["Eligibles"].map(_num).to_numpy()
    enr = df["Enrolled"].fillna("").str.strip()
    out["masked"] = enr.eq("*").astype(int).to_numpy()
    out["enrolled"] = np.where(out.masked == 1, 5.0, enr.map(_num).to_numpy())   # "*" = 1-10
    out["pen"] = 100 * out.enrolled / out.eligibles
    out["year"] = year
    return out.drop_duplicates("ssa")


def read_census() -> pd.DataFrame:
    """County x calendar year: population, 65+/85+ shares, median age, income, poverty, density."""
    C = RAW / "census"
    frames = []
    for f in sorted(C.glob("cc-est2019-agesex-*.csv")):
        d = pd.read_csv(f, dtype={"STATE": str, "COUNTY": str}, encoding="latin-1")
        d = d[d.YEAR.between(3, 12)].assign(cal=lambda x: x.YEAR + 2007)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"{C} has no Census files; run download.py")
    d = pd.read_csv(C / "cc-est2025-agesex-all.csv", dtype={"STATE": str, "COUNTY": str},
                    encoding="latin-1")
    frames.append(d[d.YEAR.between(2, 7)].assign(cal=lambda x: x.YEAR + 2018))
    pop = pd.concat(frames)
    pop = pop[pop.SUMLEV.astype(int) == 50]
    pop = pd.DataFrame({"fips": pop.STATE.str.zfill(2) + pop.COUNTY.str.zfill(3), "cal": pop.cal,
                        "pop": pop.POPESTIMATE.astype(float),
                        "share65": pop.AGE65PLUS_TOT / pop.POPESTIMATE,
                        "share85": pop.AGE85PLUS_TOT / pop.POPESTIMATE,
                        "median_age": pop.MEDIAN_AGE_TOT.astype(float)})
    sa = []
    for f in sorted(C.glob("est*all.txt")):
        yr = 2000 + int(f.name[3:5])
        for line in f.read_text(encoding="latin-1").splitlines():
            if len(line) < 139 or line[3:6].strip() in ("", "0"):
                continue
            sa.append((line[0:2] + line[3:6].strip().zfill(3), yr, _num(line[34:38]), _num(line[133:139])))
    sa = pd.DataFrame(sa, columns=["fips", "cal", "poverty", "income"])
    with zipfile.ZipFile(C / "2020_Gaz_counties_national.zip") as z:
        gz = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), sep="\t", dtype={"GEOID": str})
    gz.columns = [c.strip() for c in gz.columns]
    area = gz.set_index("GEOID")["ALAND_SQMI"].astype(float)
    return pop, sa, area


def _latest(df: pd.DataFrame, keys: pd.DataFrame, max_age: int = 1) -> pd.DataFrame:
    """For each (fips, target) the row with the latest ``cal`` in [target - max_age, target]."""
    df = df.assign(cal=df.cal.astype("int64")).sort_values("cal")
    k = keys.assign(cal=keys.target.astype("int64")).sort_values("cal")
    return pd.merge_asof(k, df, on="cal", by="fips", direction="backward", tolerance=max_age)


# --------------------------------------------------------------------------- panel
def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not force:
        p = pd.read_csv(PANEL, dtype={"ssa": str, "fips": str})
        if p.get("panel_version", pd.Series([0])).iloc[0] == PANEL_VERSION:
            return p
    rb = ratebook_panel()
    pen = pd.concat([read_penetration(y) for y in PEN_YEARS], ignore_index=True)
    xw = pen.sort_values("year").drop_duplicates("ssa", keep="last").set_index("ssa").fips
    risk = pd.concat([read_ffs_risk_score(y).to_frame().assign(year=y).rename_axis("ssa").reset_index()
                      for y in RATE_YEARS], ignore_index=True)
    df = rb[(rb.territory == 0)].copy()
    df["fips"] = df.ssa.map(xw)
    df = df.merge(risk, on=["ssa", "year"], how="left")
    p = pen.set_index(["ssa", "year"])
    for lag, tag in [(0, ""), (1, "_lag"), (-1, "_lead")]:
        idx = pd.MultiIndex.from_arrays([df.ssa, df.year - lag])
        for c in ("pen", "eligibles", "enrolled", "masked"):
            df[c + tag] = p[c].reindex(idx).to_numpy()
    # Census covariates of calendar year t-1 (the latest within one year of it)
    pop, sa, area = read_census()
    df["target"] = df.year - 1
    keys = df[["fips", "target"]].dropna().drop_duplicates()
    cp = _latest(pop.assign(pop_year=pop.cal), keys)
    cs = _latest(sa.dropna(subset=["income"]).assign(saipe_year=lambda x: x.cal), keys)
    df = (df.merge(cp[["fips", "target", "pop", "share65", "share85", "median_age", "pop_year"]],
                   on=["fips", "target"], how="left")
            .merge(cs[["fips", "target", "poverty", "income", "saipe_year"]],
                   on=["fips", "target"], how="left")
            .drop(columns=["target"]))
    df["land_sqmi"] = df.fips.map(area)
    qr = rb.set_index(["ssa", "year"]).Q_rank
    df["Q_rank_lag"] = qr.reindex(pd.MultiIndex.from_arrays([df.ssa, df.year - 1])).to_numpy()
    df["panel_version"] = PANEL_VERSION
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PANEL, index=False)
    return df


def _design(df: pd.DataFrame, lag_score: bool = False):
    df = df.copy()
    df["log_elig_lag"] = np.log(df.eligibles_lag)
    df["log_pop"] = np.log(df["pop"])
    df["log_income"] = np.log(df.income)
    df["log_density"] = np.log(df["pop"] / df.land_sqmi)
    years = sorted(df.year.unique())
    states = sorted(df.state.unique())
    yd = pd.get_dummies(df.year, prefix="y").astype(float).reindex(columns=[f"y_{y}" for y in years[1:]], fill_value=0)
    sd = pd.get_dummies(df.state, prefix="st").astype(float).reindex(columns=[f"st_{s}" for s in states[1:]], fill_value=0)
    cont = CONT + (["Q_rank_lag"] if lag_score else [])
    X = pd.concat([df[cont + ["urban_floor_2004"]].astype(float), yd, sd], axis=1)
    return df, X


def load_dose(years=(2017, 2026), score: str = "rank", outcome: str = "pen",
              lag_score: bool = False) -> RDDSample:
    """County x payment-year sample for the dose design (see module docstring)."""
    if not (RAW / "ratebooks").exists():
        raise FileNotFoundError(f"{RAW}/ratebooks missing. See README for download instructions.")
    df = build_panel()
    df = df[df.year.between(*years)]
    ycol = {"pen": "pen", "dpen": None, "pen_lead": "pen_lead"}[outcome]
    df = df.assign(dpen=df.pen - df.pen_lag)
    ycol = ycol or "dpen"
    need = ["pen_lag", "eligibles_lag", "pop", "share65", "share85", "median_age", "income",
            "poverty", "land_sqmi", "ffs_risk", "Q_rank", ycol]
    df = df.dropna(subset=need).reset_index(drop=True)
    df, X = _design(df, lag_score)
    cuts = {int(y): tuple(g[["cut25", "cut50", "cut75"]].iloc[0]) for y, g in df.groupby("year")}
    if score == "rank":
        Q = df.Q_rank.to_numpy(float)
        thr = 75.0
    elif score == "logratio":
        Q = (100 * np.log(df.rank_rate / df.cut75)).to_numpy(float)
        thr = 0.0
    else:
        raise ValueError(score)
    group = np.array([f"{y}|{'none' if pd.isna(p) else repr(round(float(p), 6))}"
                      for y, p in zip(df.year, df.prev_pct)])
    sched = partial(schedule, score=score, cuts=cuts)
    A = df.A_formula.to_numpy(float)
    extras = {
        "A": A, "a_ref": A_REF, "dose_levels": LEVELS.copy(), "schedule": sched, "group": group,
        "Z_names": list(Z_NAMES), "unit_id": df.ssa.to_numpy(), "fips": df.fips.to_numpy(),
        "year": df.year.to_numpy(), "prev_pct": df.prev_pct.to_numpy(float),
        "pure_pct": df.pure_pct.to_numpy(float), "published_dose": df.qpct.to_numpy(float),
        "ffs_rank_rate": df.rank_rate.to_numpy(float), "q_rank": df.Q_rank.to_numpy(float),
        "q_logratio": (100 * np.log(df.rank_rate / df.cut75)).to_numpy(float), "cutoffs": cuts,
        "ffs": df.ffs.to_numpy(float), "preaca": df.preaca.to_numpy(float),
        "benchmark0": df.rate0.to_numpy(float), "benchmark_ratio": (df.rate0 / df.ffs).to_numpy(float),
        "capped": df.capped.to_numpy(int), "aca_share": (1 - df.w_preaca).to_numpy(float),
        "qualifying": df.qualifying.to_numpy(int), "trans_years": df.trans_years.to_numpy(float),
        "eligibles": df.eligibles.to_numpy(float), "y_masked": df.masked.to_numpy(float),
        "pen": df.pen.to_numpy(float), "pen_lag": df.pen_lag.to_numpy(float),
        "pen_lead": df.pen_lead.to_numpy(float), "dpen": df.dpen.to_numpy(float),
        "score": score, "outcome": outcome,
    }
    return RDDSample(
        Q=Q, X=X.to_numpy(float), Y=df[ycol].to_numpy(float), threshold=thr,
        name="medicare_ma_benchmarks", feature_names=list(X.columns),
        description=("MA county benchmarks: applicable percentage (95/100/107.5/115%, averaged with "
                     "last year's) by quartile of the prior-year FFS rate; Y = MA penetration"),
        citation="CMS MA Ratebooks & Supporting Data (rate calculation data), SSA 1853(n)",
        treatment_rule=lambda q, t=thr: (np.asarray(q) <= t + 1e-9).astype(int),
        extras=extras,
    )


def load() -> RDDSample:
    """Registry entry: the dose sample; D = 1{dose above the 95% reference region} = 1{Q <= 75}."""
    return load_dose()


# --------------------------------------------------------------------------- checks
def rule_check() -> pd.DataFrame:
    """Formula vs published applicable percentage and 0%-bonus county rate, by payment year."""
    rb = ratebook_panel()
    rows = []
    for y, g in rb.groupby("year"):
        s = g[g.territory == 0]
        t = g[g.territory == 1]
        pub = s.qpct.notna()
        rows.append({
            "year": y, "n_states_dc": len(s), "n_ranked": int(s.n_ranked.iloc[0]),
            "cut25": s.cut25.iloc[0], "cut50": s.cut50.iloc[0], "cut75": s.cut75.iloc[0],
            "A_match": float((np.abs(s.A_formula - s.qpct) < 1e-6)[pub].mean()),
            "A_mismatch": int((~(np.abs(s.A_formula - s.qpct) < 1e-6))[pub].sum()),
            "A_match_territories": float((np.abs(t.A_formula - t.qpct) < 1e-6).mean()) if len(t) else np.nan,
            "rate0_match": float((np.abs(s.rate0_formula - s.rate0) <= 0.011)[s.rate0.notna()].mean()),
            "capped_share": float(s.capped.mean()),
            "capped_flag_agree": float((s.capped == s.capped0_pub).mean()) if s.capped0_pub.any() else np.nan,
            "schedule_match": float((np.abs(schedule(s.Q_rank, [f"{y}|{'none' if pd.isna(p) else repr(round(p, 6))}" for p in s.prev_pct]) - s.A_formula) < 1e-9)[s.Q_rank.notna()].mean()),
        })
    out = pd.DataFrame(rows)
    # CMS's own 2016 file of pure quartile percentages (2015 and 2016) and the blend
    f = list(_extract(RAW / "ratebooks" / "applicablepercentagescy2016.zip").glob("*.csv"))
    if f:
        q = pd.read_csv(f[0], dtype=str, encoding="latin-1")
        q.columns = [c.strip() for c in q.columns]
        q["ssa"] = q.iloc[:, 0].str.strip().str.zfill(5)
        m = rb[rb.year == 2016].merge(q, on="ssa")
        m = m[m.territory == 0]
        out.attrs["pure_2016_file"] = {
            "n": len(m),
            "pure2015_match": float((np.abs(m.prev_pct - m["2015 Quartile"].map(_num)) < 1e-6).mean()),
            "pure2016_match": float((np.abs(m.pure_pct - m["2016 Quartile"].map(_num)) < 1e-6).mean()),
            "blend_match": float((np.abs(m.A_formula - m["2016 Applicable Percentage"].map(_num)) < 1e-6).mean()),
        }
    return out


def readiness(s: RDDSample | None = None) -> dict:
    """R^2 of Q on X, sd(eta), share at a_ref, eta quantiles by dose level, bunching of FFS."""
    s = s or load_dose()
    Xt = np.column_stack([np.ones(s.n), s.X])
    gam = np.linalg.lstsq(Xt, s.Q, rcond=None)[0]
    T = Xt @ gam
    eta = s.Q - T
    A = s.extras["A"]
    by = []
    for a in s.extras["dose_levels"]:
        m = np.abs(A - a) < 1e-9
        if m.sum():
            by.append((float(a), int(m.sum()), *np.round(np.quantile(eta[m], [0.1, 0.5, 0.9]), 2)))
    # bunching of the ranking FFS rate within +-1% of each cutoff vs placebo quantiles
    rb = ratebook_panel()
    rb = rb[(rb.territory == 0) & rb.rank_rate.notna()]

    def ratio(probs, h=0.01):
        up = dn = 0
        for y, g in rb.groupby("year"):
            v = g.rank_rate.to_numpy()
            for c in np.quantile(v, probs):
                r = np.log(v / c)
                dn += int(((r > -h) & (r <= 0)).sum())
                up += int(((r > 0) & (r <= h)).sum())
        return up, dn, up / dn
    return {"n": s.n, "units": len(np.unique(s.extras["unit_id"])), "p": s.p,
            "R2": float(np.var(T) / np.var(s.Q)), "sd_Q": float(np.std(s.Q)), "sd_T": float(np.std(T)),
            "sd_eta": float(np.std(eta)), "share_ref": float(np.mean(np.abs(A - A_REF) < 1e-9)),
            "share_D": float(s.D.mean()), "by_dose": by,
            "bunching_cutoffs": ratio([0.25, 0.5, 0.75]),
            "bunching_placebo": ratio([0.125, 0.375, 0.625, 0.875])}
