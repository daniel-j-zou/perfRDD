"""Chile SEP concentration subsidy: a step schedule (dose) on the school's priority share.

Ley 20.248 (Subvencion Escolar Preferencial, SEP), Art. 16, pays every school with a SEP
agreement a "subvencion por concentracion de alumnos prioritarios": a monthly amount per
enrolled student (all students of the incorporated levels, not only priority students),
in USE (unidad de subvencion educacional), that rises in four steps with the school's
share of priority students. Art. 16 defines the share as the average enrolment of
priority students in the SEP levels over the average enrolment in those levels, both
March-December of the previous year.

Unit: school (RBD) x score year s. The score is measured in year s and sets the dose in
the payment year t = s + 1.

Q   priority share in year s, in percent (see README for the public approximation).
A   the band-A (pre-kindergarten to 4th grade) concentration rate, in USE per student
    per month, from the Art. 16 table in force in year t (``schedule``).
Y   year-t school mean SIMCE 4th-grade score, (reading + math) / 2 (``load_dose``).
X   year-s covariates (dependency, rural, region, log enrolment, girls' share, lagged
    SIMCE, lagged promotion and attendance, SIMCE socioeconomic group, year dummies).

Data are read from ``data/raw`` (see download.py) and, for the student-level performance
files, in place from ``../chile_retention/data/raw/<year>/``. School-level aggregates are
cached in ``data/processed/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
REND_RAW = HERE.parent / "chile_retention" / "data" / "raw"

# ---------------------------------------------------------------------------
# The law: Art. 16 table, in USE per enrolled student per month.
# Versions (LeyChile XML, data/raw/law/): original text 2008-02-01; Ley 20.501 table in
# force from 2011-03-01; current table (image named LEY20637) from 2012-10-26.
# Columns: grade band A = 1st transition level (pre-K) to 4th basic, B = 5th-6th basic,
# C = 7th-8th basic, D = 1st-4th secondary (not covered before 2011).
# Rows: share in [15, 30), [30, 45), [45, 60), [60, 100]. Below 15%: 0.
# ---------------------------------------------------------------------------
CUTS = (15.0, 30.0, 45.0, 60.0)
TABLES = {
    "2008": {"A": (0.098, 0.168, 0.224, 0.252), "B": (0.065, 0.112, 0.149, 0.168),
             "C": (0.033, 0.056, 0.075, 0.084), "D": (0.0, 0.0, 0.0, 0.0)},
    "2011": {"A": (0.118, 0.202, 0.269, 0.302), "B": (0.078, 0.134, 0.179, 0.202),
             "C": (0.040, 0.067, 0.090, 0.101), "D": (0.040, 0.067, 0.090, 0.101)},
    "2013": {"A": (0.118, 0.202, 0.269, 0.302), "B": (0.118, 0.202, 0.269, 0.302),
             "C": (0.078, 0.134, 0.179, 0.202), "D": (0.078, 0.134, 0.179, 0.202)},
}


def period(t) -> np.ndarray:
    """Table in force in payment year t: '2008' (2008-2010), '2011' (2011-2012),
    '2013' (2013 on; the 2012-10-26 table, taken to apply from the next school year).
    Band A is the same in '2011' and '2013'."""
    t = np.asarray(t)
    return np.where(t <= 2010, "2008", np.where(t <= 2012, "2011", "2013"))


def tier(q) -> np.ndarray:
    """Tier 0-4 of a share in percent: 0 below 15, 1 in [15,30), ..., 4 at 60 or more."""
    return np.searchsorted(np.asarray(CUTS), np.asarray(q, float), side="right")


def schedule(q, group, band: str = "A") -> np.ndarray:
    """Dose (USE per student per month) for share q (percent) and table group.

    `group` is a payment year, a table key ('2008' / '2011' / '2013'), or a band-A
    group label of the sample ('2008-2010' / '2011+'; band A is the same in 2011-2012
    and from 2013, so '2011+' is valid for band A only)."""
    q = np.atleast_1d(np.asarray(q, float))
    g = np.atleast_1d(np.asarray(group))
    if np.issubdtype(g.dtype, np.number):
        g = period(g)
    else:
        g = np.array([{"2008-2010": "2008", "2011+": "2011"}.get(str(x), str(x)) for x in g])
    g = np.broadcast_to(g, q.shape)
    k = tier(q)
    out = np.zeros(q.shape)
    for p, tab in TABLES.items():
        rates = np.r_[0.0, tab[band]]
        m = g == p
        out[m] = rates[k[m]]
    return out


# ---------------------------------------------------------------------------
# Student-level performance files (chile_retention), aggregated to school x year.
# ---------------------------------------------------------------------------
YOUTH_SECONDARY = (310, 410, 510, 610, 710, 810, 910)
REND_COLS = {"RBD", "COD_REG_RBD", "COD_COM_RBD", "COD_DEPE", "COD_DEPE2", "RURAL_RBD",
             "COD_ENSE", "COD_GRADO", "MRUN", "GEN_ALU", "ASISTENCIA", "SIT_FIN",
             "PROM_GRAL", "ESTADO_ESTAB"}


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a cache atomically (a killed run must not leave a truncated file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def _norm(col: str) -> str:
    return col.replace("﻿", "").replace("ï»¿", "").strip().upper()


def _rend_file(year: int) -> Path:
    files = sorted((REND_RAW / str(year)).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No performance CSV for {year} under {REND_RAW}. Run "
            "`python -m experiments.datasets.chile_retention.download {year}`.")
    return files[0]


def read_rend_students(year: int, chunksize: int = 1_000_000):
    """Yield chunks of the year's student records: normalized headers, numeric codes,
    a level band ('b14', 'b58', 'med', 'oth') and SIT_FIN stripped."""
    f = _rend_file(year)
    reader = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, chunksize=chunksize,
                         usecols=lambda c: _norm(c) in REND_COLS)
    for d in reader:
        d.columns = [_norm(c) for c in d.columns]
        d = d.reindex(columns=sorted(REND_COLS))
        d["SIT_FIN"] = d.SIT_FIN.fillna("").str.strip()
        for c in ("PROM_GRAL", "ASISTENCIA"):
            d[c] = pd.to_numeric(d[c].str.replace(",", ".", regex=False), errors="coerce")
        for c in REND_COLS - {"PROM_GRAL", "ASISTENCIA", "SIT_FIN"}:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        basic = d.COD_ENSE == 110
        d["band"] = np.select(
            [basic & d.COD_GRADO.between(1, 4), basic & d.COD_GRADO.between(5, 8),
             d.COD_ENSE.isin(YOUTH_SECONDARY) & d.COD_GRADO.between(1, 4)],
            ["b14", "b58", "med"], "oth")
        yield d


def rend_school(year: int) -> pd.DataFrame:
    """School-level aggregates of the year's performance file (cached).

    n_<band>      student records in the band (all final statuses, including students
                  who left during the year; a transferred student counts at both schools)
    n_reg         n_b14 + n_b58 + n_med
    female_reg    share of girls among n_reg
    prom_rate_*   promoted / (promoted + retained), basic grades 1-8 and 1-4
    att_*         mean attendance (%) of promoted or retained students, same groups
    gpa_b18       mean annual grade average (1-7) of promoted or retained, grades 1-8
    cod_depe, cod_depe2, rural, region, commune: first non-missing value (constant
                  within a school-year)
    """
    cache = PROCESSED / "rend_school" / f"rend_school_{year}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    for d in read_rend_students(year):
        pr = d.SIT_FIN.isin(["P", "R"])
        b18 = d.band.isin(["b14", "b58"])
        g = pd.DataFrame({
            "RBD": d.RBD,
            "n_b14": (d.band == "b14").astype(int), "n_b58": (d.band == "b58").astype(int),
            "n_med": (d.band == "med").astype(int), "n_oth": (d.band == "oth").astype(int),
            "fem_reg": ((d.band != "oth") & (d.GEN_ALU == 2)).astype(int),
            "pr_b18": (pr & b18).astype(int), "p_b18": (b18 & (d.SIT_FIN == "P")).astype(int),
            "pr_b14": (pr & (d.band == "b14")).astype(int),
            "p_b14": ((d.band == "b14") & (d.SIT_FIN == "P")).astype(int),
            "att_b18_sum": d.ASISTENCIA.where(pr & b18, 0).fillna(0),
            "att_b18_n": (pr & b18 & d.ASISTENCIA.notna()).astype(int),
            "att_b14_sum": d.ASISTENCIA.where(pr & (d.band == "b14"), 0).fillna(0),
            "att_b14_n": (pr & (d.band == "b14") & d.ASISTENCIA.notna()).astype(int),
            "gpa_b18_sum": d.PROM_GRAL.where(pr & b18 & (d.PROM_GRAL > 0), 0).fillna(0),
            "gpa_b18_n": (pr & b18 & (d.PROM_GRAL > 0)).astype(int),
        })
        parts.append(g.groupby("RBD").sum())
        attrs = d[["RBD", "COD_DEPE", "COD_DEPE2", "RURAL_RBD", "COD_REG_RBD",
                   "COD_COM_RBD", "ESTADO_ESTAB"]]
        # School attributes are constant within a year's file; take the first non-null.
        parts.append(attrs.groupby("RBD").first().add_prefix("attr_"))
    sums = pd.concat([p for p in parts if "n_b14" in p.columns]).groupby(level=0).sum()
    attrs = pd.concat([p for p in parts if "attr_COD_DEPE" in p.columns]).groupby(level=0).first()
    s = sums.join(attrs)
    out = pd.DataFrame({
        "year": year, "rbd": s.index.astype(int),
        "n_b14": s.n_b14, "n_b58": s.n_b58, "n_med": s.n_med, "n_oth": s.n_oth,
        "n_reg": s.n_b14 + s.n_b58 + s.n_med,
        "female_reg": s.fem_reg / (s.n_b14 + s.n_b58 + s.n_med).replace(0, np.nan),
        "prom_rate_b18": s.p_b18 / s.pr_b18.replace(0, np.nan),
        "prom_rate_b14": s.p_b14 / s.pr_b14.replace(0, np.nan),
        "att_b18": s.att_b18_sum / s.att_b18_n.replace(0, np.nan),
        "att_b14": s.att_b14_sum / s.att_b14_n.replace(0, np.nan),
        "gpa_b18": s.gpa_b18_sum / s.gpa_b18_n.replace(0, np.nan),
        "n_pr_b14": s.pr_b14, "n_pr_b18": s.pr_b18,
        "cod_depe": s.attr_COD_DEPE, "cod_depe2": s.attr_COD_DEPE2,
        "rural": s.attr_RURAL_RBD, "region": s.attr_COD_REG_RBD,
        "commune": s.attr_COD_COM_RBD, "estado": s.attr_ESTADO_ESTAB,
    }).reset_index(drop=True)
    # Grouped dependency before COD_DEPE2 exists: 1 municipal (corporation or DAEM),
    # 2 private voucher, 3 private paid, 4 delegated administration; 5 SLEP (COD_DEPE 6).
    fill = out.cod_depe.map({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5})
    out["cod_depe2"] = out.cod_depe2.fillna(fill)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write(out, cache)
    return out


# ---------------------------------------------------------------------------
# SIMCE school-level results (Agencia de Calidad de la Educacion, public bases).
# ---------------------------------------------------------------------------
GSE = {"bajo": 1, "medio bajo": 2, "medio": 3, "medio alto": 4, "alto": 5}


def _simce_file(grade: int, year: int) -> Path | None:
    root = RAW / f"simce{grade}" / f"simce{grade}_{year}"
    hits = [p for p in root.rglob("*") if p.is_file() and "rbd" in p.name.lower()
            and p.suffix.lower() in (".csv", ".txt")]
    return sorted(hits)[0] if hits else None


def read_simce(grade: int, year: int) -> pd.DataFrame | None:
    """School means (reading, math) and counts for one SIMCE grade and year."""
    f = _simce_file(grade, year)
    if f is None:
        return None
    head = f.read_bytes()[:4000].decode("latin-1").splitlines()[0]
    sep = max(("|", ";", "\t", ","), key=head.count)
    d = pd.read_csv(f, sep=sep, encoding="latin-1", dtype=str)
    d.columns = [_norm(c).lower() for c in d.columns]
    g = f"{grade}b"
    num = lambda c: pd.to_numeric(d[c].str.replace(",", ".", regex=False), errors="coerce") \
        if c in d.columns else pd.Series(np.nan, index=d.index)
    grp = d["cod_grupo"].str.strip().str.lower() if "cod_grupo" in d.columns else None
    gse = pd.to_numeric(grp, errors="coerce").fillna(grp.map(GSE)) if grp is not None else np.nan
    out = pd.DataFrame({
        "year": year, "rbd": pd.to_numeric(d.rbd, errors="coerce"),
        f"simce{grade}_lect": num(f"prom_lect{g}_rbd"),
        f"simce{grade}_mate": num(f"prom_mate{g}_rbd"),
        f"simce{grade}_nlect": num(f"nalu_lect{g}_rbd"),
        f"simce{grade}_nmate": num(f"nalu_mate{g}_rbd"),
        f"gse{grade}": gse,
    }).dropna(subset=["rbd"])
    # Suppressed results (too few students) are blank in some years and 0 in others.
    for c in (f"simce{grade}_lect", f"simce{grade}_mate"):
        out.loc[out[c] <= 0, c] = np.nan
    out["rbd"] = out.rbd.astype(int)
    out[f"simce{grade}"] = out[[f"simce{grade}_lect", f"simce{grade}_mate"]].mean(axis=1, skipna=False)
    return out.drop_duplicates("rbd")


def simce_panel(grade: int = 4) -> pd.DataFrame:
    cache = PROCESSED / f"simce{grade}_panel.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    frames = [read_simce(grade, y) for y in range(2005, 2026)]
    out = pd.concat([f for f in frames if f is not None], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    _write(out, cache)
    return out


# ---------------------------------------------------------------------------
# MINEDUC SEP files (priority students, SEP agreement), aggregated to school x year.
# Column names vary across years; they are matched by prefix after normalizing.
# ---------------------------------------------------------------------------
def _find(cols, *prefixes):
    for p in prefixes:
        for c in cols:
            if c.startswith(p):
                return c
    return None


def _flag(s: pd.Series) -> pd.Series:
    """0/1 from the files' codings (1/0, 'SI'/'NO', 'S'/'N', '1'/'0')."""
    v = s.astype(str).str.strip().str.upper()
    return v.isin(["1", "SI", "SÍ", "S", "1.0", "TRUE"]).astype(int)


def _sep_file(year: int) -> Path:
    root = RAW / "sep_students" / f"sep_students_{year}"
    hits = sorted(p for p in root.rglob("*") if p.is_file()
                  and p.suffix.lower() in (".csv", ".txt"))
    if not hits:
        raise FileNotFoundError(
            f"No SEP student file for {year} under {root}. Run "
            "`python -m experiments.datasets.chile_sep.download --only sep_students`.")
    return max(hits, key=lambda p: p.stat().st_size)


def sep_school(year: int) -> pd.DataFrame:
    """School-level counts from the year's SEP student file (cached).

    n_rows_<band>   student rows in the file by level band (b14, b58, med, parv, oth)
    n_prio_<band>   rows flagged priority (PRIORITARIO_ALU)
    n_pref, n_ben   preferential students; SEP beneficiaries (BEN_SEP)
    convenio        1 if any row of the school has CONVENIO_SEP = 1
    ingreso         first year in SEP (AÑO_INGRESO_SEP), if present
    clasif          SEP classification (autonomo / emergente / en recuperacion), if present
    """
    cache = PROCESSED / "sep_school" / f"sep_school_{year}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    f = _sep_file(year)
    head = f.read_bytes()[:20000].decode("latin-1").splitlines()[0]
    sep = max((";", "|", "\t", ","), key=head.count)
    parts = []
    for d in pd.read_csv(f, sep=sep, encoding="latin-1", dtype=str, chunksize=1_000_000):
        d.columns = [_norm(c) for c in d.columns]
        cols = list(d.columns)
        c_rbd, c_ense, c_grado = _find(cols, "RBD"), _find(cols, "COD_ENSE"), _find(cols, "COD_GRADO")
        c_prio = _find(cols, "PRIORITARIO")
        c_pref = _find(cols, "PREFERENTE")
        c_ben = _find(cols, "BEN_SEP", "BENEFICIARIO")
        c_conv = _find(cols, "CONVENIO_SEP", "CONVENIO")
        c_ing = _find(cols, "AÑO_INGRESO_SEP", "ANO_INGRESO_SEP", "AGNO_INGRESO_SEP",
                      "AÑO_INGRESO", "INGRESO_SEP")
        c_cla = _find(cols, "CLASIFICACION_SEP", "CLASIFICACION")
        ense = pd.to_numeric(d[c_ense], errors="coerce") if c_ense else pd.Series(np.nan, index=d.index)
        grado = pd.to_numeric(d[c_grado], errors="coerce") if c_grado else pd.Series(np.nan, index=d.index)
        band = np.select(
            [(ense == 110) & grado.between(1, 4), (ense == 110) & grado.between(5, 8),
             ense.isin(YOUTH_SECONDARY) & grado.between(1, 4), ense == 10],
            ["b14", "b58", "med", "parv"], "oth")
        prio = _flag(d[c_prio]) if c_prio else pd.Series(1, index=d.index)
        g = pd.DataFrame({"RBD": pd.to_numeric(d[c_rbd], errors="coerce")})
        for b in ("b14", "b58", "med", "parv", "oth"):
            g[f"n_rows_{b}"] = (band == b).astype(int)
            g[f"n_prio_{b}"] = ((band == b) & (prio == 1)).astype(int)
        g["n_pref"] = _flag(d[c_pref]) if c_pref else 0
        g["n_ben"] = _flag(d[c_ben]) if c_ben else 0
        g["convenio"] = _flag(d[c_conv]) if c_conv else np.nan
        g["ingreso"] = pd.to_numeric(d[c_ing], errors="coerce") if c_ing else np.nan
        g["clasif"] = d[c_cla].str.strip().str.upper() if c_cla else ""
        agg = {c: "sum" for c in g.columns if c.startswith(("n_rows", "n_prio", "n_pref", "n_ben"))}
        agg.update({"convenio": "max", "ingreso": "min", "clasif": "first"})
        parts.append(g.groupby("RBD").agg(agg))
    out = pd.concat(parts).groupby(level=0).agg(
        {c: ("sum" if c.startswith("n_") else {"convenio": "max", "ingreso": "min",
                                                 "clasif": "first"}[c])
         for c in parts[0].columns})
    out = out.reset_index().rename(columns={"RBD": "rbd"})
    out.insert(0, "year", year)
    out["rbd"] = out.rbd.astype(int)
    _write(out, cache)
    return out


# ---------------------------------------------------------------------------
# Panel: school x score year s (dose and outcomes in t = s + 1).
# ---------------------------------------------------------------------------
FIRST_SCORE_YEAR, LAST_SCORE_YEAR = 2008, 2024
GROUP_LABEL = {"2008": "2008-2010", "2011": "2011+", "2013": "2011+"}


def share(sep: pd.DataFrame, rend: pd.DataFrame, levels=("b14", "b58", "med")) -> pd.Series:
    """Priority share (percent) in the given level bands: priority rows in the SEP file
    over student records in the performance file, by school (aligned to `rend`)."""
    s = sep.set_index("rbd")
    r = rend.set_index("rbd")
    num = sum(s[f"n_prio_{b}"] for b in levels).reindex(r.index)
    den = sum(r[f"n_{b}"] for b in levels)
    return 100 * num / den.replace(0, np.nan)


def build_panel() -> pd.DataFrame:
    cache = PROCESSED / "sep_panel.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    s4 = simce_panel(4).set_index(["year", "rbd"])
    s8 = simce_panel(8).set_index(["year", "rbd"])
    rows = []
    for s in range(FIRST_SCORE_YEAR, LAST_SCORE_YEAR + 1):
        t = s + 1
        rs, rt = rend_school(s).set_index("rbd"), rend_school(t).set_index("rbd")
        ss, st = sep_school(s).set_index("rbd"), sep_school(t).set_index("rbd")
        d = rs.copy()
        d["score_year"], d["year"] = s, t
        d["Q"] = share(ss.reset_index(), rs.reset_index())
        d["Q_b14"] = share(ss.reset_index(), rs.reset_index(), ("b14",))
        d["Q_b18"] = share(ss.reset_index(), rs.reset_index(), ("b14", "b58"))
        d["n_prio_reg"] = ss.reindex(d.index)[["n_prio_b14", "n_prio_b58", "n_prio_med"]].sum(axis=1)
        d["n_prio_all"] = ss.reindex(d.index)[[c for c in ss.columns if c.startswith("n_prio_")]].sum(axis=1)
        d["convenio_s"] = ss.convenio.reindex(d.index)
        d["convenio_t"] = st.convenio.reindex(d.index)
        d["ingreso"] = st.ingreso.reindex(d.index)
        d["clasif_t"] = st.clasif.reindex(d.index)
        d["n_ben_t"] = st.n_ben.reindex(d.index)
        d["n_pref_s"] = ss.n_pref.reindex(d.index)
        for k, v in (("lag_", s), ("", t)):
            for src, cols in ((s4, ["simce4", "simce4_lect", "simce4_mate", "simce4_nlect", "gse4"]),
                              (s8, ["simce8"])):
                blk = src.xs(v, level="year") if v in src.index.get_level_values(0) else None
                for c in cols:
                    d[k + c] = blk[c].reindex(d.index) if blk is not None else np.nan
        for c in ("prom_rate_b18", "prom_rate_b14", "att_b18", "att_b14", "gpa_b18", "n_reg"):
            d[f"next_{c}"] = rt[c].reindex(d.index)
        rows.append(d.reset_index())
    out = pd.concat(rows, ignore_index=True)
    _write(out, cache)
    return out


def sample_frame(outcome: str = "simce4") -> pd.DataFrame:
    """The analysis sample: SEP schools in the payment year t with a defined share.

    Kept: SEP agreement in t (convenio_t = 1); municipal, private-voucher or SLEP
    (Servicio Local) dependency in s; at least one regular student in s; the outcome
    observed in t."""
    p = build_panel()
    keep = (p.convenio_t == 1) & p.cod_depe2.isin([1, 2, 5]) & (p.n_reg > 0) & p.Q.notna()
    if outcome:
        keep &= p[outcome].notna()
    d = p[keep].copy()
    d["Q"] = d.Q.clip(upper=100.0)
    d["group"] = [GROUP_LABEL[x] for x in period(d.year)]
    d["A"] = schedule(d.Q.to_numpy(), d.year.to_numpy())
    d["tier"] = tier(d.Q.to_numpy())
    # Covariates (year s).
    d["private_voucher"] = (d.cod_depe2 == 2).astype(float)
    d["slep"] = (d.cod_depe2 == 5).astype(float)
    d["rural"] = d.rural.astype(float)
    d["log_enrol"] = np.log(d.n_reg)
    d["share_b14"] = d.n_b14 / d.n_reg
    d["has_secondary"] = (d.n_med > 0).astype(float)
    d["female_reg"] = d.female_reg.fillna(d.female_reg.mean())
    for c in ("lag_simce4", "lag_gse4", "prom_rate_b18", "att_b18", "gpa_b18"):
        d[f"{c}_missing"] = d[c].isna().astype(float)
        d[c] = d[c].fillna(d[c].mean())
    # Region: the school's most recent region code (codes changed in 2007 and 2018).
    reg = build_panel().sort_values("year").groupby("rbd").region.last()
    d["region_now"] = d.rbd.map(reg)
    return d


X_BASE = ["private_voucher", "slep", "rural", "log_enrol", "share_b14", "has_secondary",
          "female_reg", "lag_simce4", "lag_simce4_missing", "lag_gse4", "lag_gse4_missing",
          "prom_rate_b18", "prom_rate_b18_missing", "att_b18", "att_b18_missing", "gpa_b18",
          "gpa_b18_missing"]
Z_NAMES = ["rural", "private_voucher", "log_enrol", "lag_simce4", "female_reg", "lag_gse4"]


def _design(d: pd.DataFrame):
    reg = pd.get_dummies(d.region_now.astype(int), prefix="region", drop_first=True, dtype=float)
    yr = pd.get_dummies(d.year.astype(int), prefix="year", drop_first=True, dtype=float)
    X = pd.concat([d[X_BASE].astype(float), reg, yr], axis=1)
    return X.to_numpy(float), list(X.columns)


def _sample(d: pd.DataFrame, y: str, name: str, description: str) -> RDDSample:
    X, names = _design(d)
    levels = np.unique(np.r_[0.0, [v for tab in TABLES.values() for v in tab["A"]]])
    return RDDSample(
        Q=d.Q.to_numpy(float), X=X, Y=d[y].to_numpy(float), threshold=CUTS[0],
        name=name, feature_names=names, description=description,
        citation=("Ley 20.248 (2008), Art. 16, as amended by Ley 20.501 (2011); MINEDUC "
                  "Centro de Estudios open data (SEP students, performance by student); "
                  "Agencia de Calidad de la Educacion, SIMCE public bases; Feigenberg, "
                  "Yan and Rivkin (2019), Economic Journal / NBER w23178."),
        treatment_rule=lambda Q: (np.asarray(Q) >= CUTS[0]).astype(int),
        extras={
            "A": d.A.to_numpy(float), "a_ref": 0.0, "dose_levels": levels,
            "schedule": schedule, "group": d.group.to_numpy(), "cuts": CUTS,
            "Z_names": [z for z in Z_NAMES if z in names], "tier": d.tier.to_numpy(),
            "rbd": d.rbd.to_numpy(int), "year": d.year.to_numpy(int),
            "score_year": d.score_year.to_numpy(int),
            "Q_b14": d.Q_b14.to_numpy(float), "Q_b18": d.Q_b18.to_numpy(float),
            "n_reg": d.n_reg.to_numpy(float), "cod_depe2": d.cod_depe2.to_numpy(),
            "tables": TABLES,
        },
    )


def load_dose() -> RDDSample:
    """Y = year-t SIMCE 4th-grade school mean, (reading + math) / 2."""
    d = sample_frame("simce4")
    return _sample(d, "simce4", "chile_sep",
                   "Chile SEP concentration subsidy (Ley 20.248 Art. 16): band-A dose in "
                   "USE per student by the school's priority share; Y = next-year SIMCE 4.")


def load_dose_math() -> RDDSample:
    d = sample_frame("simce4_mate")
    return _sample(d, "simce4_mate", "chile_sep_math", "As load_dose, Y = SIMCE 4 math.")


def load_dose_reading() -> RDDSample:
    d = sample_frame("simce4_lect")
    return _sample(d, "simce4_lect", "chile_sep_reading", "As load_dose, Y = SIMCE 4 reading.")


def load_dose_promotion() -> RDDSample:
    """Y = year-t promotion rate, basic grades 1-8 (performance files)."""
    d = sample_frame("next_prom_rate_b18")
    return _sample(d, "next_prom_rate_b18", "chile_sep_promotion",
                   "As load_dose, Y = next-year promotion rate (grades 1-8).")


def load_dose_attendance() -> RDDSample:
    """Y = year-t mean attendance (%), basic grades 1-8 (performance files)."""
    d = sample_frame("next_att_b18")
    return _sample(d, "next_att_b18", "chile_sep_attendance",
                   "As load_dose, Y = next-year mean attendance (grades 1-8).")


def load() -> RDDSample:
    """Registry entry point: the dose sample (D = 1{Q >= 15}, any concentration dose)."""
    return load_dose()
