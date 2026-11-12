"""ICU titration protocols in MIMIC-IV as dose schedules (screen on the open demo).

A lab value (the score Q) sets a drug dose A by a written protocol; the next value of the
same lab is the outcome Y. Three protocols are built as event tables:

``potassium``  Q = serum potassium (labevents 50971, mEq/L). A = potassium chloride given
               (mEq) from the time the result is stored until the next draw or 6 h, whichever
               comes first: IV KCl in the ICU (inputevents 225166; additives to bags of
               >= 500 mL are maintenance fluid, not replacement), oral KCl in the ICU and
               every route on the wards (emar + emar_detail). Reference dose 0.
``heparin``    Q = aPTT (labevents 51275, s) drawn while a heparin infusion runs. A = change
               in the infusion rate (units/kg/h) between the result and the next draw or
               4 h; ICU rates from inputevents 225152 (01-Drips), ward rates from emar_detail
               (Heparin IV Infusion: infusion_rate, prior_infusion_rate). A hold (rate 0
               inside the window, then restarted) is flagged. Reference dose 0 (no change).
``insulin``    Q = finger-stick glucose in the ICU (chartevents 225664, 226537, mg/dL) while
               a sliding-scale insulin order is active (pharmacy.sliding_scale = "Y") and no
               insulin infusion runs. A = units of subcutaneous regular or lispro insulin
               (inputevents 223258 "06-Insulin (Non IV)", 223262) charted from 15 min before
               to 2 h after the stick (or until the next stick). Reference dose 0.
``insulin_emar`` the eMAR sliding-scale records (emar_detail administration_type "Insulin SC
               Sliding Scale"), which document zero doses ("Not Given per Sliding Scale"),
               each with the latest glucose up to 1 h before. Wards have no finger-stick
               glucose in MIMIC-IV, so these are mostly ICU records.

Coverage of doses. inputevents covers every ICU stay; emar covers only admissions after
the hospital adopted it (in the demo 181 of 275 admissions have any emar record). Oral
potassium and ward doses are therefore missing before that date; ``complete`` marks events
in admissions with emar records, and ``load_dose`` keeps only those by default.

The score depends on earlier doses (a low potassium is replaced, then rechecked). X holds
the previous value, the hours since it, and the dose in the 24 h before the draw, so eta
is the part of today's value not predicted from yesterday's value and dose.

Sources. ``source="demo"`` reads ``data/raw/demo_2.2`` (open, see download.py);
``source="full"`` reads the newest ``data/raw/mimiciv_<version>/`` (credentialed; the
author's steps are in README.md). Filtered tables are cached in
``data/processed/<folder>/``; the full labevents and chartevents are read in chunks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROC = HERE / "data" / "processed"
DEMO = "demo_2.2"

LAB = {"potassium": 50971, "ptt": 51275, "glucose": 50931, "glucose_bg": 50809, "creatinine": 50912}
GLU_CHART = [225664, 226537]                          # finger stick, whole-blood glucose (ICU)
WT_CHART = {226512: 1.0, 224639: 1.0, 226531: 0.45359237}   # admission kg, daily kg, admission lb
IN = {"kcl": 225166, "kcl_crrt": 227536, "heparin": 225152, "ins_regular": 223258,
      "ins_lispro": 223262, "ins_glargine": 223260, "ins_nph": 223259}
GIVEN = ("Administered", "Partial Administered", "Delayed Administered", "Administered in Other Location",
         "Confirmed", "Delayed Confirmed", "Confirmed in Other Location", "Started", "Restarted",
         "Started in Other Location")

PROTOCOLS = ("potassium", "heparin", "insulin", "insulin_emar")
# Fine score bins for the rule search, the dose rounding, and the direction of the rule
# (-1: dose falls as the score rises).
SPEC = {
    "potassium": dict(edges=np.r_[0.0, np.round(np.arange(2.8, 5.01, 0.1), 2), 99.0], step=10.0, cap=100.0,
                      direction=-1, unit="mEq/L", dose_unit="mEq KCl", a_ref=0.0),
    "heparin": dict(edges=np.r_[0.0, np.arange(30.0, 151.0, 5.0), 999.0], step=1.0, cap=6.0,
                    direction=-1, unit="s", dose_unit="units/kg/h change", a_ref=0.0),
    "insulin": dict(edges=np.r_[0.0, np.arange(80.0, 401.0, 10.0), 9999.0], step=1.0, cap=20.0,
                    direction=1, unit="mg/dL", dose_unit="units SC insulin", a_ref=0.0),
    "insulin_emar": dict(edges=np.r_[0.0, np.arange(80.0, 401.0, 10.0), 9999.0], step=1.0, cap=20.0,
                         direction=1, unit="mg/dL", dose_unit="units SC insulin (eMAR scale)", a_ref=0.0),
}
WINDOW_H = {"potassium": 6.0, "heparin": 4.0, "insulin": 2.0}
MAX_GAP_H = {"potassium": 24.0, "heparin": 24.0, "insulin": 12.0, "insulin_emar": 12.0}


# --------------------------------------------------------------------------- reading
def root(source: str = "demo") -> Path:
    """Folder holding hosp/ and icu/ for the source ("demo", "full" or a folder name in data/raw)."""
    if source == "demo":
        p = RAW / DEMO
    elif source == "full":
        cands = sorted(p for p in RAW.glob("mimiciv_*") if (p / "hosp").exists())
        if not cands:
            raise FileNotFoundError(f"no data/raw/mimiciv_<version>/hosp under {RAW}; see README.md")
        p = cands[-1]
    else:
        p = RAW / source
    if not (p / "hosp").exists():
        hint = "run `python -m experiments.datasets.mimic_titration.download`" if source == "demo" else "see README.md"
        raise FileNotFoundError(f"{p}/hosp missing; {hint}")
    return p


def _read(source: str, rel: str, usecols: Sequence[str], key: str = "", filt: Optional[Callable] = None,
          dates: Sequence[str] = (), numeric: Sequence[str] = (), text: Sequence[str] = (),
          rebuild: bool = False) -> pd.DataFrame:
    """Columns ``usecols`` of hosp/<t> or icu/<t>, rows passing ``filt(chunk)``; cached as parquet."""
    r = root(source)
    cache = PROC / r.name / f"{rel.replace('/', '__')}{'__' + key if key else ''}.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    path = r / f"{rel}.csv.gz"
    if not path.exists():
        path = r / f"{rel}.csv"
    parts = []
    for ch in pd.read_csv(path, usecols=list(usecols), chunksize=2_000_000, low_memory=False, dtype={c: str for c in text}):
        parts.append(ch[filt(ch)] if filt is not None else ch)
    df = pd.concat(parts, ignore_index=True)
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def admissions(source="demo") -> pd.DataFrame:
    a = _read(source, "hosp/admissions", ["subject_id", "hadm_id", "admittime", "dischtime"],
              dates=("admittime", "dischtime"))
    p = _read(source, "hosp/patients", ["subject_id", "gender", "anchor_age", "anchor_year"])
    return a.merge(p, on="subject_id", how="left")


def icustays(source="demo") -> pd.DataFrame:
    return _read(source, "icu/icustays", ["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
                 dates=("intime", "outtime"))


def transfers(source="demo") -> pd.DataFrame:
    t = _read(source, "hosp/transfers", ["subject_id", "hadm_id", "careunit", "intime", "outtime"],
              dates=("intime", "outtime"), text=("careunit",))
    return t[t.intime.notna()].sort_values("intime").reset_index(drop=True)


def labs(source="demo") -> pd.DataFrame:
    ids = list(LAB.values())
    d = _read(source, "hosp/labevents", ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"],
              key="titration", filt=lambda c: c.itemid.isin(ids), dates=("charttime", "storetime"))
    return d[d.valuenum.notna() & d.charttime.notna()].reset_index(drop=True)


def chart(source="demo") -> pd.DataFrame:
    ids = GLU_CHART + list(WT_CHART)
    d = _read(source, "icu/chartevents", ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
              key="titration", filt=lambda c: c.itemid.isin(ids), dates=("charttime",))
    return d[d.valuenum.notna() & d.charttime.notna()].reset_index(drop=True)


def inputs(source="demo") -> pd.DataFrame:
    ids = list(IN.values())
    return _read(source, "icu/inputevents",
                 ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "rate",
                  "rateuom", "orderid", "ordercategoryname", "patientweight", "totalamount", "totalamountuom",
                  "statusdescription"],
                 key="titration", filt=lambda c: c.itemid.isin(ids), dates=("starttime", "endtime"),
                 text=("amountuom", "rateuom", "ordercategoryname", "totalamountuom", "statusdescription"))


def emar(source="demo") -> pd.DataFrame:
    """eMAR records for potassium chloride, heparin and insulin, with the dose fields of emar_detail.

    The top-level detail row (parent_field_ordinal empty) holds the ordered dose (dose_due),
    the administration type and the infusion rates; the component rows (1.1, 1.2, ...) hold
    the dose given per product, summed here into ``dose_given``.
    """
    pat = r"(?i)^\s*(?:potassium chloride|heparin|insulin)"
    e = _read(source, "hosp/emar", ["subject_id", "hadm_id", "emar_id", "charttime", "medication", "event_txt"],
              key="titration", filt=lambda c: c.medication.astype(str).str.contains(pat, regex=True),
              dates=("charttime",), text=("medication", "event_txt", "emar_id"))
    ids = set(e.emar_id)
    d = _read(source, "hosp/emar_detail",
              ["subject_id", "emar_id", "parent_field_ordinal", "administration_type", "dose_due", "dose_due_unit",
               "dose_given", "dose_given_unit", "product_description", "infusion_rate", "prior_infusion_rate",
               "infusion_rate_adjustment", "infusion_rate_unit"],
              key="titration", filt=lambda c: c.emar_id.isin(ids),
              numeric=("parent_field_ordinal", "dose_due", "dose_given", "infusion_rate", "prior_infusion_rate"),
              text=("emar_id", "administration_type", "dose_due_unit", "dose_given_unit", "product_description",
                    "infusion_rate_adjustment", "infusion_rate_unit"))
    top = d[d.parent_field_ordinal.isna()].drop_duplicates("emar_id")
    comp = d[d.parent_field_ordinal.notna()]
    given = comp.groupby("emar_id").dose_given.sum(min_count=1).rename("dose_given")
    prod = comp.groupby("emar_id").product_description.agg(lambda s: " | ".join(sorted(set(s.dropna()))))
    out = e.merge(top.drop(columns=["subject_id", "parent_field_ordinal", "dose_given", "dose_given_unit",
                                    "product_description"]), on="emar_id", how="left")
    out = out.merge(given, on="emar_id", how="left").merge(prod.rename("products"), on="emar_id", how="left")
    out["event_txt"] = out.event_txt.fillna("").str.strip()
    out["medication"] = out.medication.str.strip()
    return out


def pharmacy(source="demo") -> pd.DataFrame:
    return _read(source, "hosp/pharmacy", ["subject_id", "hadm_id", "starttime", "stoptime", "medication",
                                           "sliding_scale", "proc_type"],
                 key="titration", filt=lambda c: c.medication.astype(str).str.contains("(?i)insulin", regex=True),
                 dates=("starttime", "stoptime"), text=("medication", "sliding_scale", "proc_type"))


def omr_weight(source="demo") -> pd.DataFrame:
    o = _read(source, "hosp/omr", ["subject_id", "chartdate", "result_name", "result_value"], key="weight",
              filt=lambda c: c.result_name.astype(str).str.startswith("Weight"), dates=("chartdate",),
              text=("result_name", "result_value"))
    o["kg"] = pd.to_numeric(o.result_value, errors="coerce") * np.where(o.result_name.str.contains("Lbs"),
                                                                       0.45359237, 1.0)
    return o.rename(columns={"chartdate": "time"})[["subject_id", "time", "kg"]].dropna()


# --------------------------------------------------------------------------- helpers
def _asof(ev: pd.DataFrame, tcol: str, ref: pd.DataFrame, rtcol: str, cols: Sequence[str],
          direction="backward", tol: Optional[pd.Timedelta] = None) -> pd.DataFrame:
    """For each event, the latest (or next) row of ``ref`` for the same subject; ``ev`` order kept."""
    left = ev[["subject_id", tcol]].copy()
    left["_i"] = np.arange(len(ev))
    left = left.sort_values(tcol)
    right = ref[["subject_id", rtcol] + [c for c in cols if c != rtcol]].dropna(subset=[rtcol]).sort_values(rtcol)
    right = right.rename(columns={rtcol: "_rt"})
    m = pd.merge_asof(left, right, left_on=tcol, right_on="_rt", by="subject_id", direction=direction,
                      tolerance=tol, allow_exact_matches=True)
    m = m.sort_values("_i")
    out = m[[c for c in cols if c != rtcol]].reset_index(drop=True)
    out[rtcol + "_ref"] = m["_rt"].to_numpy()
    return out


def _wsum(ev: pd.DataFrame, lo: str, hi: str, doses: pd.DataFrame, val: str = "amount", tcol: str = "time"
          ) -> np.ndarray:
    """Sum of ``doses[val]`` with time in [ev[lo], ev[hi]) for the same subject."""
    out = np.zeros(len(ev))
    if len(doses) == 0:
        return out
    d = doses.sort_values(["subject_id", tcol])
    grp = {s: (g[tcol].to_numpy(), np.r_[0.0, np.cumsum(g[val].to_numpy(float))]) for s, g in d.groupby("subject_id")}
    for s, idx in ev.groupby("subject_id").indices.items():
        if s not in grp:
            continue
        t, cs = grp[s]
        a = np.searchsorted(t, ev[lo].to_numpy()[idx], side="left")
        b = np.searchsorted(t, ev[hi].to_numpy()[idx], side="left")
        out[idx] = cs[b] - cs[a]
    return out


def _rate_at(ev_subject: np.ndarray, times: np.ndarray, seg: pd.DataFrame) -> np.ndarray:
    """Infusion rate (per hour) in effect at ``times`` from segments (subject_id, start, end, rate)."""
    out = np.zeros(len(times))
    seg = seg.sort_values(["subject_id", "start"])
    grp = {s: (g.start.to_numpy(), g.end.to_numpy(), g.rate.to_numpy(float)) for s, g in seg.groupby("subject_id")}
    subj = pd.Series(ev_subject)
    for s, idx in subj.groupby(subj).indices.items():
        if s not in grp:
            continue
        st, en, ra = grp[s]
        t = times[idx]
        j = np.searchsorted(st, t, side="right") - 1
        ok = j >= 0
        jj = np.where(ok, j, 0)
        # several segments may overlap (two bags); take the max rate among those covering t
        r = np.where(ok & (t < en[jj]), ra[jj], 0.0)
        for back in (1, 2):
            jb = jj - back
            okb = ok & (jb >= 0)
            jb = np.where(okb, jb, 0)
            r = np.maximum(r, np.where(okb & (t < en[jb]) & (t >= st[jb]), ra[jb], 0.0))
        out[idx] = r
    return out


UNIT_CLASS = [
    ("icu_cardiac", r"Cardiac Vascular Intensive|Coronary Care"),
    ("icu_medical", r"Medical Intensive|Medical/Surgical Intensive"),
    ("icu_surgical", r"Surgical Intensive|Trauma SICU|Neuro Surgical Intensive"),
    ("ward_cardiac", r"Cardiology|Cardiac Surgery|Vascular"),
    ("ward_medicine", r"^Medicine|Hematology|Oncology|Transplant"),
    ("ward_surgery", r"Med/Surg|Surgery|Trauma|PACU"),
    ("ward_neuro", r"Neuro"),
]


def unit_class(careunit: pd.Series) -> pd.Series:
    out = pd.Series("other", index=careunit.index)
    cu = careunit.fillna("")
    for name, pat in reversed(UNIT_CLASS):
        out[cu.str.contains(pat, regex=True)] = name
    return out


def _context(ev: pd.DataFrame, source: str, tcol: str = "t0") -> pd.DataFrame:
    """Admission, age, sex, ICU stay, care unit, weight and creatinine at ``tcol``."""
    adm = admissions(source)
    a = _asof(ev, tcol, adm, "admittime", ["hadm_id", "dischtime", "gender", "anchor_age", "anchor_year"])
    inside = (a.dischtime.to_numpy() >= ev[tcol].to_numpy())
    ev = ev.copy()
    ev["hadm"] = np.where(inside, a.hadm_id, np.nan)
    ev["age"] = a.anchor_age.to_numpy(float) + (ev[tcol].dt.year.to_numpy() - a.anchor_year.to_numpy(float))
    ev["male"] = (a.gender == "M").astype(float).to_numpy()
    icu = icustays(source)
    s = _asof(ev, tcol, icu, "intime", ["stay_id", "outtime", "first_careunit"])
    ev["in_icu"] = (s.outtime.to_numpy() > ev[tcol].to_numpy()) & s.stay_id.notna().to_numpy()
    ev["stay_id"] = np.where(ev.in_icu, s.stay_id, np.nan)
    tr = transfers(source)
    u = _asof(ev, tcol, tr, "intime", ["careunit", "outtime"])
    cu = u.careunit.where(u.outtime.isna() | (u.outtime.to_numpy() > ev[tcol].to_numpy()))
    ev["careunit"] = cu.fillna("Unknown").to_numpy()
    ev["unit"] = unit_class(ev.careunit).to_numpy()
    ev.loc[ev.in_icu & ~ev.unit.str.startswith("icu"), "unit"] = "icu_other"
    # weight: latest before (ICU charting, inputevents, OMR), else the first after
    ch = chart(source)
    w1 = ch[ch.itemid.isin(list(WT_CHART))].copy()
    w1["kg"] = w1.valuenum * w1.itemid.map(WT_CHART)
    w2 = inputs(source)[["subject_id", "starttime", "patientweight"]].rename(columns={"starttime": "charttime",
                                                                                     "patientweight": "kg"})
    w3 = omr_weight(source).rename(columns={"time": "charttime"})
    w = pd.concat([w1[["subject_id", "charttime", "kg"]], w2, w3], ignore_index=True)
    w = w[(w.kg > 25) & (w.kg < 300)]
    wb = _asof(ev, tcol, w, "charttime", ["kg"]).kg.to_numpy()
    wf = _asof(ev, tcol, w, "charttime", ["kg"], direction="forward").kg.to_numpy()
    ev["weight"] = np.where(np.isfinite(wb), wb, wf)
    lb = labs(source)
    cr = lb[lb.itemid == LAB["creatinine"]]
    c = _asof(ev, tcol, cr, "charttime", ["valuenum"], tol=pd.Timedelta(hours=72))
    ev["creatinine"] = c.valuenum.to_numpy()
    ev["hour"] = ev[tcol].dt.hour + ev[tcol].dt.minute / 60.0
    return ev


def _complete(ev: pd.DataFrame, source: str) -> np.ndarray:
    """True when oral and ward doses are recorded: for a ward event, the admission has eMAR
    records; for an ICU event, the ICU stay itself has eMAR records (some ICU stays in eMAR-era
    admissions have none, so their oral doses are missing). Reads every eMAR record's time from
    hosp/emar, or from hosp/emar_hours (one row per subject, admission and hour with any eMAR
    record; the BigQuery export in README.md) when that file exists."""
    r = root(source)
    rel = "hosp/emar_hours" if any((r / f"hosp/emar_hours{x}").exists() for x in (".csv.gz", ".csv")) else "hosp/emar"
    e = _read(source, rel, ["subject_id", "hadm_id", "charttime"], key="times", dates=("charttime",))
    era = set(e.hadm_id.dropna().astype(float))
    s = _asof(e, "charttime", icustays(source), "intime", ["stay_id", "outtime"])
    stays = set(s.stay_id[(s.outtime.to_numpy() > e.charttime.to_numpy())].dropna())
    icu = ev.in_icu.to_numpy(bool)
    return np.where(icu, ev.stay_id.isin(stays).to_numpy(), ev.hadm.isin(era).to_numpy())


def _series(lab: pd.DataFrame) -> pd.DataFrame:
    """Previous and next value of the same lab for each draw (same subject)."""
    d = lab.sort_values(["subject_id", "charttime"]).drop_duplicates(["subject_id", "charttime"]).reset_index(drop=True)
    g = d.groupby("subject_id")
    d["prev_value"] = g.valuenum.shift(1)
    d["prev_time"] = g.charttime.shift(1)
    d["next_value"] = g.valuenum.shift(-1)
    d["next_time"] = g.charttime.shift(-1)
    return d


# --------------------------------------------------------------------------- doses
def kcl_doses(source="demo") -> pd.DataFrame:
    """Potassium chloride given: subject_id, time, amount (mEq), route ("iv_icu", "emar_iv", "emar_oral"),
    maintenance (additive to a bag of >= 500 mL or a TPN-like fluid)."""
    ie = inputs(source)
    k = ie[(ie.itemid == IN["kcl"]) & (ie.amountuom.str.lower() == "meq")].copy()
    k = k[~k.statusdescription.isin(["Rewritten"])]
    k["maintenance"] = k.totalamount.fillna(0) >= 500
    iv = k.rename(columns={"starttime": "time"})[["subject_id", "time", "amount", "maintenance"]]
    iv["route"] = "iv_icu"
    e = emar(source)
    e = e[e.medication.str.match(r"(?i)potassium chloride") & e.event_txt.isin(GIVEN)].copy()
    e["amount"] = e.dose_given.fillna(e.dose_due)
    prod = e.products.fillna("") + " " + e.medication
    e["oral"] = prod.str.contains(r"(?i)tablet|powder|packet|oral|liquid|elixir|ER ")
    e["maintenance"] = prod.str.contains(r"(?i)/ ?(?:500|1000) ?mL")
    e["route"] = np.where(e.oral, "emar_oral", "emar_iv")
    e = e.rename(columns={"charttime": "time"})
    # IV KCl inside an ICU stay is charted in inputevents too: keep only the eMAR oral doses there
    icu = icustays(source)
    s = _asof(e, "time", icu, "intime", ["outtime"])
    e["in_icu"] = (s.outtime.to_numpy() > e.time.to_numpy())
    e = e[~(e.in_icu & ~e.oral)]
    out = pd.concat([iv, e[["subject_id", "time", "amount", "maintenance", "route"]]], ignore_index=True)
    return out[out.amount.notna() & (out.amount > 0)].reset_index(drop=True)


def crrt_times(source="demo") -> pd.DataFrame:
    ie = inputs(source)
    c = ie[ie.itemid == IN["kcl_crrt"]]
    return c.rename(columns={"starttime": "start", "endtime": "end"})[["subject_id", "start", "end"]]


def heparin_segments(source="demo") -> pd.DataFrame:
    """Heparin infusion rate segments (subject_id, start, end, rate units/h, weight, src).

    ICU: inputevents 225152, ordercategory 01-Drips (each row runs from starttime to endtime at
    ``rate``). Wards: eMAR "Heparin IV Infusion" records; each sets the rate (0 when stopped)
    until the next record, at most 12 h. eMAR records inside an ICU stay are dropped (the same
    drip is charted in inputevents)."""
    ie = inputs(source)
    h = ie[(ie.itemid == IN["heparin"]) & (ie.ordercategoryname == "01-Drips") & (ie.rateuom == "units/hour")]
    icu_seg = h.rename(columns={"starttime": "start", "endtime": "end"})[["subject_id", "start", "end", "rate",
                                                                          "patientweight"]]
    icu_seg = icu_seg.rename(columns={"patientweight": "weight"})
    icu_seg["src"] = "icu"
    e = emar(source)
    e = e[(e.medication == "Heparin") & (e.administration_type == "Heparin IV Infusion")].copy()
    stop = e.event_txt.str.startswith("Stopped") | e.event_txt.isin(["Not Started", "Hold Dose"])
    e["rate"] = np.where(stop, 0.0, e.infusion_rate)
    e = e[e.rate.notna() & e.event_txt.ne("Not Confirmed")].sort_values(["subject_id", "charttime"])
    icu = icustays(source)
    s = _asof(e, "charttime", icu, "intime", ["outtime"])
    e = e[~(s.outtime.to_numpy() > e.charttime.to_numpy())]
    nxt = e.groupby("subject_id").charttime.shift(-1)
    e["start"] = e.charttime
    e["end"] = pd.concat([nxt, e.charttime + pd.Timedelta(hours=12)], axis=1).min(axis=1)
    ward = e[["subject_id", "start", "end", "rate"]].copy()
    ward["weight"] = np.nan
    ward["src"] = "ward"
    seg = pd.concat([icu_seg, ward], ignore_index=True)
    return seg[seg.end > seg.start].reset_index(drop=True)


def heparin_boluses(source="demo") -> pd.DataFrame:
    ie = inputs(source)
    b = ie[(ie.itemid == IN["heparin"]) & (ie.ordercategoryname == "05-Med Bolus")]
    b = b.rename(columns={"starttime": "time"})[["subject_id", "time", "amount"]]
    e = emar(source)
    e = e[(e.medication == "Heparin") & (e.administration_type == "Heparin IV Bolus") & e.event_txt.isin(GIVEN)]
    e = e.rename(columns={"charttime": "time"})
    e["amount"] = e.dose_given.fillna(e.dose_due)
    return pd.concat([b, e[["subject_id", "time", "amount"]]], ignore_index=True)


def insulin_doses(source="demo") -> pd.DataFrame:
    """Subcutaneous short-acting insulin (units) from inputevents, and long-acting doses (flag)."""
    ie = inputs(source)
    short = ie[((ie.itemid == IN["ins_regular"]) & (ie.ordercategoryname == "06-Insulin (Non IV)"))
               | (ie.itemid == IN["ins_lispro"])].copy()
    short["kind"] = "short"
    long_ = ie[ie.itemid.isin([IN["ins_glargine"], IN["ins_nph"]])].copy()
    long_["kind"] = "long"
    d = pd.concat([short, long_]).rename(columns={"starttime": "time"})
    return d[["subject_id", "time", "amount", "kind"]].dropna(subset=["amount"])


def insulin_drip(source="demo") -> pd.DataFrame:
    ie = inputs(source)
    d = ie[(ie.itemid == IN["ins_regular"]) & (ie.ordercategoryname == "01-Drips")]
    return d.rename(columns={"starttime": "start", "endtime": "end"})[["subject_id", "start", "end", "rate"]]


# --------------------------------------------------------------------------- events
def _finish(ev: pd.DataFrame, protocol: str, source: str) -> pd.DataFrame:
    ev = ev.copy()
    ev["gap_h"] = (ev.next_time - ev.t0).dt.total_seconds() / 3600
    ev["prev_gap_h"] = (ev.t0 - ev.prev_time).dt.total_seconds() / 3600
    ev["Y"] = np.where(ev.gap_h <= MAX_GAP_H[protocol], ev.next_value, np.nan)
    sp = SPEC[protocol]
    ev["A_level"] = np.clip(np.round(ev.A / sp["step"]) * sp["step"], -sp["cap"], sp["cap"])
    ev["patient"] = ev.subject_id.astype(int)
    ev["group"] = np.where(ev.in_icu, "icu", "ward")
    ev["protocol"] = protocol
    ev["source"] = root(source).name
    return ev.reset_index(drop=True)


def potassium_events(source="demo") -> pd.DataFrame:
    lb = labs(source)
    k = lb[(lb.itemid == LAB["potassium"]) & lb.valuenum.between(1.5, 9.0)]
    d = _series(k).rename(columns={"charttime": "t0", "valuenum": "Q"})
    d["t_result"] = d.storetime.fillna(d.t0 + pd.Timedelta(hours=1.5))
    d.loc[d.t_result < d.t0, "t_result"] = d.t0
    d["t_end"] = pd.concat([d.next_time.fillna(d.t_result + pd.Timedelta(days=99)),
                            d.t_result + pd.Timedelta(hours=WINDOW_H["potassium"])], axis=1).min(axis=1)
    d = d[d.t_end > d.t_result]
    d["t_prev24"] = d.t0 - pd.Timedelta(hours=24)
    kd = kcl_doses(source)
    rep = kd[~kd.maintenance]
    d["A"] = _wsum(d, "t_result", "t_end", rep)
    d["dose_pending"] = _wsum(d, "t0", "t_result", rep)            # given between draw and result
    d["prev_dose"] = _wsum(d, "t_prev24", "t0", rep)
    d["maint_k_24h"] = _wsum(d, "t_prev24", "t0", kd[kd.maintenance])
    oral = rep[rep.route == "emar_oral"]
    d["A_oral"] = _wsum(d, "t_result", "t_end", oral)
    cr = crrt_times(source)
    cr["time"] = cr.start
    cr["amount"] = 1.0
    d["t_m12"] = d.t0 - pd.Timedelta(hours=12)
    d["t_p12"] = d.t0 + pd.Timedelta(hours=12)
    d["crrt"] = _wsum(d, "t_m12", "t_p12", cr) > 0
    d = _context(d, source, "t0")
    d = d[d.hadm.notna()]
    d["complete"] = _complete(d, source)
    return _finish(d, "potassium", source)


def heparin_events(source="demo") -> pd.DataFrame:
    lb = labs(source)
    p = lb[(lb.itemid == LAB["ptt"]) & lb.valuenum.between(10, 250)]
    d = _series(p).rename(columns={"charttime": "t0", "valuenum": "Q"})
    d["t_result"] = d.storetime.fillna(d.t0 + pd.Timedelta(hours=1))
    d.loc[d.t_result < d.t0, "t_result"] = d.t0
    d["t_end"] = pd.concat([d.next_time.fillna(d.t_result + pd.Timedelta(days=99)),
                            d.t_result + pd.Timedelta(hours=WINDOW_H["heparin"])], axis=1).min(axis=1)
    d = d[d.t_end > d.t_result + pd.Timedelta(minutes=30)]
    seg = heparin_segments(source)
    subj = d.subject_id.to_numpy()
    eps = np.timedelta64(1, "m")
    d["rate_before"] = _rate_at(subj, (d.t_result.to_numpy() - eps), seg)
    d["rate_draw"] = _rate_at(subj, d.t0.to_numpy(), seg)
    d["rate_after"] = _rate_at(subj, (d.t_end.to_numpy() - eps), seg)
    # minimum rate over the window on a 10-minute grid (a hold shows as 0)
    span = (d.t_end - d.t_result).to_numpy()
    mins = np.full(len(d), np.inf)
    for f in np.linspace(0.0, 1.0, 25)[1:-1]:
        mins = np.minimum(mins, _rate_at(subj, d.t_result.to_numpy() + (span * f).astype("timedelta64[ns]"), seg))
    d["rate_min"] = mins
    d = d[(d.rate_before > 0) & (d.rate_draw > 0)]           # infusion running at the draw and the result
    d["stopped"] = d.rate_after == 0
    d["hold"] = (d.rate_min == 0) & ~d.stopped
    srcmap = seg.groupby("subject_id").src.agg(lambda s: "icu" if (s == "icu").any() else "ward")
    d["rate_src"] = d.subject_id.map(srcmap)
    bol = heparin_boluses(source)
    d["bolus"] = _wsum(d, "t_result", "t_end", bol) > 0
    d["t_prev24"] = d.t0 - pd.Timedelta(hours=24)
    d = _context(d, source, "t0")
    d = d[d.hadm.notna() & d.weight.notna()]
    # weight from the drip rows if charted there
    d["A"] = (d.rate_after - d.rate_before) / d.weight
    d["rate_kg"] = d.rate_before / d.weight
    d["category"] = np.select([d.stopped, d.hold, d.A > 0.25, d.A < -0.25], ["stopped", "hold", "increase", "decrease"],
                              "no change")
    d = d.sort_values(["subject_id", "t0"])
    d["prev_change"] = d.groupby("subject_id").A.shift(1).fillna(0.0)
    d["complete"] = _complete(d, source) | d.in_icu
    return _finish(d, "heparin", source)


def insulin_events(source="demo") -> pd.DataFrame:
    ch = chart(source)
    g = ch[ch.itemid.isin(GLU_CHART) & ch.valuenum.between(20, 1500)]
    d = _series(g).rename(columns={"charttime": "t0", "valuenum": "Q"})
    d["t_lo"] = d.t0 - pd.Timedelta(minutes=15)
    d["t_end"] = pd.concat([d.next_time.fillna(d.t0 + pd.Timedelta(days=99)) - pd.Timedelta(minutes=15),
                            d.t0 + pd.Timedelta(hours=WINDOW_H["insulin"])], axis=1).min(axis=1)
    d = d[d.t_end > d.t0]
    # active sliding-scale order
    ph = pharmacy(source)
    ss = ph[ph.sliding_scale.eq("Y")].copy()
    ss["time"] = ss.starttime
    ss["amount"] = 1.0
    stop = ss.copy()
    stop["time"] = stop.stoptime
    stop["amount"] = -1.0
    d["t_eps"] = d.t0 + pd.Timedelta(seconds=1)
    d["_lo0"] = pd.Timestamp("1900-01-01")
    active = _wsum(d, "_lo0", "t_eps", pd.concat([ss, stop])[["subject_id", "time", "amount"]].dropna())
    d["on_scale"] = active > 0
    drip = insulin_drip(source)
    subj = d.subject_id.to_numpy()
    hr = np.timedelta64(1, "h")
    d["on_drip"] = ((_rate_at(subj, d.t0.to_numpy(), drip) > 0) | (_rate_at(subj, d.t0.to_numpy() - hr, drip) > 0)
                    | (_rate_at(subj, d.t0.to_numpy() + hr, drip) > 0))
    ins = insulin_doses(source)
    sh, lg = ins[ins.kind == "short"], ins[ins.kind == "long"]
    d["A"] = _wsum(d, "t_lo", "t_end", sh)
    d["t_prev24"] = d.t0 - pd.Timedelta(hours=24)
    d["prev_dose"] = _wsum(d, "t_prev24", "t_lo", sh)
    d["long_24h"] = _wsum(d, "t_prev24", "t_end", lg)
    d = d[d.on_scale & ~d.on_drip]
    d = _context(d, source, "t0")
    d = d[d.hadm.notna() & d.in_icu]
    d["complete"] = True
    return _finish(d, "insulin", source)


def insulin_emar_events(source="demo") -> pd.DataFrame:
    e = emar(source)
    e = e[(e.administration_type == "Insulin SC Sliding Scale")
          & e.event_txt.isin(["Administered", "Not Given per Sliding Scale", "Delayed Administered",
                              "Administered in Other Location"])].copy()
    e["A"] = np.where(e.event_txt == "Not Given per Sliding Scale", 0.0, e.dose_given.fillna(e.dose_due))
    e["dose_due"] = e.dose_due
    e = e.rename(columns={"charttime": "t_dose"})
    ch = chart(source)
    lb = labs(source)
    glu = pd.concat([ch[ch.itemid.isin(GLU_CHART)][["subject_id", "charttime", "valuenum", "itemid"]],
                     lb[lb.itemid.isin([LAB["glucose"], LAB["glucose_bg"]])][["subject_id", "charttime", "valuenum",
                                                                               "itemid"]]])
    glu = glu[glu.valuenum.between(20, 1500)]
    s = _series(glu)
    m = _asof(e, "t_dose", s, "charttime", ["valuenum", "itemid", "prev_value", "prev_time", "next_value", "next_time"],
              tol=pd.Timedelta(hours=1))
    e = pd.concat([e.reset_index(drop=True), m], axis=1)
    e = e[e.valuenum.notna()].rename(columns={"valuenum": "Q", "charttime_ref": "t0"})
    e["glucose_item"] = e.itemid
    e["t_prev24"] = e.t0 - pd.Timedelta(hours=24)
    e = _context(e, source, "t0")
    e = e[e.hadm.notna()]
    e["complete"] = True
    return _finish(e, "insulin_emar", source)


BUILDERS = {"potassium": potassium_events, "heparin": heparin_events, "insulin": insulin_events,
            "insulin_emar": insulin_emar_events}


def events(protocol: str = "potassium", source: str = "demo", rebuild: bool = False) -> pd.DataFrame:
    """Event table for ``protocol``; cached in data/processed/<folder>/events_<protocol>.parquet."""
    if protocol not in BUILDERS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    f = PROC / root(source).name / f"events_{protocol}.parquet"
    if f.exists() and not rebuild:
        return pd.read_parquet(f)
    ev = BUILDERS[protocol](source)
    f.parent.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=[c for c in ev.columns if c.startswith("_")]).to_parquet(f)
    return ev


# --------------------------------------------------------------------------- covariates
X_BASE = ["age", "male", "weight", "creatinine", "prev_value", "prev_gap_h", "prev_dose", "hour"]
X_EXTRA = {"potassium": ["maint_k_24h"], "heparin": ["rate_kg", "prev_change"], "insulin": ["long_24h"],
           "insulin_emar": []}


def design(ev: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """X: age, sex, weight, creatinine, previous value and dose, hours since the previous value,
    hour of day, unit type indicators; missing values set to the median with an indicator."""
    ev = ev.copy()
    if "prev_dose" not in ev:
        ev["prev_dose"] = 0.0
    X = pd.DataFrame(index=ev.index)
    for c in X_BASE + X_EXTRA[protocol]:
        v = ev[c].astype(float) if c in ev else pd.Series(np.nan, index=ev.index)
        if c == "prev_gap_h":
            v = np.log1p(v.clip(lower=0, upper=24 * 14))
        med = v.median() if v.notna().any() else 0.0
        X[c] = v.fillna(med)
        if v.isna().any():
            X[c + "_na"] = v.isna().astype(float)
    units = sorted(ev.unit.unique())
    for u in units[1:]:
        X["unit_" + u] = (ev.unit == u).astype(float)
    return X.loc[:, X.std(axis=0) > 0]


def r2_score_on_x(ev: pd.DataFrame, protocol: str) -> float:
    X = design(ev, protocol)
    Z = np.column_stack([np.ones(len(ev)), X.to_numpy(float)])
    q = ev.Q.to_numpy(float)
    res = q - Z @ np.linalg.lstsq(Z, q, rcond=None)[0]
    return float(1 - res.var() / q.var())


# --------------------------------------------------------------------------- the rule
def fit_step_rule(Q, A, edges, direction: int = -1) -> Dict:
    """Best step rule on fine score bins: the dose per bin that most events received.

    direction 0: the modal dose in each bin (an upper bound on any step rule's fit with these
    bins). direction -1 / +1: the best rule whose dose never rises / never falls as the score
    rises (dynamic programme over the occupied bins). Returns the bins' doses, the merged
    steps [(lo, hi, dose)] and the share of events matching."""
    Q, A = np.asarray(Q, float), np.asarray(A, float)
    b = np.clip(np.searchsorted(edges, Q, side="right") - 1, 0, len(edges) - 2)
    occ = np.unique(b)
    levels = np.unique(A)
    li = np.searchsorted(levels, A)
    cnt = np.zeros((len(occ), len(levels)))
    np.add.at(cnt, (np.searchsorted(occ, b), li), 1)
    if direction == 0:
        pick = cnt.argmax(1)
    else:
        nb, nl = cnt.shape
        best = np.zeros((nb, nl))
        arg = np.zeros((nb, nl), int)
        best[0] = cnt[0]
        for i in range(1, nb):
            prev = best[i - 1]
            if direction > 0:      # level at i >= level at i-1: prefix max
                cm = np.maximum.accumulate(prev)
                ca = np.array([int(np.argmax(prev[: l + 1])) for l in range(nl)])
            else:                  # level at i <= level at i-1: suffix max
                cm = np.maximum.accumulate(prev[::-1])[::-1]
                ca = np.array([l + int(np.argmax(prev[l:])) for l in range(nl)])
            best[i] = cnt[i] + cm
            arg[i] = ca
        pick = np.zeros(nb, int)
        pick[-1] = int(np.argmax(best[-1]))
        for i in range(nb - 1, 0, -1):
            pick[i - 1] = arg[i, pick[i]]
    rule = levels[pick]
    match = float(cnt[np.arange(len(occ)), pick].sum() / len(A))
    steps: List[tuple] = []
    for j, bi in enumerate(occ):
        lo, hi = edges[bi], edges[bi + 1]
        if steps and steps[-1][2] == rule[j]:
            steps[-1] = (steps[-1][0], hi, rule[j])
        else:
            steps.append((lo, hi, rule[j]))
    return {"bins": occ, "bin_lo": edges[occ], "dose": rule, "steps": steps, "match": match}


def apply_rule(fit: Dict, edges, Q) -> np.ndarray:
    """Dose of a fitted step rule at score Q (bins with no events take the nearest step)."""
    Q = np.asarray(Q, float)
    out = np.empty(len(Q))
    for k, (lo, hi, a) in enumerate(fit["steps"]):
        m = (Q >= lo) & (Q < hi)
        if k == 0:
            m |= Q < lo
        if k == len(fit["steps"]) - 1:
            m |= Q >= hi
        out[m] = a
    # gaps between steps: nearest step below
    gap = ~np.isfinite(out) | np.isnan(out)
    if gap.any():
        los = np.array([s[0] for s in fit["steps"]])
        j = np.clip(np.searchsorted(los, Q[gap], side="right") - 1, 0, len(los) - 1)
        out[gap] = np.array([s[2] for s in fit["steps"]])[j]
    return out


CATEGORY_ORD = {"hold": -2.0, "decrease": -1.0, "no change": 0.0, "increase": 1.0}
CATEGORY_NAME = {int(v): k for k, v in CATEGORY_ORD.items()}
SCALE_GRID = dict(c=(100, 120, 130, 140, 150, 160, 180, 200), w=(30, 40, 50), k=(1, 2, 3, 4, 5, 6))


def sliding_scale(g, c: float = 150.0, w: float = 50.0, k: float = 2.0) -> np.ndarray:
    """Insulin units of a linear sliding scale: k units per band of width w above c,
    k * ceil((g - c) / w) for g > c and 0 otherwise (c = 150, w = 50, k = 2 gives
    151-200: 2, 201-250: 4, 251-300: 6, ...)."""
    g = np.asarray(g, float)
    return np.where(g > c, k * np.ceil((g - c) / w - 1e-9), 0.0)


def scale_fit(ev: pd.DataFrame) -> Dict:
    """Best linear sliding scale (c, w, k) for all events, and for each patient separately.

    The scale is written in each patient's order, but MIMIC-IV does not store its text, so the
    per-patient scale is inferred (3 parameters per patient; optimistic for patients with few
    events)."""
    g, a = ev.Q.to_numpy(float), ev.A_level.to_numpy(float)
    grid = [(c, w, k) for c in SCALE_GRID["c"] for w in SCALE_GRID["w"] for k in SCALE_GRID["k"]]
    M = np.column_stack([np.isclose(sliding_scale(g, c, w, k), a) for c, w, k in grid])
    j = int(M.sum(0).argmax())
    out = {"pooled": {"c": grid[j][0], "w": grid[j][1], "k": grid[j][2], "match": float(M[:, j].mean())},
           "standard_150_50_2": float(np.isclose(sliding_scale(g, 150, 50, 2), a).mean())}
    best, ks, n = 0, [], 0
    for _, idx in ev.groupby("patient").indices.items():
        s = M[idx].sum(0)
        jj = int(s.argmax())
        best += s[jj]
        ks.append(grid[jj][2])
    out["per_patient_match"] = float(best / len(ev))
    out["per_patient_k"] = pd.Series(ks).value_counts().sort_index().to_dict()
    return out


def dose_table(ev: pd.DataFrame, protocol: str, by: Optional[str] = None) -> pd.DataFrame:
    """Events by score band (rows) and rounded dose (columns)."""
    edges = SPEC[protocol]["edges"]
    band = pd.cut(ev.Q, edges, right=False)
    cols = [band] if by is None else [ev[by], band]
    return pd.crosstab(cols, ev.A_level)


def sample(protocol: str, source: str = "demo", complete_only: bool = True, with_outcome: bool = True,
           rebuild: bool = False) -> pd.DataFrame:
    """Events used for the rule check and the dose sample: complete dose capture, dose known, no
    pending dose (KCl given between the draw and the result), not on CRRT, heparin infusion not
    stopped for good inside the window, outcome observed."""
    ev = events(protocol, source, rebuild)
    keep = ev.A.notna().to_numpy()
    if complete_only:
        keep &= ev.complete.to_numpy(bool)
    if protocol == "potassium":
        keep &= (ev.dose_pending == 0).to_numpy() & ~ev.crrt.to_numpy(bool)
    if protocol == "heparin":
        keep &= ~ev.stopped.to_numpy(bool)
    if with_outcome:
        keep &= ev.Y.notna().to_numpy()
    return ev[keep].reset_index(drop=True)


def rule_check(protocol: str = "potassium", source: str = "demo", rebuild: bool = False) -> Dict:
    """Counts, dose-by-score table, best step rules and their match shares, share at the
    reference dose, R^2 of the score on X and the outcome timing."""
    sp = SPEC[protocol]
    allev = events(protocol, source, rebuild)
    ev = sample(protocol, source)
    out: Dict = {"protocol": protocol, "source": root(source).name,
                 "events_all": int(len(allev)), "events_complete": int(allev.complete.sum()),
                 "events": int(len(ev)), "patients": int(ev.patient.nunique()),
                 "admissions": int(ev.hadm.nunique()), "icu_share": float(ev.in_icu.mean()) if len(ev) else np.nan}
    if len(ev) == 0:
        return out
    pts_all = admissions(source).subject_id.nunique()
    out["events_per_patient"] = float(len(ev) / pts_all)
    out["events_per_icu_stay"] = float(ev.in_icu.sum() / max(1, icustays(source).stay_id.nunique()))
    out["share_ref"] = float((ev.A_level == sp["a_ref"]).mean())
    out["dose_levels"] = ev.A_level.value_counts().sort_index().to_dict()
    out["table"] = dose_table(ev, protocol)
    for name, dirn in (("modal", 0), ("monotone", sp["direction"])):
        f = fit_step_rule(ev.Q, ev.A_level, sp["edges"], dirn)
        out[name] = {"match": f["match"], "steps": [(float(a), float(b), float(c)) for a, b, c in f["steps"]]}
    out["r2"] = r2_score_on_x(ev, protocol)
    g = ev.gap_h
    out["gap_h"] = {"median": float(g.median()), "q25": float(g.quantile(0.25)), "q75": float(g.quantile(0.75))}
    out["outcome_share_le"] = {h: float((allev.gap_h <= h).mean()) for h in (6, 12, 24)}
    coarse = {"potassium": [0, 3.0, 3.5, 3.8, 4.0, 4.5, 99], "heparin": [0, 40, 50, 60, 80, 100, 120, 999],
              "insulin": [0, 150, 200, 250, 300, 350, 9999], "insulin_emar": [0, 150, 200, 250, 300, 350, 9999]}
    band = pd.cut(ev.Q, coarse[protocol], right=False)
    out["dosed_share_by_band"] = ev.groupby(band).A_level.agg(
        n="size", dosed=lambda a: float((a != sp["a_ref"]).mean())).reset_index()
    if protocol == "heparin":
        out["category"] = ev.category.value_counts().to_dict()
        out["category_table"] = pd.crosstab(band, ev.category)
        out["bolus_share"] = float(ev.bolus.mean())
        out["stopped_in_window"] = int(allev.stopped.sum())
        f = fit_step_rule(ev.Q, ev.category.map(CATEGORY_ORD).to_numpy(float), sp["edges"], -1)
        out["category_monotone"] = {"match": f["match"], "steps": [
            (float(a), float(b), CATEGORY_NAME[int(c)]) for a, b, c in f["steps"]]}
    if protocol.startswith("insulin"):
        out["scale_fit"] = scale_fit(ev)
    if "group" in ev:
        out["by_group"] = {k: {"n": int(len(g_)), "share_ref": float((g_.A_level == sp["a_ref"]).mean()),
                               "monotone_match": fit_step_rule(g_.Q, g_.A_level, sp["edges"], sp["direction"])["match"]}
                           for k, g_ in ev.groupby("group") if len(g_) >= 20}
    return out


# --------------------------------------------------------------------------- dose sample
def load_dose(protocol: str = "potassium", source: str = "demo", on_rule: bool = False,
              complete_only: bool = True, rebuild: bool = False) -> RDDSample:
    """Event-level dose sample: Q = the lab value (the score), A = the rounded dose after it,
    Y = the next value of the same lab, X = ``design``. Cluster id: the patient.

    ``schedule(Q, group)`` is the fitted monotone step rule (fitted on this sample, so its
    match share is optimistic); ``on_rule=True`` keeps only events whose dose equals it."""
    sp = SPEC[protocol]
    ev = sample(protocol, source, complete_only, True, rebuild)
    if len(ev) == 0:
        raise ValueError(f"no {protocol} events in {source}")
    fit = fit_step_rule(ev.Q, ev.A_level, sp["edges"], sp["direction"])

    def schedule(q, group=None, _fit=fit, _edges=sp["edges"]):
        return apply_rule(_fit, _edges, q)

    A_rule = schedule(ev.Q.to_numpy(float))
    ev["on_rule"] = np.isclose(A_rule, ev.A_level.to_numpy(float))
    if on_rule:
        keep = ev.on_rule.to_numpy()
        ev, A_rule = ev[keep].reset_index(drop=True), A_rule[keep]
    X = design(ev, protocol)
    levels = np.array(sorted(set(ev.A_level) - {sp["a_ref"]}), float)
    z = [c for c in ("age", "male", "weight", "creatinine") if c in X.columns]
    # threshold: where the fitted rule leaves the reference dose (lower edge of the reference
    # step when the dose falls with the score, upper edge when it rises)
    ref_steps = [s for s in fit["steps"] if s[2] == sp["a_ref"]]
    thr = (ref_steps[0][0] if sp["direction"] < 0 else ref_steps[-1][1]) if ref_steps else fit["steps"][0][1]
    return RDDSample(
        Q=ev.Q.to_numpy(float), X=X.to_numpy(float), Y=ev.Y.to_numpy(float), threshold=float(thr),
        name=f"mimic_titration_{protocol}", feature_names=list(X.columns),
        description=(f"MIMIC-IV ({root(source).name}) {protocol} titration events. Q = lab value ({sp['unit']}); "
                     f"A = {sp['dose_unit']} after the result (rounded to {sp['step']:g}); Y = the next value."),
        citation=("Johnson et al., MIMIC-IV (PhysioNet); MIMIC-IV Clinical Database Demo v2.2, "
                  "doi:10.13026/dp1f-ex47 (ODbL 1.0)."),
        treatment_rule=lambda q, _s=schedule: (np.asarray(_s(q)) != 0).astype(int),
        extras={
            "A": ev.A_level.to_numpy(float), "A_raw": ev.A.to_numpy(float), "a_ref": sp["a_ref"],
            "dose_levels": levels, "schedule": schedule, "schedule_steps": fit["steps"], "rule_match": fit["match"],
            "A_rule": A_rule, "on_rule": ev.on_rule.to_numpy(bool), "group": ev.group.to_numpy(),
            "Z_names": z, "patient": ev.patient.to_numpy(), "hadm": ev.hadm.to_numpy(float),
            "gap_h": ev.gap_h.to_numpy(float), "unit": ev.unit.to_numpy(), "t0": ev.t0.to_numpy(),
        },
    )
