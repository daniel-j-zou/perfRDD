"""UI weekly benefit schedules in the Upjohn reemployment-bonus experiments.

Dose design. The weekly benefit amount (WBA) is a published function of base-period
earnings: linear between a floor and a cap, in whole dollars. The running score is the
average of the claimant's two highest base-period quarters, ``s = (Q1 + Q2) / 2``.

    Washington 1988 (RCW 50.20.120, as quoted in the WREB final report):
        WBA = clip(floor(s / 25), MIN, MAX)
        MIN/MAX = 55/205 for benefit years starting before 3 July 1988,
                  57/209 from 3 July 1988 (read from the data).
    Illinois 1984 (final report: "48 percent of the average weekly wage during the
    two quarters of the base period in which earnings were highest"):
        AWW = round(2 s / 26)                    (nearest dollar; found in the data)
        WBA = clip(ceil(0.48 AWW), 51, 161)
        dependents' allowance = min(ceil(0.144 AWW), 48)  (found in the data)
        total = WBA + allowance; the cap on the total is 209.

Pennsylvania 1988-89 is not in ``load_dose``: its WBA comes from a statutory table on
high-quarter wages, and the public file has only total base-year wages. ``pa_frame``
returns the cleaned PA table and ``rule_check`` checks what can be checked there.

Reference dose: the cap. Above the cap kink the benefit is flat, and 32-35% of claimants
are there; below it the benefit falls with the score. ``Q = s / s_cap(group)``, so the
threshold is 1 in every group, and the dose varies for ``Q < 1``. ``ref="floor"`` uses
the lower kink instead (few claimants, and not available for Illinois claimants with
dependents, whose total keeps rising below the floor).

The randomized bonus is a second dose at a fixed score: Washington offered 2, 4 or 6
times the WBA (``extras["bonus"]``), Illinois a flat $500 to the claimant (JSIE) or to
the employer (HIE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).parent / "data" / "raw"
WA_FILE = RAW / "wa_reemp_bonus" / "data" / "wreb.dta"
IL_FILE = RAW / "il_ui_experiment_data-images-install" / "data" / "illinois.dta"
PA_FILE = RAW / "PA_ReempBonus" / "cd" / "data" / "Upjohn Institute Created files" / "Recsfile.dta"

WA_SWITCH = pd.Timestamp("1988-07-03")  # first benefit year with MIN/MAX 57/209

# group -> parameters. s_cap: smallest s with the dose at its cap. s_floor: smallest s
# with the dose above its floor (None if the dose has no flat floor region).
GROUPS: Dict[str, dict] = {
    "WA_88H1": {"state": "WA", "min": 55, "max": 205, "s_cap": 5125.0, "s_floor": 1400.0},
    "WA_88H2": {"state": "WA", "min": 57, "max": 209, "s_cap": 5225.0, "s_floor": 1450.0},
    "IL_nodep": {"state": "IL", "min": 51, "max": 161, "s_cap": 4335.5, "s_floor": 1384.5},
    "IL_dep": {"state": "IL", "min": 51, "max": 209, "s_cap": 4335.5, "s_floor": None},
}

WA_MULT = {"0": 0, "1": 2, "2": 4, "3": 6, "4": 2, "5": 4, "6": 6}
WA_QUAL = {"0": "none", "1": "short", "2": "short", "3": "short",
           "4": "long", "5": "long", "6": "long"}
PA_MULT = {0: 0, 1: 3, 2: 3, 3: 6, 4: 6, 5: 6, 6: 6}
PA_MAX = {1988: 252, 1989: 266}

Z_NAMES = ["age", "male", "black", "hispanic", "log_lag_earn", "educ", "log_prebase_earn"]


# --------------------------------------------------------------------------- schedule
def wba_dollars(s: np.ndarray, group: np.ndarray, total: bool = True) -> np.ndarray:
    """Weekly benefit in dollars from the score ``s`` (dollars) and the group label.

    ``total=True`` adds the Illinois dependents' allowance for group ``IL_dep``.
    """
    s = np.asarray(s, dtype=float)
    group = np.asarray(group).astype(str)
    out = np.full(s.shape, np.nan)
    for g, par in GROUPS.items():
        m = group == g
        if not m.any():
            continue
        if par["state"] == "WA":
            raw = np.floor(np.round(s[m], 3) / 25.0 + 1e-9)
            out[m] = np.clip(raw, par["min"], par["max"])
        else:
            aww = np.floor(np.round(s[m], 3) / 13.0 + 0.5 + 1e-9).astype(np.int64)
            base = np.clip((48 * aww + 99) // 100, 51, 161)
            if g == "IL_dep" and total:
                base = base + np.minimum((144 * aww + 999) // 1000, 48)
            out[m] = base
    return out


def _ref_param(group: np.ndarray, ref: str, key: str) -> np.ndarray:
    """Per-unit reference score (key="s": s_cap or s_floor) or dose (key="a": max or min)."""
    name = {("s", "cap"): "s_cap", ("s", "floor"): "s_floor",
            ("a", "cap"): "max", ("a", "floor"): "min"}[(key, ref)]
    group = np.asarray(group).astype(str)
    out = np.full(group.shape, np.nan)
    for g, par in GROUPS.items():
        if par[name] is not None:
            out[group == g] = par[name]
    return out


def make_schedule(ref: str = "cap", dose: str = "relative"):
    """Return ``schedule(q, group) -> dose`` in the units of ``load_dose``."""

    def schedule(q, group):
        q = np.asarray(q, dtype=float)
        group = np.broadcast_to(np.asarray(group).astype(str), q.shape)
        s = q * _ref_param(group, ref, "s")
        a = wba_dollars(s, group)
        return a / _ref_param(group, ref, "a") if dose == "relative" else a

    return schedule


# --------------------------------------------------------------------------- frames
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m experiments.datasets.ui_upjohn.download")


def _log1p(x) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def wa_frame() -> pd.DataFrame:
    """One row per WREB claimant (17,554), with the score, the schedule and outcomes."""
    _require(WA_FILE)
    df = pd.read_stata(WA_FILE)
    qs = [f"{y}{q}" for y in range(85, 90) for q in range(1, 5)]          # 85:1 .. 89:4
    W = np.column_stack([df["wg" + q].to_numpy(float) for q in qs])
    W = np.where(W >= 9e7, 0.0, W)  # 99999999.99 = no wage record = no covered wages
    per = df["edc"].dt.to_period("Q")
    fq = ((per.dt.year - 1985) * 4 + per.dt.quarter - 1).to_numpy()
    ok = fq >= 9
    fq = np.where(ok, fq, 9)
    idx = np.arange(len(df))
    by = np.column_stack([W[idx, fq - k] for k in (5, 4, 3, 2)])  # first 4 of last 5
    top2 = np.sort(by, axis=1)[:, -2:].sum(axis=1)
    prebase = np.column_stack([W[idx, fq - k] for k in (9, 8, 7, 6)])
    post = np.column_stack([W[idx, np.minimum(fq + k, 19)] for k in (1, 2, 3)])
    late = (df["edc"] >= WA_SWITCH).to_numpy()
    ppi = df["ppi"].astype(str)
    eth = df["ethnic"].astype(str)
    out = pd.DataFrame({
        "id": "WA" + df["idno"].astype(int).astype(str),
        "state": "WA",
        "group": np.where(late, "WA_88H2", "WA_88H1"),
        "claim_date": df["edc"],
        "s": top2 / 2.0,
        "base_wages": by.sum(axis=1),
        "bw_published": df["bwages"].to_numpy(float),
        "wba_published": df["wba1"].to_numpy(float),
        "wba_bye": df["wba2"].to_numpy(float),
        "dep_allow_published": 0.0,
        "potential_weeks": df["entitle"].to_numpy(float),
        "mbp": df["mbp"].to_numpy(float),
        "bonus_arm": "WA_" + np.where(ppi == "0", "C", "T" + ppi),
        "bonus_mult": ppi.map(WA_MULT).to_numpy(float),
        "bonus": np.where(ppi == "0", 0.0, df["bonusamt"].to_numpy(float)),
        "employer_bonus": 0.0,
        "qual_period": ppi.map(WA_QUAL).to_numpy(),
        "weeks": df["totalwks"].to_numpy(float),
        "benefits": df["uitotal"].to_numpy(float),
        "spell": df["spell2"].to_numpy(float),
        "earn_post": post.mean(axis=1),
        "age": df["age"].to_numpy(float),
        "male": df["sex"].to_numpy(float),
        "black": (eth == "2").astype(float).to_numpy(),
        "hispanic": (eth == "3").astype(float).to_numpy(),
        "other_race": eth.isin(["4", "5", "6"]).astype(float).to_numpy(),
        "lag_earn": W[idx, fq - 1],
        "educ": df["educate"].to_numpy(float),
        "prebase_earn": prebase.sum(axis=1),
        "n_prebase_q": (prebase > 0).sum(axis=1).astype(float),
        "disloc_emp": df["disloc1"].to_numpy(float),
        "disloc_ind": df["disloc2"].to_numpy(float),
        "ind": df["bsic"].astype(str).to_numpy(),
        "occ": df["dot"].astype(str).str[0].to_numpy(),
        "office": df["jsc"].astype(str).to_numpy(),
        "asc": df["asc"].to_numpy(float),
        "valid": ok & (df["wba1"].to_numpy(float) > 0),
    })
    out["bw_consistent"] = np.abs(out["base_wages"] - out["bw_published"]) < 1.0
    return out


def il_frame() -> pd.DataFrame:
    """One row per Illinois experiment claimant (12,101)."""
    _require(IL_FILE)
    df = pd.read_stata(IL_FILE)
    Q = df[["l1qwages", "l2qwages", "l3qwages", "l4qwages"]].to_numpy(float)
    top2 = np.sort(Q, axis=1)[:, -2:].sum(axis=1)
    claim = pd.to_datetime(df["claimdt"].to_numpy(float), unit="D", origin="1960-01-01")
    # wgetot1..6 = 84:II .. 85:III; claim quarter index in that list
    cq = np.where(claim.quarter == 3, 1, 2)
    E = df[[f"wgetot{i}" for i in range(1, 7)]].to_numpy(float)
    idx = np.arange(len(df))
    post = np.column_stack([E[idx, cq + k] for k in (1, 2, 3)])
    dep = df["depallow"].to_numpy(float)
    arm = np.where(df["jsie"] == 1, "IL_JSIE", np.where(df["hie"] == 1, "IL_HIE", "IL_C"))
    out = pd.DataFrame({
        "id": "IL" + pd.Series(np.arange(len(df))).astype(str),
        "state": "IL",
        "group": np.where(dep > 0, "IL_dep", "IL_nodep"),
        "claim_date": claim,
        "s": top2 / 2.0,
        "base_wages": Q.sum(axis=1),
        "bw_published": 4.0 * df["avprearn"].to_numpy(float),
        "wba_published": df["wkbeneft"].to_numpy(float),
        "wba_bye": df["wkbeneft"].to_numpy(float),
        "dep_allow_published": dep,
        "potential_weeks": 26.0,
        "mbp": df["mbasr"].to_numpy(float),
        "bonus_arm": arm,
        "bonus_mult": np.nan,
        "bonus": np.where(arm == "IL_JSIE", 500.0, 0.0),
        "employer_bonus": np.where(arm == "IL_HIE", 500.0, 0.0),
        "qual_period": np.where(arm == "IL_C", "none", "11wk"),
        "weeks": df["wkspdbye"].to_numpy(float),
        "benefits": df["benpdbye"].to_numpy(float),
        "spell": df["wkpaid"].to_numpy(float),
        "earn_post": post.mean(axis=1),
        "age": df["age"].to_numpy(float),
        "male": df["male"].to_numpy(float),
        "black": df["black"].to_numpy(float),
        "hispanic": df["hispanic"].to_numpy(float),
        "other_race": ((df["natvamer"] == 1) | (df["otherace"] == 1)).astype(float).to_numpy(),
        "lag_earn": df["prepearn"].to_numpy(float),
        "educ": np.nan,
        "prebase_earn": np.nan,
        "n_prebase_q": np.nan,
        "disloc_emp": np.nan,
        "disloc_ind": np.nan,
        "ind": "na",
        "occ": "na",
        "office": "na",
        "asc": 1.0,
        "valid": True,
    })
    out["bw_consistent"] = np.abs(out["base_wages"] - out["bw_published"]) < 4.0
    return out


def pa_frame() -> pd.DataFrame:
    """Pennsylvania records file (17,513 rows; 15,005 demonstration claimants).

    No score: the WBA comes from a table on high-quarter wages, which the file lacks.
    """
    _require(PA_FILE)
    df = pd.read_stata(PA_FILE)
    tg = df["tg"]
    return pd.DataFrame({
        "id": "PA" + pd.Series(np.arange(len(df))).astype(str),
        "demo": df["dem_inel"].to_numpy(float) == 1,
        "ab_date": df["ab_dt"],
        "base_wages": df["wages"].to_numpy(float),
        "wba_published": df["wba"].to_numpy(float),
        "mba": df["mba"].to_numpy(float),
        "dependents": df["dep"].to_numpy(float),
        "bonus_arm": np.where(tg.isna(), "PA_inelig", "PA_" + np.where(tg == 0, "C", "T" + tg.fillna(0).astype(int).astype(str))),
        "bonus_mult": tg.map(PA_MULT).to_numpy(float),
        "bonus": df["bonus"].to_numpy(float),
        "weeks": df["inuidur1"].to_numpy(float),
        "age": df["r_age"].to_numpy(float),
        "male": (df["sex"] == 1).astype(float).to_numpy(),
    })


# --------------------------------------------------------------------------- samples
OUTCOMES = {
    "weeks": "weeks with UI paid in the benefit year (WA: incl. waiting week)",
    "benefits": "UI dollars paid in the benefit year",
    "spell": "weeks in the first UI spell",
    "earn_post": "mean covered earnings in the three quarters after the claim quarter",
}


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """Covariates. Common block, then state blocks (zero outside the state)."""
    wa = (df["state"] == "WA").astype(float)
    il = 1.0 - wa
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["male"] = df["male"]
    X["black"] = df["black"]
    X["hispanic"] = df["hispanic"]
    X["other_race"] = df["other_race"]
    lag = df["lag_earn"].astype(float)
    X["log_lag_earn"] = _log1p(lag.fillna(0.0))
    X["zero_lag_earn"] = (lag.fillna(0.0) <= 0).astype(float)
    X["missing_lag_earn"] = lag.isna().astype(float)
    if il.any() and wa.any():
        X["IL"] = il
    if il.any():
        X["IL_dep_allow"] = (df["group"] == "IL_dep").astype(float)
    if wa.any():
        X["educ"] = df["educ"].fillna(0.0) * wa
        X["log_prebase_earn"] = _log1p(df["prebase_earn"].fillna(0.0)) * wa
        X["n_prebase_q"] = df["n_prebase_q"].fillna(0.0) * wa
        X["disloc_emp"] = df["disloc_emp"].fillna(0.0) * wa
        X["disloc_ind"] = df["disloc_ind"].fillna(0.0) * wa
        for col, ref in (("ind", "5"), ("occ", "2"), ("office", "370")):
            for v in sorted(set(df.loc[wa == 1, col]) - {ref}):
                X[f"{col}_{v}"] = ((df[col] == v) & (wa == 1)).astype(float)
    month = df["state"] + "_" + df["claim_date"].dt.strftime("%Y%m")
    levels = sorted(month.unique())
    for v in levels[1:]:
        X[f"month_{v}"] = (month == v).astype(float)
    return X.loc[:, X.std(axis=0) > 0]


def dose_frame(states: Sequence[str] = ("WA", "IL"), asc_only: bool = True) -> pd.DataFrame:
    parts = []
    if "WA" in states:
        wa = wa_frame()
        keep = wa["valid"] & ((wa["asc"] == 1) if asc_only else True)
        parts.append(wa[keep])
    if "IL" in states:
        parts.append(il_frame())
    if not parts:
        raise ValueError("states must include 'WA' and/or 'IL' (PA has no score; see pa_frame)")
    df = pd.concat(parts, ignore_index=True)
    df["dose_dollars"] = wba_dollars(df["s"], df["group"])
    df["base_wba_formula"] = wba_dollars(df["s"], df["group"], total=False)
    df["published_dollars"] = df["wba_published"] + df["dep_allow_published"]
    return df


def load_dose(states: Sequence[str] = ("WA", "IL"), outcome: str = "weeks",
              ref: str = "cap", dose: str = "relative", asc_only: bool = True) -> RDDSample:
    """Dose-design sample: Q = s / s_ref(group), A = weekly benefit by the formula.

    dose="relative": A = benefit / benefit at the reference (cap or floor) of the
    group, so a_ref = 1. dose="dollars": A in nominal dollars (1984 IL, 1988 WA), and
    a_ref is the per-unit array of group reference levels.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    if ref not in ("cap", "floor"):
        raise ValueError("ref must be 'cap' or 'floor'")
    df = dose_frame(states, asc_only)
    if ref == "floor":
        df = df[df["group"] != "IL_dep"].reset_index(drop=True)
    df = df[np.isfinite(df[outcome])].reset_index(drop=True)
    grp = df["group"].to_numpy()
    s_ref = _ref_param(grp, ref, "s")
    a_ref_g = _ref_param(grp, ref, "a")
    Q = df["s"].to_numpy(float) / s_ref
    schedule = make_schedule(ref, dose)
    A = schedule(Q, grp)
    pub = df["published_dollars"].to_numpy(float)
    if dose == "relative":
        a_ref = 1.0
        published = pub / a_ref_g
    else:
        a_ref = a_ref_g
        published = pub
    X = _design(df)
    if ref == "cap":
        rule = lambda q: (np.asarray(q) < 1.0).astype(int)  # noqa: E731
    else:
        rule = lambda q: (np.asarray(q) >= 1.0).astype(int)  # noqa: E731
    kinks = {g: {"s_floor": p["s_floor"], "s_cap": p["s_cap"], "min": p["min"], "max": p["max"]}
             for g, p in GROUPS.items() if g in set(grp)}
    return RDDSample(
        Q=Q,
        X=X.to_numpy(float),
        Y=df[outcome].to_numpy(float),
        threshold=1.0,
        name="ui_upjohn",
        feature_names=list(X.columns),
        description=(
            "UI weekly benefit schedules (WA 1988, IL 1984 bonus experiments). Q = average "
            f"of the two highest base-period quarters / the group's {ref} kink; A = weekly "
            f"benefit by the formula ({dose}); Y = {OUTCOMES[outcome]}."
        ),
        citation=(
            "O'Leary, Spiegelman and Kline (1993), Upjohn WP 93-22 (WA); Spiegelman and "
            "Woodbury (1987), Illinois UI Incentive Experiments; public-use files from the "
            "W.E. Upjohn Institute Employment Research Data Center."
        ),
        treatment_rule=rule,
        extras={
            "A": A,
            "a_ref": a_ref,
            "a_ref_by_group": {g: float(GROUPS[g]["max" if ref == "cap" else "min"]) for g in kinks},
            "dose_levels": None,  # whole-dollar steps between floor and cap
            "schedule": schedule,
            "group": grp,
            "state": df["state"].to_numpy(),
            "Z_names": [z for z in Z_NAMES if z in X.columns],
            "published_dose": published,
            "published_dose_dollars": pub,
            "dose_dollars": df["dose_dollars"].to_numpy(float),
            "dose_match": df["dose_dollars"].to_numpy(float) == pub,
            "score_dollars": df["s"].to_numpy(float),
            "kinks": kinks,
            "ref": ref,
            "dose_side": "below" if ref == "cap" else "above",
            "bonus_arm": df["bonus_arm"].to_numpy(),
            "bonus": df["bonus"].to_numpy(float),
            "bonus_mult": df["bonus_mult"].to_numpy(float),
            "employer_bonus": df["employer_bonus"].to_numpy(float),
            "qual_period": df["qual_period"].to_numpy(),
            "potential_weeks": df["potential_weeks"].to_numpy(float),
            "bw_consistent": df["bw_consistent"].to_numpy(bool),
            "outcome": outcome,
            "outcomes": {k: df[k].to_numpy(float) for k in OUTCOMES},
            "id": df["id"].to_numpy(),
            "claim_date": df["claim_date"].to_numpy(),
        },
    )


def load() -> RDDSample:
    """Registry entry point: the default dose sample (D = 1 below the cap kink)."""
    return load_dose()


# --------------------------------------------------------------------------- checks
def rule_check() -> pd.DataFrame:
    """Share of claimants whose recorded benefit matches the schedule, by group."""
    rows = []
    wa = wa_frame()
    wa = wa[wa["valid"]]
    wa["pred"] = wba_dollars(wa["s"], wa["group"])
    for g, d in wa.groupby("group"):
        for name, m in (("all valid", d.index == d.index), ("ASC = 1", d["asc"] == 1),
                        ("base wages = published", d["bw_consistent"])):
            x = d[m]
            rows.append({"check": "WA WBA (at filing)", "group": g, "subset": name, "n": len(x),
                         "match": float((x["pred"] == x["wba_published"]).mean()),
                         "within_1": float(((x["pred"] - x["wba_published"]).abs() <= 1).mean())})
    t = wa[wa["bonus_mult"] > 0]
    rows.append({"check": "WA bonus = mult x WBA", "group": "WA", "subset": "treated", "n": len(t),
                 "match": float((t["bonus"] == t["bonus_mult"] * t["wba_published"]).mean()),
                 "within_1": np.nan})
    il = il_frame()
    base = wba_dollars(il["s"], np.where(il["group"] == "IL_dep", "IL_dep", "IL_nodep"), total=False)
    tot = wba_dollars(il["s"], il["group"])
    for g, d in il.groupby("group"):
        m = (il["group"] == g).to_numpy()
        rows.append({"check": "IL WBA (base)", "group": g, "subset": "all", "n": int(m.sum()),
                     "match": float((base[m] == d["wba_published"]).mean()),
                     "within_1": float((np.abs(base[m] - d["wba_published"]) <= 1).mean())})
        rows.append({"check": "IL WBA + dependents' allowance", "group": g, "subset": "all",
                     "n": int(m.sum()),
                     "match": float((tot[m] == d["wba_published"] + d["dep_allow_published"]).mean()),
                     "within_1": float((np.abs(tot[m] - d["wba_published"] - d["dep_allow_published"]) <= 1).mean())})
    if PA_FILE.exists():
        pa = pa_frame()
        yr = pa["ab_date"].dt.year.clip(1988, 1989).map(PA_MAX)
        rows.append({"check": "PA WBA <= max ($252 1988, $266 1989)", "group": "PA", "subset": "all",
                     "n": len(pa), "match": float((pa["wba_published"] <= yr).mean()), "within_1": np.nan})
        rows.append({"check": "PA share at max", "group": "PA", "subset": "all", "n": len(pa),
                     "match": float((pa["wba_published"] == yr).mean()), "within_1": np.nan})
        t = pa[pa["demo"] & (pa["bonus_mult"] > 0)]
        rows.append({"check": "PA bonus = mult x WBA", "group": "PA", "subset": "treated", "n": len(t),
                     "match": float((t["bonus"] == t["bonus_mult"] * t["wba_published"]).mean()),
                     "within_1": np.nan})
        rows.append({"check": "PA MBA = 26 x WBA (or 16 x)", "group": "PA", "subset": "all", "n": len(pa),
                     "match": float(((pa["mba"] == 26 * pa["wba_published"]) | (pa["mba"] == 16 * pa["wba_published"])).mean()),
                     "within_1": np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(rule_check().to_string(index=False))
    smp = load_dose()
    print(smp.summary())
