"""Chile university-application eligibility through the top-10% route (PAES 2024).

Same files and outcomes as ``chile_two_score`` (MINEDUC open data, linked by MRUN).
DEMRE's eligibility rule for admission 2024 and 2025 is
    eligible = elective test taken  and  ((reading + math-1)/2 >= 458  or  top 10% of class).
``chile_two_score`` drops the top-10% students and studies the score route. This folder
builds the other route: a cutoff on the high-school grade average, one per class.

The flag is PORC_SUP_NOTAS == 10 (MINEDUC's decile of the student within the graduating
class). Within each class = school (RBD) x teaching program (CODIGO_ENS) x graduation
year, it is a step function of the grade average PROMEDIO_NOTAS (1-7 scale, on a 0.025
grid) in every 2024 class whose registrants include flagged and non-flagged students.
NEM is a strictly increasing function of the grade average within a class, so it is
equally sharp; the ranking score is not (see ``flag_sharpness``).

The cutoffs are not published. ``classes`` brackets each class's cutoff from its
registrants in the same cohort file:
    lo = max grade among the non-flagged < cut <= hi = min grade among the flagged.
Classes with both sides observed and lo < hi are kept. The default cut is hi, so
D = 1{grade >= hi}; cut="mid" (load_mid(), load_two_score_mid()) uses (lo + hi) / 2.
Either way D equals the flag. With cut = hi, the lowest flagged registrant of each class
sits at exactly Q = 0 by construction; the midpoint removes that heap.

load():           elective takers whose PAES average is below 458, for whom the top-10%
                  route is the only way to eligibility.
                  Q = grade average - class cutoff (grade points); D = 1{Q >= 0}.
load_two_score(): all elective takers. Q = (PAES average, grade average - class cutoff);
                  D = 1{Q1 >= 458 or Q2 >= 0}, the deployed either-rule.
X = chile_two_score's covariates without NEM and the ranking score (both are functions of
    the grade average), plus north and south zone indicators.
Y = load(), load_two_score(): university enrollment in 2024 (admission 2024, same year).
Other cohorts: load_2025() and load_two_score_2025() (admission 2025, university
enrollment 2025); load_2023_univ_2024() and load_two_score_2023_univ_2024() (admission
2023, university enrollment the next year; no published eligibility flag).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.chile_admission.adapter import _num
from experiments.datasets.chile_two_score.adapter import SCORE_FILES, THRESHOLD
from experiments.datasets.chile_two_score.adapter import X_COLS as TWO_SCORE_X
from experiments.datasets.chile_two_score.adapter import build as build_two_score
from experiments.datasets.chile_two_score.adapter import rule as two_score_rule

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
GRID = 0.025                    # spacing of the grade average (shown rounded to 2 decimals)
CLASS = ["rbd", "ens", "grad_year"]
NORTH = {15, 1, 2, 3, 4}        # Arica y Parinacota to Coquimbo
SOUTH = {16, 8, 9, 14, 10, 11, 12}  # Nuble to Magallanes; the centre (5, 6, 7, 13) is the base
X_COLS = [c for c in TWO_SCORE_X if c not in ("nem", "ranking")] + ["north", "south"]
CITATION = "MINEDUC Centro de Estudios / DEMRE, PAES {} and Matricula Ed. Superior"


def _write(df: pd.DataFrame, path: Path) -> None:
    """Atomic cache write: other processes may load the registry concurrently."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def _read_raw(cohort: int) -> pd.DataFrame:
    """The registrant columns this folder needs, one row per raw record."""
    f = SCORE_FILES[cohort]
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; see experiments/datasets/chile_admission/README.md")
    cols = ["MRUN", "RBD", "CODIGO_ENS", "ANYO_DE_EGRESO", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS",
            "PTJE_NEM", "PTJE_RANKING", "PROMEDIO_CM_MAX", "CLEC_MAX", "MATE1_MAX",
            "CODIGO_REGION_EGRESO"]
    header = pd.read_csv(f, sep=";", nrows=0, encoding="utf-8-sig").columns
    if "HABILITACION_POST" in header:
        cols.append("HABILITACION_POST")
    s = pd.read_csv(f, sep=";", dtype=str, encoding="utf-8-sig", usecols=cols)
    r = pd.DataFrame({
        "mrun": _num(s.MRUN), "rbd": s.RBD.fillna("").str.strip(),
        "ens": s.CODIGO_ENS.fillna("").str.strip(), "grad_year": _num(s.ANYO_DE_EGRESO),
        "grade": _num(s.PROMEDIO_NOTAS), "decile": _num(s.PORC_SUP_NOTAS),
        "nem": _num(s.PTJE_NEM), "ranking": _num(s.PTJE_RANKING), "cm_avg": _num(s.PROMEDIO_CM_MAX),
        "cl": _num(s.CLEC_MAX), "m1": _num(s.MATE1_MAX), "region": _num(s.CODIGO_REGION_EGRESO),
        "hab_code": _num(s.HABILITACION_POST) if "HABILITACION_POST" in s else np.nan,
    })
    ok = ((r.grade > 0) & r.decile.between(10, 100) & (r.rbd != "") & (r.ens != "")
          & r.grad_year.notna())
    r["class_id"] = np.where(ok, r.rbd + "-" + r.ens + "-" + r.grad_year.fillna(0).astype(int).astype(str), "")
    return r


def _brackets(r: pd.DataFrame, score: str = "grade", keys=("class_id",)) -> pd.DataFrame:
    """Per group: registrants, flagged registrants, lo = max `score` among the non-flagged,
    hi = min `score` among the flagged, and the status of the bracket."""
    v = r[(r.class_id != "") & (r[score] > 0)]
    keys = list(keys)
    flag = v.decile == 10
    out = pd.DataFrame({"n": v.groupby(keys).size(), "n_flagged": flag.groupby([v[k] for k in keys]).sum(),
                        "lo": v[~flag].groupby(keys)[score].max(),
                        "hi": v[flag].groupby(keys)[score].min()})
    both = out.lo.notna() & out.hi.notna()
    out["status"] = np.select([both & (out.lo < out.hi), both, out.hi.notna()],
                              ["sharp", "overlap", "no_unflagged"], "no_flagged")
    out["tie_split"] = both & (out.lo == out.hi)
    return out.reset_index()


def classes(cohort: int = 2024, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per class (school x teaching program x graduation year) in one cohort file:
    registrants with a grade average and decile, flagged registrants, the cutoff bracket
    (lo, hi] on the grade average, its width in grid steps, and its status: sharp (both
    sides observed, lo < hi), overlap, no_flagged or no_unflagged."""
    cache = PROCESSED / f"classes_{cohort}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, dtype={"class_id": str})
    out = _brackets(_read_raw(cohort) if raw is None else raw)
    out["width_steps"] = ((out.hi - out.lo) / GRID).round()
    out["grad_year"] = out.class_id.str.rsplit("-", n=1).str[1].astype(int)
    _write(out, cache)
    return out


def build(cohort: int = 2024) -> pd.DataFrame:
    """chile_two_score's frame for one cohort (registrants with a valid PAES average and NEM),
    plus the grade average, the class and its cutoff bracket, the published eligibility code
    and zone indicators."""
    cache = PROCESSED / f"top10_{cohort}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, dtype={"class_id": str}, keep_default_na=False,
                           na_values=[""]).assign(class_id=lambda d: d.class_id.fillna(""))
    base = build_two_score(cohort)
    r = _read_raw(cohort)
    # chile_two_score.build's row filter, in its order, so the rows line up with `base`
    s = r[r.mrun.notna() & ~r.mrun.duplicated(keep=False) & (r.cm_avg > 0) & (r.nem > 0)]
    s = s[s[["cl", "m1", "nem", "ranking"]].notna().all(axis=1)].reset_index(drop=True)
    if len(s) != len(base) or not all(np.allclose(s[c], base[c]) for c in ["cl", "m1", "nem", "ranking"]):
        raise RuntimeError(f"rows do not line up with chile_two_score.build({cohort})")
    if not np.array_equal((s.decile == 10).to_numpy(float), base.top10.to_numpy(float)):
        raise RuntimeError("top-10% flag differs from chile_two_score's")
    cls = classes(cohort, r).set_index("class_id")
    out = base.assign(
        grade=s.grade.where(s.grade > 0).to_numpy(), class_id=s.class_id.to_numpy(),
        class_lo=s.class_id.map(cls.lo).to_numpy(), class_hi=s.class_id.map(cls.hi).to_numpy(),
        class_status=s.class_id.map(cls.status).fillna("no_class").to_numpy(),
        hab_code=s.hab_code.to_numpy(float),
        north=s.region.isin(NORTH).astype(float).to_numpy(),
        south=s.region.isin(SOUTH).astype(float).to_numpy())
    _write(out, cache)
    return out


def _top10_rule(Q: np.ndarray) -> np.ndarray:
    return (Q >= 0).astype(int)


def _either_rule(Q: np.ndarray) -> np.ndarray:
    return ((Q[:, 0] >= THRESHOLD) | (Q[:, 1] >= 0)).astype(int)


def _sample(cohort: int, y_col: str, two_score: bool = False, cut: str = "min") -> RDDSample:
    df = build(cohort)
    df = df[(df.elective == 1) & (df.class_status == "sharp") & df[y_col].notna()]
    if not two_score:
        df = df[df.cm_avg < THRESHOLD]
    class_cut = df.class_hi if cut == "min" else (df.class_lo + df.class_hi) / 2
    q2 = (df.grade - class_cut).round(4).to_numpy(float)
    if not np.array_equal(q2 >= 0, df.top10.to_numpy() == 1):
        raise RuntimeError("1{grade >= class cut} does not reproduce the top-10% flag")
    extras = {"cohort": cohort, "cut_rule": cut, "grid": GRID,
              "class_id": df.class_id.to_numpy(), "class_cut": class_cut.to_numpy(float),
              "class_bracket": df[["class_lo", "class_hi"]].to_numpy(float),
              "grade": df.grade.to_numpy(float), "paes_avg": df.cm_avg.to_numpy(float),
              "top10": df.top10.to_numpy(float)}
    if df.hab_code.notna().all():
        extras["eligible_published"] = df.hab_code.isin([1, 2]).to_numpy(float)
        extras["top10_route_published"] = (df.hab_code == 2).to_numpy(float)
    tag = "" if cohort == 2024 else f"_{cohort}"
    mid = "_mid" if cut == "mid" else ""
    cut_text = ("the lowest grade among its flagged registrants" if cut == "min"
                else "the midpoint of its bracket")
    if two_score:
        return RDDSample(
            Q=np.column_stack([df.cm_avg.to_numpy(float), q2]), X=df[X_COLS].to_numpy(float),
            Y=df[y_col].to_numpy(float), threshold=(THRESHOLD, 0.0),
            name=f"chile_top10_two_score{tag}_{y_col}{mid}", feature_names=list(X_COLS),
            description=(f"Chile PAES admission {cohort}, elective-test takers. Q = (best "
                         "reading/math-1 average, high-school grade average minus the class's "
                         f"top-10% cutoff, taken as {cut_text}); treatment = 1{{Q1 >= 458 or "
                         f"Q2 >= 0}} (university-application eligibility); Y = {y_col}."),
            citation=CITATION.format(cohort), treatment_rule=_either_rule,
            extras={**extras, "cutoffs": (THRESHOLD, 0.0), "rule": "either"},
        )
    return RDDSample(
        Q=q2, X=df[X_COLS].to_numpy(float), Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"chile_top10{tag}_{y_col}{mid}", feature_names=list(X_COLS),
        description=(f"Chile PAES admission {cohort}, elective-test takers with a reading/math-1 "
                     "average below 458. Q = high-school grade average minus the class's top-10% "
                     f"cutoff, taken as {cut_text}; treatment = 1{{Q >= 0}} (eligible through "
                     f"the top-10% route); Y = {y_col}."),
        citation=CITATION.format(cohort), treatment_rule=_top10_rule,
        extras={**extras, "cutoff": 0.0},
    )


def load() -> RDDSample:
    return _sample(2024, "univ_2024")


def load_mid() -> RDDSample:
    """As load(), with each class cutoff at the midpoint of its bracket."""
    return _sample(2024, "univ_2024", cut="mid")


def load_2025() -> RDDSample:
    """Admission 2025 cohort; university enrollment in the admission year (2025)."""
    return _sample(2025, "univ_2025")


def load_2023_univ_2024() -> RDDSample:
    """Admission 2023 cohort; university enrollment the following year (2024)."""
    return _sample(2023, "univ_2024")


def load_two_score() -> RDDSample:
    return _sample(2024, "univ_2024", two_score=True)


def load_two_score_mid() -> RDDSample:
    return _sample(2024, "univ_2024", two_score=True, cut="mid")


def load_two_score_2025() -> RDDSample:
    return _sample(2025, "univ_2025", two_score=True)


def load_two_score_2023_univ_2024() -> RDDSample:
    return _sample(2023, "univ_2024", two_score=True)


def flag_sharpness(cohort: int = 2024) -> pd.DataFrame:
    """Which score the flag is a step function of. For each candidate score and grouping,
    among groups with flagged and non-flagged registrants: the share of groups, and of their
    registrants, where max(score | not flagged) < min(score | flagged), and the number of
    groups where that max equals that min (a tie split by the flag). Reads the raw file."""
    r = _read_raw(cohort)
    r["school_year"] = np.where(r.class_id != "", r.rbd + "-" + r.grad_year.fillna(0).astype(int).astype(str), "")
    rows = []
    for keys in (["school_year"], ["class_id"]):
        for score in ("grade", "nem", "ranking"):
            b = _brackets(r.assign(class_id=r.class_id if keys == ["class_id"] else r.school_year),
                          score)
            both = b[b.status.isin(["sharp", "overlap"])]
            sharp = both.status == "sharp"
            rows.append({"grouping": "school x program x year" if keys == ["class_id"] else "school x year",
                         "score": score, "groups": len(both), "share_groups_sharp": sharp.mean(),
                         "share_students_sharp": both.n[sharp].sum() / both.n.sum(),
                         "tie_splits": int(both.tie_split.sum())})
    return pd.DataFrame(rows)


def bracket_report(cohort: int = 2024) -> dict:
    """Coverage and width of the class cutoff brackets in one cohort file."""
    c = classes(cohort)
    both = c[c.status.isin(["sharp", "overlap"])]
    sharp = c[c.status == "sharp"]
    cur = sharp[sharp.grad_year == cohort - 1]
    w_students = np.repeat(sharp.width_steps.to_numpy(), sharp.n.to_numpy())
    q = lambda w: {f"q{int(p * 100)}": float(np.quantile(w, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}  # noqa: E731
    return {
        "cohort": cohort, "classes": len(c), "registrants": int(c.n.sum()),
        "by_status": {k: {"classes": int(len(g)), "registrants": int(g.n.sum())}
                      for k, g in c.groupby("status")},
        "share_classes_sharp_of_both_sides": float((both.status == "sharp").mean()),
        "share_registrants_sharp_of_both_sides": float(both.n[both.status == "sharp"].sum() / both.n.sum()),
        "tie_splits": int(c.tie_split.sum()),
        "width_steps_classes": {**q(sharp.width_steps), "share_one_step": float((sharp.width_steps == 1).mean())},
        "width_steps_current_graduates": {**q(cur.width_steps), "classes": int(len(cur)),
                                          "share_one_step": float((cur.width_steps == 1).mean())},
        "width_steps_registrants": {**q(w_students), "share_one_step": float(np.mean(w_students == 1))},
    }


def rule_check(cohort: int = 2024) -> dict:
    """Agreement of the deployed rule with the published eligibility code HABILITACION_POST
    (1 and 2 eligible, 3 not; 2 = eligible only through the top-10% route)."""
    df = build(cohort)
    if df.hab_code.isna().all():
        return {"cohort": cohort, "published_flag": False}
    pub = df.hab_code.isin([1, 2]).to_numpy(int)
    route = ((df.elective == 1) & (df.top10 == 1) & (df.cm_avg < THRESHOLD)).to_numpy()
    y = {2024: "univ_2024", 2025: "univ_2025"}[cohort]
    one, two = _sample(cohort, y), _sample(cohort, y, two_score=True)
    return {
        "cohort": cohort, "registrants": int(len(df)),
        "rule_with_flag_vs_published_all": float(np.mean(two_score_rule(df) == pub)),
        "code2_vs_top10_route_all": float(np.mean((df.hab_code == 2).to_numpy() == route)),
        "code2_count": int((df.hab_code == 2).sum()), "top10_route_count": int(route.sum()),
        "load_n": one.n, "load_D_vs_published": float(np.mean(one.D == one.extras["eligible_published"])),
        "load_D_vs_code2": float(np.mean(one.D == one.extras["top10_route_published"])),
        "two_score_n": two.n,
        "two_score_D_vs_published": float(np.mean(two.D == two.extras["eligible_published"])),
        "two_score_mismatches": int(np.sum(two.D != two.extras["eligible_published"])),
    }
