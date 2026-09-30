"""Chile retention: why gpa_next_level is missing (2017 cohort, eligible vs others).

Replicates build_continuous's cohort and outcome, then classifies each missing case by
the student's 2018 (and, where needed, 2019) records in the raw files, all statuses:
held back again, moved to an adult/special/other track, withdrew without a final grade,
no record at all, or other. Reported for all grades, grades 2-6 and grades 2-4.

    PYTHONPATH=. python experiments/scripts/chile_retention_missing_outcome.py [OUT.json]
"""
import json
import sys
import numpy as np
import pandas as pd

from experiments.datasets.chile_retention.adapter import (RAW, YOUTH_SECONDARY, _norm,
                                                          _read_year)


def level(d):
    return np.where(d.COD_ENSE == 110, d.COD_GRADO,
                    np.where(d.COD_ENSE.isin(YOUTH_SECONDARY), d.COD_GRADO + 8, np.nan))


def raw(year):
    f = list((RAW / str(year)).glob("*.csv"))[0]
    keep = {"MRUN", "SIT_FIN", "COD_ENSE", "COD_GRADO"}
    d = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, usecols=lambda c: _norm(c) in keep)
    d.columns = [_norm(c) for c in d.columns]
    d["SIT_FIN"] = d.SIT_FIN.fillna("").str.strip()
    for c in ("MRUN", "COD_ENSE", "COD_GRADO"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


now, prior = _read_year(2017), _read_year(2016)
regular = (((now.COD_ENSE == 110) & now.COD_GRADO.between(2, 8))
           | ((now.COD_ENSE == 310) & now.COD_GRADO.between(1, 3)))
now = now[regular & (now.PROM_GRAL > 0)]
now = now.join(prior[["PROM_GRAL", "ASISTENCIA"]].add_prefix("P_"), how="inner")
now = now[(now.P_PROM_GRAL > 0) & now.P_ASISTENCIA.notna()]
now["L"] = level(now)
now["elig"] = now.PROM_GRAL.round(1) <= 4.4

info = {}
for y in (2018, 2019):
    r = raw(y)
    print(y, "status counts:", r.SIT_FIN.value_counts().to_dict())
    pr = r[r.SIT_FIN.isin(["P", "R"])]
    cnt = pr.groupby("MRUN").size()
    one = pr[pr.MRUN.map(cnt) == 1].set_index("MRUN")
    one = one.assign(LVL=level(one))
    info[y] = {"lvl": one.LVL, "st": one.SIT_FIN, "dup": set(cnt[cnt > 1].index),
               "any": set(r.MRUN), "prset": set(pr.MRUN)}

ix = now.index
L = now.L
l18 = info[2018]["lvl"].reindex(ix); s18 = info[2018]["st"].reindex(ix)
l19 = info[2019]["lvl"].reindex(ix)
observed = (l18 == L + 1) | (l19 == L + 1)


def situation(y):
    """Why the student has no usable youth-track P/R record in year y."""
    i = info[y]
    lv = i["lvl"].reindex(ix)
    out = pd.Series("other level", index=ix)
    out[lv.isna() & ix.isin(list(i["prset"]))] = "adult / special / other track"
    out[ix.isin(list(i["dup"]))] = "two records (ambiguous)"
    out[~ix.isin(list(i["prset"])) & ix.isin(list(i["any"]))] = "withdrew (no final grade)"
    out[~ix.isin(list(i["any"]))] = "no record (left the system)"
    return out


sit18, sit19 = situation(2018), situation(2019)
reason = pd.Series("", index=ix)
held17 = now.SIT_FIN == "R"
# Promoted in 2017: needed L+1 in 2018 (or 2019).
m = ~observed & ~held17
reason[m] = "promoted 2017; 2018: " + sit18[m]
reason[m & (l18 == L)] = "promoted 2017; 2018: repeated level L"
# Retained in 2017: repeats L in 2018, needs L+1 in 2019.
m = ~observed & held17
at_L = l18 == L
reason[m & at_L & (s18 == "R")] = "retained 2017; retained again 2018"
reason[m & at_L & (s18 == "P")] = "retained 2017; promoted 2018; 2019: " + sit19[m & at_L & (s18 == "P")]
reason[m & ~at_L] = "retained 2017; 2018: " + sit18[m & ~at_L]
reason[observed] = "observed"

summary = {}
for lab, sub in [("All grades", np.ones(len(now), bool)),
                 ("Grades 2-6", L.between(2, 6).to_numpy()),
                 ("Grades 2-4", L.between(2, 4).to_numpy())]:
    d = now[sub]
    rs = reason[sub]
    print(f"\n=== {lab}: cohort {len(d):,}, eligible {int(d.elig.sum()):,}; "
          f"retained in 2017: eligible {d.SIT_FIN[d.elig].eq('R').mean():.3f}, "
          f"others {d.SIT_FIN[~d.elig].eq('R').mean():.4f}")
    tab = pd.DataFrame({"eligible": rs[d.elig].value_counts(normalize=True),
                        "others": rs[~d.elig].value_counts(normalize=True),
                        "n_eligible": rs[d.elig].value_counts()}).fillna(0)
    tab = tab.sort_values("eligible", ascending=False)
    print(tab.to_string(float_format="{:.4f}".format))
    grp = rs.str.replace(r"^.*retained again.*$", "held back again", regex=True)
    grp = grp.str.replace(r"^.*withdrew.*$", "withdrew (no final grade)", regex=True)
    grp = grp.str.replace(r"^.*no record.*$", "no record (left the system)", regex=True)
    grp = grp.str.replace(r"^.*adult.*$", "adult / special / other track", regex=True)
    grp = grp.str.replace(r"^.*(other level|two records|repeated level).*$", "other", regex=True)
    g = pd.DataFrame({"eligible": grp[d.elig].value_counts(normalize=True),
                      "others": grp[~d.elig].value_counts(normalize=True)}).fillna(0)
    print("  condensed:\n" + g.sort_values("eligible", ascending=False).to_string(float_format="{:.4f}".format))
    summary[lab] = {"n_eligible": int(d.elig.sum()), "n_others": int((~d.elig).sum()),
                    "eligible": g.eligible.to_dict(), "others": g.others.to_dict()}

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump(summary, f, indent=1)
