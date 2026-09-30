# Chile grade retention — MINEDUC open student-performance files

| Field | Value |
|---|---|
| **Q** | `PROM_GRAL`, annual general grade average (1.0-7.0, reported on a 0.1 grid; 0.0 for withdrawn students) |
| **Rule** | repeat the year if 1 failed subject (< 4.0) and average < 4.5, or 2 failed subjects and average < 5.0; attendance < 85% can also retain |
| **Treatment** | `SIT_FIN == "R"` (reprobado); `P` promoted, `Y` withdrawn |
| **X (available)** | prior-year `PROM_GRAL`/`ASISTENCIA`/`SIT_FIN` via `MRUN`, grade (`COD_GRADO`, `COD_ENSE`), gender, birth year-month, age, school (`RBD`), dependency, rural, commune |
| **Not in public file** | subject grades / number of failed subjects, so the design is fuzzy in `PROM_GRAL` |
| **n** | about 3.3 million student records per year, 2002-2025 |
| **Source** | https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/ (public, no registration) |

## Getting the data

One RAR per year, 2002-2025: 37-48 MB each, about 1.06 GB compressed in total, and
about 11 GB of CSV once extracted. Download and extract with the idempotent script
(uses the macOS `bsdtar`, libarchive >= 3.4):

```bash
python -m experiments.datasets.chile_retention.download            # all years
python -m experiments.datasets.chile_retention.download 2017 2018  # a subset
```

The script checks each file against the server's size before extracting. All 24 years
were downloaded on 2026-09-28. Reported record counts match the ministry schema
(`ER Rendimiento por alumno, bases Web.pdf`, inside each archive) for every year it
lists (2002-2020).

Each archive contains the CSV (`;`-separated, decimal comma), the variable schema, and
(in most years) a usage-recommendations PDF and a frequency workbook.

Parsing notes:
- Encodings vary: ASCII, UTF-8 with or without BOM, and Latin-1 (2015). Header case also
  varies. Read with `encoding="latin-1"` and normalize headers by stripping the BOM and
  uppercasing (see `experiments/scripts/chile_retention_profile.py`).
- `MRUN` repeats within a year for students who transferred (about 209k duplicate rows
  in 2018; see `SIT_FIN_R == "T"`).
- Filter `ESTADO_ESTAB == 1` for operating schools (the variable exists from 2015).

## First look (2018; `SIT_FIN` in {P, R} and `PROM_GRAL > 0`; n = 2,998,611)

Share retained by grade average:

| Average | 4.3 | 4.4 | 4.5 | 4.6 | 4.9 | 5.0 | 5.1 |
|---|---|---|---|---|---|---|---|
| Share retained | 0.804 | 0.731 | 0.355 | 0.295 | 0.094 | 0.021 | 0.012 |
| Count | 10,860 | 13,315 | 26,425 | 31,726 | 70,254 | 91,550 | 104,117 |

- There are clear first-stage drops at 4.5 (about 38 points) and 5.0 (about 7 points).
- The count doubles from 4.4 to 4.5, far above the neighboring trend. That is heaping at
  the cutoff, which suggests grade nudging and/or rounding. It must be checked before any
  RD use; it is also relevant to the performative-threshold direction.
- Linkage: 90.6% of 2017 promoted or retained students appear in 2018; the rest include
  graduates and leavers. 82.7% of retained students are in the same grade next year, and
  0% of promoted students are.

## All years (`python experiments/scripts/chile_retention_profile.py OUT.csv`)

Heap = count at the cutoff / geometric mean of the counts 0.1 below and above it. Sample:
`SIT_FIN` in {P, R}, `PROM_GRAL > 0`.

| Year | Records | Retained | Heap at 4.5 | Retained 4.4 → 4.5 | Heap at 5.0 | Retained 4.9 → 5.0 |
|---|---|---|---|---|---|---|
| 2002 | 3,376,045 | 140,769 | 1.41 | 0.55 → 0.24 | 1.10 | 0.050 → 0.014 |
| 2003 | 3,528,762 | 181,531 | 1.43 | 0.63 → 0.27 | 1.11 | 0.055 → 0.014 |
| 2004 | 3,515,838 | 188,363 | 1.43 | 0.64 → 0.26 | 1.10 | 0.053 → 0.012 |
| 2005 | 3,506,407 | 199,293 | 1.45 | 0.64 → 0.26 | 1.10 | 0.054 → 0.012 |
| 2006 | 3,465,584 | 213,661 | 1.41 | 0.66 → 0.28 | 1.11 | 0.056 → 0.013 |
| 2007 | 3,413,110 | 200,135 | 1.41 | 0.66 → 0.27 | 1.10 | 0.054 → 0.012 |
| 2008 | 3,356,256 | 202,512 | 1.39 | 0.69 → 0.29 | 1.09 | 0.061 → 0.014 |
| 2009 | 3,326,575 | 186,466 | 1.39 | 0.68 → 0.29 | 1.09 | 0.060 → 0.012 |
| 2010 | 3,335,825 | 187,724 | 1.40 | 0.71 → 0.30 | 1.08 | 0.065 → 0.012 |
| 2011 | 3,326,746 | 221,422 | 1.40 | 0.72 → 0.31 | 1.08 | 0.072 → 0.019 |
| 2012 | 3,308,477 | 177,758 | 1.37 | 0.71 → 0.30 | 1.07 | 0.062 → 0.013 |
| 2013 | 3,255,518 | 168,382 | 1.38 | 0.72 → 0.31 | 1.08 | 0.073 → 0.016 |
| 2014 | 3,227,534 | 152,919 | 1.35 | 0.72 → 0.33 | 1.07 | 0.073 → 0.016 |
| 2015 | 3,238,586 | 152,359 | 1.33 | 0.73 → 0.35 | 1.09 | 0.080 → 0.018 |
| 2016 | 3,226,943 | 144,895 | 1.32 | 0.75 → 0.36 | 1.08 | 0.086 → 0.019 |
| 2017 | 3,246,824 | 130,510 | 1.30 | 0.74 → 0.36 | 1.08 | 0.092 → 0.022 |
| 2018 | 3,293,750 | 121,004 | 1.29 | 0.73 → 0.36 | 1.07 | 0.094 → 0.021 |
| 2019 | 3,328,915 | 98,269 | 1.28 | 0.69 → 0.33 | 1.09 | 0.084 → 0.019 |
| 2020 | 3,164,534 | 58,664 | 1.35 | 0.03 → 0.01 | 1.16 | 0.003 → 0.001 |
| 2021 | 3,237,043 | 89,557 | 1.51 | 0.20 → 0.07 | 1.07 | 0.016 → 0.004 |
| 2022 | 3,405,130 | 79,746 | 1.25 | 0.59 → 0.31 | 1.10 | 0.084 → 0.019 |
| 2023 | 3,584,330 | 81,660 | 1.25 | 0.62 → 0.34 | 1.11 | 0.083 → 0.018 |
| 2024 | 3,568,930 | 77,281 | 1.23 | 0.64 → 0.35 | 1.12 | 0.086 → 0.018 |
| 2025 | 3,540,980 | 75,620 | 1.21 | 0.64 → 0.35 | 1.11 | 0.085 → 0.017 |

- **Heaping at 4.5 in every year:** 21-51% more students than the neighboring counts
  imply, and about 7-16% at 5.0. It is a stable feature of grading, not a one-year
  artifact.
- **The rule's bite changes over time.** The retention jump at 4.5 is about 0.3-0.4
  through 2019. The 2020 and 2021 school years (COVID) nearly suspended retention: only
  2.5% retained at 4.4 in 2020. From 2022 the jump returns, but retention at 4.4 stays
  lower than in 2010-2019 (0.59-0.64 vs 0.69-0.75). That fits schools having more
  discretion over promotion from about 2019 (the rule change itself is not verified here).
  The pre-2019 years are the cleanest for a fixed rule.
- Withdrawn (`Y`) counts rise from about 190k to about 300k around 2010-2011, possibly a
  change in how withdrawals are recorded. Check this before using withdrawal as an outcome.
- `MRUN` is never missing, and every average is on the 0.1 grid.

## Adapter and pre-screen (2026-09-28)

`adapter.py` builds one cohort (default decision year 2017, cached in
`data/processed/cohort_2017.csv.gz`). It links X from 2016 and Y from 2018 by `MRUN`
and uses regular primary grades 2-8 and academic secondary grades 1-3. `load()` has
Y = next-year average; `load_enrolled()` has Y = completes next year.

- n = 2,167,129 students; 1.8% have Q <= 4.4.
- First-stage R^2 = 0.68.
- The retained share falls from 0.84 at 4.4 to 0.41 at 4.5.
- D is eligibility (intent-to-treat).

Screen (`python -m experiments.scripts.differing_slopes_screen --datasets
chile_retention chile_retention_enrolled --out ...`), c = 0, trim eps 0.10. Flatness from
`PYTHONPATH=. python experiments/scripts/screen_flatness.py OUT.json chile_retention
chile_retention_enrolled`. Entries are the optimum [share of the trim window treated].
The window holds 3.4-3.9% of students, with window Q from about 4.1 to 5.4.

| Outcome | alpha-only U_J | Diff. slopes U_J | Mean effect in window (alpha-only / DS) | Share of window with negative effect (DS) |
|---|---|---|---|---|
| Next-year average | 6.89 [100%] (bnd) | 5.87 [98%] | +0.36 / +0.20 points | 18% |
| Completes next year | 4.00 [4%] (bnd) | 6.89 [100%] (bnd) | -7.1 / +1.5 pp | 38% |

- **Next-year average:** both models treat essentially the whole window. The DS interior
  optimum beats treating everyone by 0.0004 points per window student, so it is
  effectively a boundary. The outcome is also mechanical, because retained students
  repeat the grade.
- **Completing next year:** the models disagree in sign. Alpha-only gives -7.1 pp for
  everyone, so treat no one; DS gives +1.5 pp on average (10th-90th percentile -11 to
  +12 pp), so treat everyone. The flip means DS leans on D x X extrapolation. Prior
  average in X likely makes X ⊥ eta fail. Do not read either as a finding.
- **Open issues before any inference-grade run:**
  - heaping at 4.5 (the first untreated value);
  - the second cutoff at 5.0 sits inside the window's Q range and is not modeled;
  - D is eligibility, not retention;
  - longer-run outcomes (secondary completion, PAES, higher-education enrollment) are
    available in the public data by `MRUN` and are more policy-relevant.

## Checks after the pre-screen (2026-09-28)

**Long-run cohort.** `build_long_run(2012)` uses decision year 2012 with X from 2011.
It covers primary 6-8 and youth secondary 1-3, including technical-vocational tracks.
The outcome is `completed_secondary`: promoted from regular 4th-year secondary or from
the final adult level, in any year 2013-2025. Loaders: `load_completed()` and
`load_completed_donut()`, which drops Q = 4.5.
- n = 1,441,489; 4.7% have Q <= 4.4; first-stage R^2 = 0.60.
- Older files lack age and grouped school type. The adapter derives age from the birth
  date (YYYYMM in recent files, YYYYMMDD in older ones) and school type from `COD_DEPE`.

**Validation against the cutoff itself** (`experiments/scripts/chile_retention_local_checks.py`;
effect columns from `screen_flatness.py`). The first two columns are local linear RD ITT
estimates at 4.5, bandwidth 0.5, with robust SEs. The next two are the models' mean fitted
effect for students within 0.5 of the cutoff; both are intent-to-treat quantities.

| Outcome | Local RD ITT | Local RD ITT, donut | alpha-only effect near cutoff | DS effect near cutoff | Optimum: alpha-only / DS |
|---|---|---|---|---|---|
| Next-year average (2017) | +0.146 (0.008) | +0.144 (0.008) | +0.31 | +0.23 | treat all / treat 98% |
| Completes next year (2017) | -0.003 (0.004) | -0.007 (0.004) | -0.089 | -0.035 | treat none / treat all |
| Completes secondary (2012) | -0.004 (0.004) | -0.011 (0.004) | -0.132 | -0.073 | treat none / treat none |

- **The global models overstate the local effect by roughly 2x to 40x.** Differing
  slopes is closer than alpha-only, but neither is validated at the cutoff.
- **X ⊥ eta fails clearly.** SD(eta) rises from about 0.21 to 0.45 (2017) and from 0.30
  to 0.53 (2012) across deciles of the fitted index. Mean eta is inverted-U (about
  -0.06 at both ends, +0.03 in the middle), so the linear index also misses curvature.
- **Sorting at the cutoff.** Just above 4.5, students are more often previously retained,
  older, in municipal schools and in lower grades (2017: prior average jump +0.060
  (0.006), prior retention -0.018 (0.004), municipal -0.042 (0.006)). The jumps survive
  dropping Q = 4.5, so they are not confined to the heaped value. This is consistent
  with teachers promoting students they do not want to retain again.
- **Heterogeneity by level (2012, ITT on completion):** secondary -0.019 (0.005);
  primary 6-8 +0.005 (0.007). Next-year enrollment: -0.015 (0.005) vs +0.017 (0.005).
  This is a real sign change across X, but small.
- **Verdict:** not a usable PerfRDD application as specified. Every optimum is at a
  boundary; the maintained X ⊥ eta assumption fails; the global fits are not validated
  at the cutoff; and the cutoff shows sorting. The heaping and sorting may still serve as
  evidence for the performative-threshold direction.

## Continuous outcomes (2026-09-29)

`build_continuous(year)` has the same sample and X as `build_cohort` plus four outcomes:
- next-year attendance (%);
- the average two years later;
- the average in the next grade level the first time it is reached (the same curriculum
  for both groups, but observed only for students who reach it);
- the first PAES reading/math-1 average, 2023-2025, for grade levels 6-7. Test takers
  only; PAES files are read from `../chile_admission/data/raw/`.

Loaders: `load_attendance_next`, `load_gpa_two_years`, `load_gpa_next_level`,
`load_paes`, and the robustness loaders `load_gpa_next_level_donut`, `_2016`, `_2015`.
Outputs: `outputs/screen_chile_retention_continuous_20260929/`.

Differencing grades does not help:
- Y minus this year's average gives exactly the same local and global estimates, because
  Q is continuous at the cutoff and is absorbed by b(eta) + X'beta.
- Y minus last year's average only nets out the sorting jump in prior grades.

The real issue is that held-back students repeat material, which is why the
same-grade-level outcome is included.

Cohort 2017. Local RD ITT at 4.45 (bandwidth 0.5) vs the models' fitted effect within
0.5 of the cutoff:

| Y | Local ITT (SE) | alpha-only near / optimum | DS near / optimum [share of window treated] | DS gain over better boundary |
|---|---|---|---|---|
| Next-year attendance | +0.79 (0.16) | -0.58 / treat none | -0.72 / treat all | - (wrong sign in both) |
| Average two years later | +0.122 (0.009) | +0.29 / treat all | +0.21 / 5.60 [89%] | 0.004 (flat) |
| Average in the next grade level | +0.219 (0.008) | +0.39 / treat all | +0.27 / **5.34 [74%]** | **0.026** |
| PAES (levels 6-7) | +4.6 (3.4) | +49 / treat all | +48 / 4.93 [45%] | 6.6 points |

- **PAES:** the interior optimum comes from a model that overstates the local effect about
  10x, so it is not credible. Eligibility also raises test-taking by +0.032 (0.010).
- **Average in the next grade level is the one credible candidate.**
  - DS interior optimum at 5.34 (treat Q <= 5.3). Stable: donut 5.36; 2016 cohort 5.31;
    2015 cohort 5.20.
  - Gain over retaining the whole window: 0.022-0.045 grade points per window student.
    Gain over the deployed cutoff: 0.19 vs 0.08 in 2017.
  - DS near-cutoff effect is +0.27 / +0.23 / +0.23 (2017/2016/2015) vs local
    +0.219 / +0.199 / +0.197: 15-25% high, the closest agreement in any real dataset.
    alpha-only is about 1.7x high.
  - By prior-average tercile near the cutoff, DS gives +0.31 / +0.24 / +0.26 vs local
    +0.21 / +0.20 / +0.21.
  - **Second-cutoff check.** At the 5.0 cutoff (two failed subjects) the local Wald is
    +0.410 per retained student (first stage 0.036). DS near Q = 4.95 implies +0.409 per
    retained student, scaling by the 4.5 first stage (0.388). alpha-only implies +0.82.
    This is suggestive only: the compliers differ between the two cutoffs.
  - **Caveats:**
    - c = 0 ignores the cost of an extra school year, so any positive cost moves the
      optimum down.
    - The outcome is observed only for students who reach the next grade by t+2; held-back
      students are 7 pp less likely to.
    - Sorting at 4.5 and SD(eta) doubling across T (X ⊥ eta) still hold.
    - D is eligibility, not retention.

**Correction (2026-09-29, Claude): at-cutoff validation.** The "15-25% high" figures
above compare the model's mean effect over the whole ±0.5 band with the local RD. The
band is count-weighted toward Q = 4.5-4.9, where the model's effects are smaller, so the
comparison flatters the model.

The earlier Wald values (+0.565 at 4.5, +0.410 at 5.0) also divided the outcome ITT
(students who reach the next level) by a first stage from all students. With both on the
same sample (`experiments/scripts/chile_retention_curves.py`, `curves.npz`), per retained
student:

| | Local Wald (95% CI) | DS at Q = 4.4-4.5 (or 4.9-5.0) | alpha-only |
|---|---|---|---|
| Cutoff 4.5 | +0.53 (0.49, 0.57) | +0.97 | +1.10 |
| Cutoff 5.0 | +0.46 (0.26, 0.65) | +0.43 | +0.81 |

So DS overstates the effect at the main cutoff by about 1.8x, and matches at 5.0 only
within a wide interval. The interior optimum near 5.3 is driven by DS's steep decline in
the effect (0.97 to 0.43 between the cutoffs, zero near 5.55). The local points decline
far less (0.53 to 0.46), though the 5.0 interval cannot exclude the steep slope. The
optimum is therefore **not validated**. Figures:
`outputs/screen_chile_retention_continuous_20260929/retention_curves*.png`.

**Restricting to one failed subject is not possible with public data** (2026-09-29).
Subject grades are not in the files. The closest observable restriction drops the
attendance route to retention: attendance >= 85% in the decision year
(`load_gpa_next_level_att85`, 90% of students). It changes almost nothing:
- Retained share: 0.84 at 4.4 and 0.40 at 4.5 (vs 0.85 and 0.45 below 85% attendance).
- Local Wald: +0.50 [0.46, 0.55] at 4.5 and +0.46 [0.25, 0.66] at 5.0.
- DS optimum: 5.34 [73%].
- DS per retained at Q = 4.4/4.5: 0.93, still about 1.85x the local estimate; at
  4.9/5.0: 0.47.

The fuzziness at the cutoff comes from the unobserved failed-subject count. Just below
4.5, the promoted 16% failed no subject. From 4.5 to 5.0, the retained 9-40% failed two
or more. Attendance is not the source.
