# Massachusetts Chapter 70 school aid: the yearly aid increase per pupil as a dose

| Field | Value |
|---|---|
| **Q** | The score: the foundation gap per pupil of district d in fiscal year t, (foundation aid_t - base aid_t) / foundation enrollment_t, in dollars. Foundation aid = foundation budget - required local contribution; base aid = last year's Chapter 70 aid. Negative for "hold-harmless" districts whose aid already exceeds their formula aid. 1-99%: -$3,781 to +$1,386 |
| **Threshold** | m_t, the year's minimum aid increase per pupil: $25 (FY2016), $55 (FY2017), $30 (FY2018-20, FY2022), $0 (FY2021), $60 (FY2023-24), $104 (FY2025), $150 (FY2026), $160 (FY2027) |
| **Treatment** | Dose `extras["A"]` = (Chapter 70_t - base aid_t) / enrollment_t = max(Q, m_t) dollars per foundation pupil. Reference dose `a_ref` = $30. `load()`'s D = 1{Q > 30} |
| **Rule class** | A floor (censored line): A = m_t below the kink at Q = m_t, A = Q above it. Twelve fiscal-year schedules that differ only in the floor m_t |
| **X** | log foundation enrollment; low-income (economically disadvantaged FY2015-21), EL, disability, Hispanic, Black, Asian, multiracial shares; K-5 and 9-12 shares; regional flag; latitude / longitude of the member towns; lagged outcome (+ missing flag); last year's score Q_{t-1}; fiscal-year and county indicators (37 columns) |
| **Y** | MCAS grades 3-8, percent meeting or exceeding expectations, ELA (default) or math, spring of FY t+1 (`outcome="ela"`, `"math"`). Options: average scaled score (`"ela_ss"`, `"math_ss"`), spring of t (`"ela_t0"`), in-district spending per pupil in FY t (`"ppx"`) |
| **n** | 2,414 district-years (289 districts), FY2016-18 and FY2020-25 (FY2019's t+1 test, spring 2020, was cancelled); 2,268 (288 districts) in the estimator run after trimming |
| **Source** | DESE Chapter 70 "Complete formula spreadsheet" FY2016-FY2027 (sheet `Aid436`; FY2016-22 from the Internet Archive), "Chapter 70 Trends" (keyfactors.xlsx), DESE white papers and FY pages; E2C hub MCAS, enrollment and spending files; Census 2023 Gazetteer |

**Fit: weak.** The rule is clean and the score is well predicted, but the sample is small
and the doses above the floor are few, so the transported dose response is unstable.
- **The rule is reproduced.** Published aid equals formula (1) within $1 for 83-100% of
  operating districts in each year. Every miss is a district with a published overlay
  column in the same sheet (hold-harmless to an earlier budget run, transitional
  low-income relief, the SOA minimum aid adjustment). The foundation-aid district counts
  match DESE's text in every year checked.
- **The reference is held by many districts.** 71% of the sample sits at its year's floor
  (26% at exactly $30). The rest get A = Q: 532 district-years above $104, 196 above $500.
- **X predicts Q.** In the estimator sample R^2 = 0.90 with last year's score in X (0.39 without it); sd(eta) = $281
  against sd(T) = $846. Districts with the same dose differ in eta: 10-90% range about
  $550-920 in every dose decile.
- **Second schedule.** Yes: nine fiscal years with floors $0-104 in the outcome window.
  The score-direction modifier is still 97% absorbed by f (`r2_absorbed` = 0.97), so the
  test has little power.
- **The estimates are unstable.** At a = $250 the effect on ELA is +0.5 to +2.7 points
  (rotation sd 0.2-0.6) with seed 0, but +0.2 to +1.4 and -0.2 to +1.3 with seeds 1 and 2.
  At a >= $500 the rotation sd is 1.1-58 points. A linear dose term gives +0.22 points
  of ELA per $100 (cluster se 0.10) and +0.35 of math (0.11). No pass-through to
  spending per pupil is detected.

## The rule

DESE white papers, FY2023-FY2027, "Chapter 70 aid" section (`data/raw/ch70_fy*/chapter-*-whitepaper.docx`).
FY2024 wording: "In FY24, the Chapter 70 aid calculation begins with each district's FY23
Chapter 70 amount. If the sum of that amount and the required local contribution is less
than the district's foundation budget, then foundation aid is added to cover the gap; 203
districts receive foundation aid in FY24. Otherwise, districts receive minimum aid
increases of $60 per pupil." The FY pages add that the minimum aid "guarantees all
districts receive at least the same amount of aid in FY2024 as they did in FY2023 plus a
$60 per pupil increase", i.e. a top-up. Together, for every operating district:

    Chapter 70_t = base_t + max(FA_t - base_t, m_t x enr_t),   FA_t = FB_t - RLC_t       (1)

Dividing by enrollment: A = max(Q, m_t).

**Parameters by year.** Sources: DESE FY pages (FY2023-27 live:
https://www.doe.mass.edu/finance/chapter70/fy2024/chapter-2024.html etc.; FY2015-22 via the
Internet Archive, e.g. https://web.archive.org/web/20190918135549/http://www.doe.mass.edu/finance/chapter70/fy2016/chapter-16.html).
The adapter also checks m_t against the "Min Aid Per Pupil" header cell of each
year's spreadsheet.

| FY | m_t | DESE text | Effort reduction | Overlays in the sheet |
|---|---:|---|---:|---|
| 2016 | 25 | "Every operating district receives at least $25 in additional aid per student" | 50% | none bind |
| 2017 | 55 | "... at least $55 ..."; 25 districts held harmless to the low-income change | 85% | hold-harmless (low income) |
| 2018 | 30 | "... at least $30 ..."; 15 districts get transitional relief | 85% | relief (econ. disadv.) |
| 2019 | 30 | "... at least $30 ..." | 100% | relief; adjustment to House budget H.4401 |
| 2020 | 30 | "... at least $30 ..." | 100% | hold-harmless to the House budget (75 districts) |
| 2021 | 0 | "increases for inflation and enrollment that will keep all school districts at foundation" (no minimum aid) | 100% | hold-harmless to the 7/30/2020 run |
| 2022 | 30 | "Minimum aid: $30 per pupil" | 100% | SOA minimum aid adjustment; hold-harmless to H1 run |
| 2023 | 60 | "$60/pupil" | 100% | minimum aid adjustment |
| 2024 | 60 | "$60 per pupil" | 100% | minimum aid adjustment, hold-harmless |
| 2025 | 104 | "$104 per pupil" | 100% | hold-harmless (7) |
| 2026 | 150 | "$150 per pupil" | 100% | none bind |
| 2027 | 160 | "$160 per pupil" | 100% | minimum aid adjustment (1) |

**Spreadsheet columns used** (sheet `Aid436`, 0-based, `adapter.COLS`; the header text of
each column is checked on reading):
- FY2016-19: base 4, foundation budget 5, required contribution 6, foundation aid 7,
  foundation enrollment 17, Chapter 70 34 (FY2016) or 35.
- FY2020: base 6, FB 7, RLC 8, FA 9, enrollment 19, Chapter 70 37, House hold-harmless amount 54.
- FY2021: base 6, FB 7, RLC 8, FA 9, enrollment 13, Chapter 70 38.
- FY2022-27: base 7, FB 8, RLC 9, FA 10, enrollment 14, Chapter 70 24.
- Column 2 ("Non-Op") = 1 marks an operating district.

**Checks of the parse.**
- **Foundation-aid counts.** The number of operating districts with FA > base equals
  DESE's text: 83 (FY2016), 101 (FY2017), 97 (FY2018), 133 (FY2019), 144 (FY2020), 113
  (FY2021), 203 (FY2024), 120 (FY2025), 96 (FY2026), 62 (FY2027). FY2023's 205 counts
  non-operating districts too; operating only: 188.
- **Against Chapter 70 Trends** (`keyfactors_check()`). Enrollment, FB, RLC and Chapter 70
  agree to the dollar for every operating district-year FY2016-2026, except RLC for 2 of
  319 in FY2026.

**The required contribution** (not needed for the dose, only for the score) moves from
last year's contribution times the municipal revenue growth factor toward a wealth
target. The target is 59% of the statewide foundation budget, split equally between
property (equalized valuation) and income. It is capped at 82.5% of foundation, and
municipalities above 175% must pay at least 82.5%. Since FY2019, 100% of any excess above
target is removed.

**The alternative design** (Q = the local share RLC / FB, A = aid / FB) is not a
schedule. Aid equals FB - RLC only for the foundation-aid districts: 9-60% of
district-years by year (`share_aid_eq_FB_minus_RC` below). Everywhere else aid is history
(base aid plus the minimum).

## Rule check (2026-11-12)

`adapter.rule_check()` covers operating districts. "Exact" means published Chapter 70 =
(1) within $1.

| FY | m_t | n | Exact | Within $1/pupil | Misses | Misses explained by an overlay column | At the floor | Aid = FB - RLC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 | 25 | 322 | 1.000 | 1.000 | 0 | 0 | 0.764 | 0.236 |
| 2017 | 55 | 322 | 0.922 | 0.922 | 25 | 25 | 0.755 | 0.186 |
| 2018 | 30 | 322 | 0.953 | 0.953 | 15 | 15 | 0.730 | 0.239 |
| 2019 | 30 | 320 | 0.856 | 0.916 | 46 | 46 | 0.609 | 0.300 |
| 2020 | 30 | 318 | 0.830 | 0.852 | 54 | 54 | 0.572 | 0.261 |
| 2021 | 0 | 318 | 0.855 | 0.906 | 46 | 46 | 0.645 | 0.211 |
| 2022 | 30 | 318 | 0.868 | 0.921 | 42 | 42 | 0.783 | 0.094 |
| 2023 | 60 | 318 | 0.997 | 0.997 | 1 | 1 | 0.447 | 0.553 |
| 2024 | 60 | 319 | 0.984 | 0.984 | 5 | 5 | 0.389 | 0.599 |
| 2025 | 104 | 319 | 0.978 | 0.981 | 7 | 7 | 0.718 | 0.263 |
| 2026 | 150 | 319 | 1.000 | 1.000 | 0 | 0 | 0.765 | 0.235 |
| 2027 | 160 | 319 | 0.997 | 0.997 | 1 | 1 | 0.865 | 0.132 |

`load_dose(on_rule=True)` (default) keeps only the exact district-years.

**Bunching at the kink** (`bunching()`): operating district-years with Q in [m - $25, m)
vs [m, m + $25) are 7/4, 12/7, 9/10, 8/11, 7/7, 8/14, 8/7, 6/4, 2/7, 8/6, 2/2, 1/1
(FY2016-27). There is no excess mass on either side. Districts do not set Q: it follows
from October enrollment, the state's rates and DOR's wealth data.

## Dose design

`load_dose(outcome="ela", years=(2016, 2017, 2018, 2020, ..., 2025), on_rule=True, score_lag=True)`

| Key | Content |
|---|---|
| `Q`, `threshold` | foundation gap per pupil ($); 30 |
| `extras["A"]`, `["a_ref"]`, `["dose_levels"]` | max(Q, m_t); 30.0; None (continuous) |
| `extras["schedule"]` | `adapter.schedule(q, group)` = max(q, m_group); reproduces `A` exactly |
| `extras["group"]`, `["m"]`, `["at_min"]` | fiscal year; its floor m_t; Q < m_t |
| `extras["Z_names"]` | `log_enroll`, `share_lowinc`, `share_el`, `y_lag` |
| `extras["lea_id"]` | DESE LEA number (cluster on it; districts repeat up to 9 times); `["dist_code"]` 8-digit org code |
| `extras["A_published"]`, `["exact"]` | (published Chapter 70 - base) / enrollment; rule holds |
| `extras["base_pp"]`, `["fb_pp"]`, `["target_aid_pct"]` | base aid and foundation budget per pupil; the district's wealth-based target aid share (in the panel, not in X) |
| `extras["ppx"]`, `["df"]` | in-district spending per pupil, FY t; the panel rows |

**Readiness.** From `readiness()` and the estimator log; eta is the residual of the OLS
fit of Q on (1, X).

| Quantity | Default sample (n = 2,414) | Estimator run (trimmed, n = 2,268) | Without Q_{t-1} (trimmed) |
|---|---:|---:|---:|
| R^2 of Q on X | 0.843 | 0.900 | 0.393 |
| sd(Q), sd(T), sd(eta) | 1,078, 990, 426 | -, 846, 281 | -, 526, 654 |
| Share at a floor / at $30 | 0.705 / 0.252 | 0.715 / 0.259 | same |

eta by dose (estimator run): at $30, 10/50/90% = -346 / -25 / 236. In the dose deciles
above $104 the medians rise from +57 to +205, and the 10-90% ranges are -460..362,
-273..312, -276..478 and -271..650 for $104-124, $124-288, $288-549 and $549-1,370.

Doses by year: the share at the floor is 0.77, 0.81, 0.77, 0.73, 0.75, 0.89, 0.48, 0.42 and
0.75 (FY2016-18, 2020-25). The SOA years FY2023-24 carry most doses above $500.

## Estimator results (2026-11-12)

Config `dose_config.py`.
- **Sample.** Score trimmed at 1/99%; the 2% tails of the full-sample first-stage residual
  dropped.
- **Dose basis.** A cubic in A on [$0, $722] with B($30) = 0 (l = 0).
- **Grids.** Score bins in dollars; reference $30; clustered by district.
- **Logs.** `experiments/runs/dose_schedule/ma_chapter70.log` (ELA) and
  `ma_chapter70_math.log`.

ELA percent meeting or exceeding, spring t+1 (mean 48.3, sd 14.8), seed 0. Entries are
the effect vs $30 (rotation sd, coverage):

| Score bin ($) | a = 0 | 60 | 104 | 250 | 500 | 750 |
|---|---|---|---|---|---|---|
| -4000 to -1000 | 0.24 (0.44, 92%) | -0.03 (0.27, 94%) | 0.22 (0.48, 94%) | 1.49 (0.63, 88%) | -8.6 (12.7, 81%) | -50.4 (58.5, 89%) |
| -1000 to -500 | 0.16 (0.34, 94%) | -0.03 (0.21, 97%) | 0.10 (0.32, 97%) | 0.96 (0.35, 94%) | -3.2 (2.8, 92%) | -22.8 (17.8, 96%) |
| -500 to -250 | 0.10 (0.39, 93%) | 0.02 (0.25, 95%) | 0.19 (0.40, 97%) | 0.97 (0.30, 91%) | -4.6 (2.9, 91%) | -27.1 (17.9, 94%) |
| -250 to 0 | 0.16 (0.48, 95%) | -0.04 (0.31, 96%) | 0.06 (0.51, 97%) | 0.88 (0.20, 94%) | -2.4 (4.2, 93%) | -18.6 (23.6, 96%) |
| 0 to 250 | 0.06 (0.24, 93%) | 0.01 (0.18, 95%) | 0.11 (0.37, 96%) | 0.51 (0.56, 92%) | -3.4 (2.5, 93%) | -17.7 (13.4, 95%) |
| 250 to 500 | -0.16 (0.50, 89%) | 0.21 (0.32, 90%) | 0.58 (0.54, 92%) | 1.76 (0.31, 89%) | -0.4 (1.6, 90%) | -11.5 (17.3, 96%) |
| 500 to 1000 | -0.53 (0.48, 77%) | 0.46 (0.34, 81%) | 1.00 (0.63, 85%) | 2.02 (0.64, 77%) | 2.5 (1.8, 83%) | 5.1 (12.0, 98%) |
| 1000 to 2500 | -0.72 (0.54, 72%) | 0.63 (0.37, 76%) | 1.39 (0.66, 78%) | 2.65 (0.28, 70%) | 1.1 (1.1, 74%) | -1.8 (10.3, 89%) |

**Modifier test (ELA).** Coefficient on (A - 30) T_hat = 3.8e-6 (se 6.2e-6), i.e. 0.04
points (se 0.06) per $100 of dose x $100 of T. `r2_absorbed` = 0.971, n_fit = 1,449. Not
rejected. Math: 1.4e-6 (se 6.9e-6), `r2_absorbed` 0.971.

**Math** (seed 0). At a = $250: +0.9 to +3.8 points across bins (rotation sd 0.4-2.8). At
a = $500: -0.7 to +4.9 (sd 1.4-4.9).

**Sensitivity.**
- **Seed.** With seeds 1 and 2 the ELA effect at $250 is +0.2 to +1.4 and -0.2 to +1.3. At
  $500 it is -2.1 to +2.9 and -5.4 to +4.2.
- **Basis.** Without the residual trim (`outcome="ela_notrim"`), or with l >= 1 dose
  knots, values in the hundreds or thousands appear. The outcome fit then has empty
  tensor cells: the lowest two eta basis functions times the dose basis functions
  that cover doses above about $250 hold 0-5 district-years in a fold, and the design is rank-deficient (74-75 of 77
  columns).
- **No lagged score** (`"ela_nolag"`). The lowest score bins lose support (coverage
  11-30% at $250-750) and explode.
- **Average scaled score** (`"ela_ss"`, sd 8.2). +0.4 to +0.8 at $250; -1.3 to +1.4 at $500.

**Linear check** (outside the estimator, same trimmed sample). Y on P_K(eta), X and
(A - 30)/100, clustered se. Per $100 of dose:

| Outcome | Coefficient (se) |
|---|---|
| ELA % met | +0.22 (0.10) |
| Math % met | +0.35 (0.11) |
| ELA scaled score | +0.07 (0.05) |
| Math scaled score | +0.18 (0.05) |
| ELA % met, spring t | +0.19 (0.09) |
| In-district spending per pupil, FY t | -$27 (20) |
| Actual net school spending per pupil, FY2016-22 | -$5 (34) |
| Change in net school spending per pupil | +$45 (20) |

Adding enrollment growth and the changes in EL and low-income shares to X leaves the MCAS
coefficients at +0.26 (0.10) and +0.34 (0.11).

## Caveats

- **The dose is a one-year increment that never goes away.** Hold-harmless puts year t's
  increase into every later base aid. Y in spring t+1 also reflects the t+1 increment,
  and X carries Q_{t-1} (last year's dose is a function of it).
- **What drives the score.** The score's year-to-year part (eta) is mostly enrollment
  change, low-income and EL counts, and SOA rate steps. EL influxes (FY2024-25) raise the
  foundation budget and may lower test results at the same time.
- **The SOA years.** Most doses above $500 come from FY2023-24, which are also the
  post-COVID recovery years. District recovery that differs by poverty is controlled
  only linearly (low-income share, lagged outcome).
- **Spending.** No pass-through to spending per pupil is detected. The per-pupil
  denominators differ: foundation enrollment for the dose, average membership for
  spending. Many districts spend far above their requirement, so the aid is fungible.
- **Low-income definitions.** "Economically disadvantaged" in FY2015-21, low income
  (185% of poverty, with supplemental verification) from FY2022. Year indicators absorb
  the level shift.
- **Sample.** Operating districts with grades 3-8 MCAS (regional vocational districts drop
  out), FY2019 excluded (no spring 2020 test). The lagged outcome is missing for FY2016-17:
  legacy MCAS / PARCC, flagged. Tiny districts (enrollment < 200) have the most volatile
  per-pupil scores.
- **Archived files.** FY2016-22 formula spreadsheets come from the Internet Archive. Two
  captures were truncated at 1 MiB (FY2016 2021-06-15, FY2022 2021-10-19). Complete
  captures (2021-07-10 and 2025-08-31) were used; they reproduce keyfactors.xlsx to the
  dollar.

## Getting the data

    # from code/ (the repository root); never run Python inside data/
    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_module('experiments.datasets.ma_chapter70.download', run_name='__main__')"
    # add --check (via sys.argv) for sizes and hashes only

- **Totals.** 34 files, 48.1 MB, downloaded 2026-11-12.
- **Access.** No login, form, cookie or challenge.
- **Hashes.** `download.py` records the SHA-256 of every file except the live E2C exports.
- **Macros.** The `.xlsm` workbooks are read with `xlsx.py`, a standard-library reader of
  cached cell values (copied from `ca_lcff`; this environment has no openpyxl/xlrd).
  Macros are never run.

| Files (data/raw/) | URL | Bytes | Use |
|---|---|---:|---|
| `ch70_fy2023..2027/chapter-YYYY.xlsm`, `-local.xlsx`, `-reg.xlsx`, `-whitepaper.docx` | https://www.doe.mass.edu/finance/chapter70/fy2024/chapter-2024.xlsm etc. | 16.5 M | formula, rule text |
| `ch70_fy2016..2022_wayback/chapter-16.xlsm` ... `chapter-2022.xlsm` | https://web.archive.org/web/20210710015256id_/https://www.doe.mass.edu/finance/chapter70/fy2016/chapter-16.xlsm etc. (`download.WAYBACK`) | 15.5 M | formula FY2016-22 |
| `keyfactors/keyfactors.xlsx` | https://www.doe.mass.edu/finance/chapter70/keyfactors.xlsx | 4.6 M | Chapter 70 Trends FY2007-26 (cross-check, FY2015 score) |
| `profile/profile.xlsx` | https://www.doe.mass.edu/finance/chapter70/profile.xlsx | 3.8 M | district profiles FY1993 on (not used yet) |
| `e2c_mcas/mcas_district.csv` | https://educationtocareer.data.mass.gov/resource/i9w6-niyt.csv?$where=... | 2.7 M | MCAS 2017-2026, district, all students, grades 3-8 and 10 |
| `e2c_enroll/enrollment_district.csv` | .../resource/t8td-gens.csv?$where=org_type='District' | 3.8 M | enrollment, race, EL, SWD, low income, 1992-2026 |
| `e2c_ppx/expenditures_per_pupil.csv` | .../resource/8szy-v4ed.csv | 0.5 M | spending per pupil FY2009-25 |
| `e2c_nss/nss.csv` | .../resource/5izv-jyrd.csv | 0.5 M | foundation budget and net school spending FY2008-22 |
| `census_gaz/2023_gaz_cousubs_25.txt` | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_gaz_cousubs_25.txt | 0.07 M | town county and lat/lon |

**robots.txt** (checked 2026-11-12):

| Host | Rules | How requests were made |
|---|---|---|
| www.doe.mass.edu | Crawl-delay 15; disallows /pdf, /img, /images, /mcas/search, /infoservices/research, /cms/ | none of these paths fetched; 16 s apart |
| educationtocareer.data.mass.gov | Crawl-delay 1; disallows faceted /browse, /page, /catalog, /facet queries | /resource/ API; 2 s apart |
| web.archive.org | no robots.txt (404) | 5 s apart |
| www2.census.gov | rules for named crawlers only | - |

**Cache.** `data/processed/ma_chapter70_panel.csv.gz` (5,263 LEA-years, all LEAs
FY2016-27; about 3 s to build). Delete it or call `panel(rebuild=True)` after changing
`adapter.py`.

**Running the estimator** (from `code/`, without editing shared files):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -c "import argparse; from experiments.scripts import dose_schedule_apply as app; from experiments.datasets.ma_chapter70 import dose_config as c; app.DATASETS['ma_chapter70']=c.config; app.run('ma_chapter70', argparse.Namespace(seed=0, x='linear', outcome=None))"

## Not covered / author steps

- **FY2007-15.** These need legacy MCAS outcomes on another scale, and FY2013-15 carry
  target-aid phase-in and down-payment aid (a second score). `keyfactors.xlsx` has all
  the components if wanted.
- **Archived copies.** The FY2016-22 spreadsheets are Internet Archive copies of DESE files.
  If only live official copies are acceptable, ask sfinance@doe.mass.edu (DESE's page:
  "For earlier spending requirements related to Chapter 70 please contact
  sfinance@doe.mass.edu").
