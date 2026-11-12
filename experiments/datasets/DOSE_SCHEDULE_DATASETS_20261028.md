# Datasets for the dose-schedule extension (Claude, 2026-10-28)

The author asked for datasets that fit the continuous-dose setting of the advisor's draft
"Endogenous treatment intensity" (manuscript `dose_schedule_sketch.tex`), then: "I give you
access and download as much data as you'd like. Also, keep a list of potential datasets you
actually need me to do. Go try out, and see what sticks." This is groundwork for a possible
next paper, outside `PROJECT_SCOPE.md`.

## What the data must have

1. A dose (dollars, a percentage, a rate) set by a known formula from an observed score:
   A = s(Q, group).
2. Covariates that predict the score only partly, so units with the same dose differ in the
   latent part eta = Q - gamma'(1, X).
3. A reference dose (zero, a floor or a cap) held by many units across eta.
4. An outcome after the dose, and enough units.
5. Best: a second schedule for comparable units (another state, year, vendor or rate table),
   which makes the score-direction modifier testable (one one-to-one schedule cannot).
6. **The outcome is continuous** (author, 2026-11-02): binary outcomes such as exit or default
   are out.
7. **Interpretable for a paper** (author, 2026-11-12): "A big component of success is
   interpretability. The goal is to present in a paper. I want the treatment to make sense,
   eta to be mostly interpretable." So the dose should be a lever a policymaker or clinician
   actually sets (a benefit amount, a grant, a drug dose), and X should be a natural set of
   observable traits, so that eta = Q - E(Q | X) reads as something nameable (earning
   capacity, need beyond what demographics predict, a patient's response). A higher R^2
   bought with covariates that make eta opaque (for example last year's outcome) is a bad
   trade.

## Method used

`experiments/methods/dose_schedule.py` (tests: `experiments/tests/test_dose_schedule.py`):
three-role rotation; OLS first stage; outcome fit with P_K(eta_hat), X, the tensor spline
P_k(eta_hat) x B(A) with f(e, a_ref) = 0 (or level indicators for a step schedule), and
(A - a_ref) Z; transport of every evaluation unit to every dose; averages within score bins.
`coverage` reports, for each score bin and dose, the share of the bin's units whose eta lies
inside the eta range (1-99%) of the units actually given that dose. Below about 90% the
transported value is an extrapolation. `modifier_test` adds (A - a_ref) T_hat,
T_hat = gamma_hat'(1, X), to the outcome fit; it is zero under the transport restriction
and has power only when the column is not absorbed by f (`r2_absorbed` < 1).

Runs: `python -m experiments.scripts.dose_schedule_apply <name>`, logs and JSON in
`experiments/runs/dose_schedule/`.

## Results so far

| Dataset | Dose on score | n | R^2 (Q on X) | Supported region | Main estimate | Modifier test | Verdict |
|---|---|---:|---:|---|---|---|---|
| `ui_upjohn` (new): WA 1988 + IL 1984 bonus experiments | weekly benefit / cap on log(score / cap kink); reference the cap (34% of claimants) | 27,038 | 0.51 | coverage 90-99% for doses 0.45-0.9 of the cap at all but the lowest score bin | weeks of UI paid: -2.7 to -3.2 at 60% of the cap vs the cap in the supported bins (mean 17.0; rotation sd 0.1-1.7), elasticity about 0.4 | -10.6 (se 4.3), r2_absorbed 0.98 | **sticks**; small n, two states, nominal dollars |
| `taxi` VTS | middle tip suggestion: $3 below $15, else 25% of the fare | 541,318 | 0.88 | 84-100% everywhere | tip +$0.32-0.41 at $3.75, +$1.47-1.60 at $6.25 vs $3, nearly flat in the fare | no power (r2_absorbed 1.00) | **sticks** with a linear baseline in distance; high doses unstable with a cubic baseline |
| `taxi` VTS + CMT | two schedules: VTS as above, CMT 20% of the fare at every fare (Haggag and Paci 2014: CMT 15/20/25%) | 1,019,330 | 0.88 | wide | tip about -$0.58 at $1, -$0.35 at $2, +$0.28 at $3.75, +$0.56 at $4.50, +$1.5-1.7 at $6.25 vs $3 | 0.028 (se 0.003): rejects | sticks; the rejection may reflect vendor differences (menus, fleets) |
| `ui_cwbh` (new, 2026-11-02): CWBH claims, LA, ID, MO, NM, benefit years 1976-83 (Landais 2015) | weekly benefit / cap on log(score / cap kink); 19 state-period schedules (caps $85-205); reference the cap (45% of claims); rule reproduced for 95.9-98.2% of claims after two rounding changes found in the data | 128,315 | 0.44 | coverage >= 90% for doses 0.45-0.9 of the cap in the middle score bins | weeks of UI paid: -2.3 to -3.5 at 60% of the cap, -1.5 to -2.7 at 75% (mean 14.2), elasticity 0.3-0.5; U at theta = 0.6: -3.1 (rotations 2.9-3.4). Budget-neutral cap 0.85 with rate x1.86: -0.32 weeks (rotations -0.33 to -0.31, 84% supported) | shares: absorbed (r2 1.00); in 1984 dollars: -0.075 (se 0.007), r2_absorbed 0.90, rejects | **sticks**: the main UI application. Data-use agreement to settle (readme: contact Meyer or Anderson) |
| `freddie_mac` (new, 2026-11-02): sample files 2008-2026 | up-front credit fee on the credit-score x LTV grid (fee not in the data) | 925,000 | 0.12 | about one 20-point band | not run; rates fall by about 8 bp at 11 of 12 grid cutoffs (640-740, 2009-22) against -1.6 at placebos | not run | does not stick: score barely predicted; grids would need rebuilding |
| `lending_club` (in hand) | interest rate (%) on the sub-grade rank; rate table by issue month (95% of loans at their cell's modal rate), many tables 2009-2013 | 171,175 | 0.40 | about +/- one grade of the rate's natural grade | default probability +0.6 to +1.8 pp at 15% vs 12% (grades B-D), +2.9 to +4.3 pp at 18% (grades C-E), where coverage >= 89% | 0.0002 (se 0.0003), r2_absorbed 0.96: not rejected | **sticks locally** |
| `erate` (in hand) | C1 discount (%), urban/rural matrices; reference 50% | 73,434 | 0.29 | NSLP share 5-57%, discounts 40-80% | log requested dollars next year +0.37 to +0.44 at 80% vs 50% (NSLP 33-62%) | 0.0005 (se 0.0001): rejects | partly; bunching at 49.5, low R^2 limits transport |
| `medicare_dsh` (new): IPPS impact files FY 2008-13 | DSH add-on (%) on the DPP: 0 to 15%, then 2.5% + 65% of the excess to 20.2%, then 82.5%; 12% cap for small hospitals; reproduced for every hospital-year 2007-27 but one | 19,061 | 0.37 | DPP 15-30% | exit by t+2: -0.9 to -1.1 pp at 2.5%, -2.4 to -2.6 pp at 8%; log Medicare cases t+1: +3% at 2.5%, +8 to +10% at 8%. Formula cut to 25% (as in FY 2014) for hospitals at <= 16%: exit +2.2 pp (rotations 1.5-2.8), cases -5.3% (1.8-7.3) | 0.0007 (se 0.0003), r2_absorbed 0.99 | **sticks**; DPP is CMS's projection basis, mild excess mass above 15%, financial outcomes need HCRIS |
| `ca_lcff` (new): LEA-years 2020-21 to 2024-25 | concentration grant ($/ADA) = c x base x max(UPP - 0.55, 0), c = 0.50 then 0.65; reproduced to the dollar for districts | 10,645 | 0.82 | most bins for grants up to $1,500 | CAASPP % met next spring: ELA +0.7 to +2.3 pp at $1,000/ADA (math +1.1 to +1.9) in supported bins; c = 0 lowers ELA by 0.9 pp (rotations 0.3-1.5) | 0.0000 (se 0.0038): not rejected | **sticks**; small n, persistent score, COVID years, charter caps |
| `medicare_ma_benchmarks` (new): counties 2017-26 | MA benchmark % (quartile steps, blended with last year) on the FFS rank; reproduced for 50,296 of 50,298 county-years | 31,181 | 0.51 | middle ranks | change in MA penetration: mostly 0 to +0.9 pp, rotation sd as large | 0.06 (se 0.04) | weak: clean rule, near-null and noisy effects; pre-ACA cap binds for 23% |
| `title1` (new): school districts, F-33 FY 2016-19 | Targeted-Grant weighted formula children per child on the SAIPE poverty share (convex, kinks at 15.58/22.11/30.16/38.24%, zero below 5%) | 43,205 | 0.59 | low and middle poverty bins | current spending per pupil: about -$60 to -$160 at supported doses, rotation sd $190-500 (mean $12,463); the formula predicts actual Title I money only as well as a line in the poverty share (R^2 0.52) | absorbed (r2 1.00) | does not stick: small, blurred dose; only 5% at dose 0 |
| `cms_esrd_qip` (new): dialysis Quality Incentive Program, PY 2014-2024 | Medicare payment cut 0 / 0.5 / 1 / 1.5 / 2% on TPS minus the year's minimum (10-point steps); reproduced for 91,681 of 91,681 facility-years | 57,328 | 0.46 (0.25 within year) | cuts of 0.5-1% for facilities 0-30 points below the minimum | TPS two years later: -0.4 to +1.5 points for a 0.5-1% cut (sd of Y 16.5; rotation sd 0.3-1.4), i.e. no detectable effect; 1.5-2% cuts too rare (768 and 451 facility-years) | 0.04 (se 0.06): not rejected | weak: clean rule and no bunching, but a null effect and rare high doses. Another session has a duplicate `esrd_qip/` (NBER mirror) |
| `cms_hrrp` (new): readmissions penalty | continuous cut, capped at 3%, from six readmission ratios (multi-score) | 10,608 | 0.14 | not run | not run | not run | weak fit: several scores, 84% already penalised at the heart-failure kink |
| `nhanes_lunch` (in hand) | free / reduced / paid lunch on the income-poverty ratio | 12,730 | 0.31 | wide | noisy, signs change across bins | not run | does not stick (small n, eligibility not take-up) |
| `mimic_titration` (new, 2026-11-12; open 100-patient demo only) | heparin: infusion rate change (units/kg/h) on aPTT, a 4-step nomogram (raise < 55 s, no change 55-90, lower 90-120, hold > 120); insulin: sliding-scale units on finger-stick glucose; potassium: KCl on serum K | heparin 209 events (21 patients); insulin 808 / 328; potassium 981 | heparin 0.23, insulin 0.58-0.70, potassium 0.48 | not run beyond potassium (pipeline check only) | heparin nomogram matches 81% of titrations, 61% at "no change", next aPTT after a median 7.4 h; insulin scales differ by patient and must be inferred (75-84% per-patient match); potassium is discretionary (half the draws below 3.5 get none) | - | **heparin promising** (needs full MIMIC-IV: about 137,000 titrations projected); insulin promising with caveats; potassium does not fit |
| `wc_injury` (new): workers' compensation, KY and MI (Meyer, Viscusi and Durbin 1995; R package `wooldridge`) | weekly benefit = min(2/3 wage, cap), cap raised in each state | 7,150 | 0.35 | little: the published sample drops middle earners, so low and high earners barely overlap in eta (35% KY, 21% MI) | log weeks: high earners +0.20 after the cap rise, low earners +0.01 (MVD's difference in differences) | - | weak: the design reduces to MVD's comparison |
| `medicare_wage_index` (new): FY 2016-2023 | low-wage-index boost = 0.5 x max(p25 - wage index, 0), FY 2020-24; reproduced 100% | 14,547 hospital-years | 0.71 | wide | log average hourly wage four years on: -0.015 to +0.019 at boosts of 1-3 points, same size as a placebo on FY 2018-19 | absorbed (r2 1.00) | weak: null, abstract dose |
| `medicare_ime` (new): FY 2005-2024 | IME = c x [(1 + residents per bed)^0.405 - 1]; reproduced 100% | 61,997 | 0.36 | low doses only | log cases noisy, high doses explode | -0.0006 (se 0.0002), r2 0.99 | does not stick: reference is a mass at r = 0 |
| `ma_chapter70` (new): FY 2016-2025 | aid above base per pupil = max(foundation gap, minimum aid $0-160); reproduced 83-100% by year, every miss a published overlay | 2,268 district-years | 0.90 (0.39 without last year's gap) | most bins up to $250 | MCAS ELA % met: +0.5 to +2.7 at $250 vs $30; unstable across seeds at $500+ | not rejected (r2 0.97) | weak: few large doses, eta opaque |
| `mi_proposal_a` (new): FY 2001-2024 | foundation-allowance increase above the basic increase (2x formula, catch-ups, equity payments); reproduced 99.4% | 11,349 district-years | 0.31 within year | only near the minimum (77% of district-years sit exactly at it) | change in current spending: negative, -50 to -100 $/pupil at small doses; test scores need a manual download (MI School Data disallows bots) | not rejected | weak |
| `cep_meals` (new): Texas school sites 2016-2026 | offered free-claiming share = 1{ISP >= 0.40, 0.25 from 2024-25} x min(1, 1.6 ISP) | 57,310 site-years | 0.63 | wide | lunch participation +0.010 to +0.035 at offered 0.64-0.85; adoption is chosen at group and cycle level, no jump at the threshold | -0.545 (se 0.050): rejects | weak |
| `snap_foodaps` (new): FoodAPS-1 public file, 2012-13 | SNAP benefit = maximum allotment - 0.3 x net income | 1,288 households | 0.34 | - | unstable | - | does not stick: public file cannot reproduce the benefit (15-22% in the recorded $50 bin) |

Partly built: `chile_sep` (Art. 16 tiers verified against the law text for 2008, 2011 and 2012; SIMCE school files and
covariates in hand). The score Q is blocked: datosabiertos.mineduc.cl and the other MINEDUC hosts did not answer
from this machine on 2026-10-28, so `load_dose()` raises FileNotFoundError and the registry skips it. Still down on
2026-10-29 from this machine and in the author's browser; the author put it on hold ("keep chile pending").

Figures and a short evaluation: `simulations/dose_schedule/dose_applications_eval.tex` (alpha, psi and U
for UI (Upjohn and CWBH), DSH, LCFF, taxi and Lending Club), made by `experiments/scripts/dose_schedule_figures.py`.

## Interpretability ranking (2026-11-12)

The author's test (criterion 7): does the dose make sense as a lever, and does eta have a
plain meaning? Settings that run but fail this test are not paper material.

| Setting | Dose | What eta means | Fit for a paper |
|---|---|---|---|
| UI, CWBH (and Upjohn) | weekly benefit as a share of the cap: the replacement rate and the cap are the policy | log best-quarter earnings relative to claimants of the same age, sex, race, education, family, industry, work history, state and quarter: pay level relative to similar workers | **best**: stable effects, alpha smooth in eta |
| Heparin (MIMIC) | change in the heparin rate set by the nomogram | aPTT relative to what the previous aPTT, the current rate and patient traits predict: how much more anticoagulated the patient is than expected | **strong if the full data are obtained**; eta partly includes lab noise |
| Taxi | the middle tip suggestion | fare beyond distance and time: mostly traffic delay | clean mechanics, trivial policy |
| Medicare DSH | add-on % of each payment | low-income patient share beyond size, teaching and location: safety-net intensity | readable, but the only continuous outcome is volume |
| LCFF | concentration grant $ per student | with the current X (race shares, last year's score) opaque; with a plain X (county, size, type, grades) the high-need share relative to similar LEAs in the county, but then R^2 falls to 0.25 and only high-need LEAs are supported (+1.7 to +3.5 points at $1,000, UPP 0.7-0.85) | trade-off between interpretability and support |
| Insulin (MIMIC) | sliding-scale units | glucose beyond what the last value and insulin predict | scales must be inferred |
| Lending Club | interest rate | Lending Club's information beyond the borrower data | local only |
| Others (wage index, IME, Chapter 70, Proposal A, CEP, SNAP, workers' comp) | abstract dose, chosen dose, or rule not reproduced | eta opaque or the score degenerate | no |

## Lessons so far

- **Transport distance is limited by sd(T).** A dose given at score q' reaches units at
  score q only through units with the same eta, i.e. units whose T differs by q' - q. With
  R^2 = 0.3-0.4 (E-Rate, Lending Club) only neighbouring doses are supported; with
  R^2 = 0.9 (taxi) every dose is.
- **The baseline in X carries identification.** Units at high doses have X outside the
  range of the reference units (long taxi trips), so the baseline must extrapolate: linear
  in X (Model 2) gives stable answers; a cubic in distance makes the high-dose taxi
  estimates explode.
- **Steps and several schedules make the transport restriction testable.** With one
  one-to-one schedule the modifier column is absorbed (taxi VTS: r2_absorbed 1.00).

## AUTHOR TO-DO (downloads Claude cannot do)

| Dataset | Why it matters | What to do |
|---|---|---|
| ~~CWBH UI microdata~~ | done 2026-10-29 (author); adapter `ui_cwbh` built 2026-11-02 | CWBH data use: the readme asks users to contact Bruce Meyer or Patricia Anderson for a formal agreement; decide whether to write before publishing |
| ~~Freddie Mac single-family loan-level data~~ | done 2026-10-29 (sample files 2008-2026, 5.5 GB) | screened 2026-11-02: does not stick (R^2 0.12) |
| Kentucky KEES scholarship (KYSTATS) | award rises with GPA and ACT; latent ability score | data request with an MOU at https://kystats.ky.gov/Reports/IndividualDataRequest (up to 4 months) |
| Texas ERC (Pell) | Pell award = max - EFC | proposal and fee (https://texaserc.utexas.edu) |
| HRRP FY2023 supplemental file | the cms.gov FY2023 page links the FY2024 ZIP by mistake | download the real FY2023 IPPS final-rule supplemental file by hand into `experiments/datasets/cms_hrrp/data/raw/supp/hrrp_fy2023.zip` |
| ESRD QIP minimum TPS for PY2020, 2021, 2023 (59, 56, 83, read from the data) | no CMS fact sheet for those years | optional: confirm in the ESRD PPS final rules |
| Medicare DSH financial outcomes | margins and uncompensated care | the HCRIS Hospital-2010 cost reports are public on cms.gov; say if a parser should be written |
| LCFF 2016-17 to 2018-19 (more years at c = 0.50) | a second schedule with more data | save the Principal Apportionment pages (e.g. https://www.cde.ca.gov/fg/aa/pa/pa1819.asp) from a browser; cde.ca.gov blocks scripts |
| SEDA district test scores (for Title I) | achievement outcomes | entering an e-mail at https://edopportunity.org/opportunity/data/ accepts a data use agreement; download district achievement CSVs into `experiments/datasets/title1/data/raw/seda/` |
| Chile SEP priority-student files (on hold, author 2026-10-29) | the score Q for `chile_sep` | when datosabiertos.mineduc.cl answers (it may need a Chilean IP), run `experiments.datasets.chile_sep.download`; else ask estadisticas@mineduc.cl, or decide whether Internet Archive copies of the official files are acceptable |
| Spain MCVL (UI with dependent-based caps) | large administrative UI data | institutional application (https://www.seg-social.es/descarga/es/120853); eligibility of non-EU institutions unclear |
| **MIMIC-IV full access** (for heparin, insulin) | the best medical setting: about 137,000 heparin titrations projected | PhysioNet account with institutional e-mail (https://physionet.org/register/); CITI "Data or Specimens Only Research" via the "Massachusetts Institute of Technology Affiliates" affiliation (https://physionet.org/about/citi-course/), upload the training report; apply for credentialing with your supervisor as reference (https://physionet.org/settings/credentialing/); sign the DUA (https://physionet.org/sign-dua/mimiciv/3.1/). PhysioNet warned of review delays on 2026-06-18. Steps and export SQL in `mimic_titration/README.md` |
| All of Us (statins) | LDL before and after statins, the 2026 guideline change (7.5% -> 3% risk) as a second schedule | confirm that U-M has a signed DURA (Confirm DURA tool at https://www.researchallofus.org/register/ or aoudurasupport@vumc.org); then register, verify identity, training, code of conduct; $300 initial credits |
| Michigan test scores (optional; Proposal A is weak) | outcome for `mi_proposal_a` | MI School Data disallows bots: download M-STEP grades 3-8 district files by hand (https://www.mischooldata.org/grades-3-8-state-testing-includes-psat-data-performance) into `mi_proposal_a/data/raw/mischooldata/` |
| Chapter 70 archive copies (optional) | FY 2016-22 spreadsheets came from the Internet Archive | say whether archive copies of the official DESE files are acceptable, or ask sfinance@doe.mass.edu |
