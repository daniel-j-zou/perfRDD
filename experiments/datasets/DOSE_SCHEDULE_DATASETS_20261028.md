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
| `lending_club` (in hand) | interest rate (%) on the sub-grade rank; rate table by issue month (95% of loans at their cell's modal rate), many tables 2009-2013 | 171,175 | 0.40 | about +/- one grade of the rate's natural grade | default probability +0.6 to +1.8 pp at 15% vs 12% (grades B-D), +2.9 to +4.3 pp at 18% (grades C-E), where coverage >= 89% | 0.0002 (se 0.0003), r2_absorbed 0.96: not rejected | **sticks locally** |
| `erate` (in hand) | C1 discount (%), urban/rural matrices; reference 50% | 73,434 | 0.29 | NSLP share 5-57%, discounts 40-80% | log requested dollars next year +0.37 to +0.44 at 80% vs 50% (NSLP 33-62%) | 0.0005 (se 0.0001): rejects | partly; bunching at 49.5, low R^2 limits transport |
| `medicare_dsh` (new): IPPS impact files FY 2008-13 | DSH add-on (%) on the DPP: 0 to 15%, then 2.5% + 65% of the excess to 20.2%, then 82.5%; 12% cap for small hospitals; reproduced for every hospital-year 2007-27 but one | 19,061 | 0.37 | DPP 15-30% | exit by t+2: -0.9 to -1.1 pp at 2.5%, -2.4 to -2.6 pp at 8%; log Medicare cases t+1: +3% at 2.5%, +8 to +10% at 8%. Formula cut to 25% (as in FY 2014) for hospitals at <= 16%: exit +2.2 pp (rotations 1.5-2.8), cases -5.3% (1.8-7.3) | 0.0007 (se 0.0003), r2_absorbed 0.99 | **sticks**; DPP is CMS's projection basis, mild excess mass above 15%, financial outcomes need HCRIS |
| `ca_lcff` (new): LEA-years 2020-21 to 2024-25 | concentration grant ($/ADA) = c x base x max(UPP - 0.55, 0), c = 0.50 then 0.65; reproduced to the dollar for districts | 10,645 | 0.82 | most bins for grants up to $1,500 | CAASPP % met next spring: ELA +0.7 to +2.3 pp at $1,000/ADA (math +1.1 to +1.9) in supported bins; c = 0 lowers ELA by 0.9 pp (rotations 0.3-1.5) | 0.0000 (se 0.0038): not rejected | **sticks**; small n, persistent score, COVID years, charter caps |
| `medicare_ma_benchmarks` (new): counties 2017-26 | MA benchmark % (quartile steps, blended with last year) on the FFS rank; reproduced for 50,296 of 50,298 county-years | 31,181 | 0.51 | middle ranks | change in MA penetration: mostly 0 to +0.9 pp, rotation sd as large | 0.06 (se 0.04) | weak: clean rule, near-null and noisy effects; pre-ACA cap binds for 23% |
| `title1` (new): school districts, F-33 FY 2016-19 | Targeted-Grant weighted formula children per child on the SAIPE poverty share (convex, kinks at 15.58/22.11/30.16/38.24%, zero below 5%) | 43,205 | 0.59 | low and middle poverty bins | current spending per pupil: about -$60 to -$160 at supported doses, rotation sd $190-500 (mean $12,463); the formula predicts actual Title I money only as well as a line in the poverty share (R^2 0.52) | absorbed (r2 1.00) | does not stick: small, blurred dose; only 5% at dose 0 |
| `cms_esrd_qip` (new): dialysis Quality Incentive Program, PY 2014-2024 | Medicare payment cut 0 / 0.5 / 1 / 1.5 / 2% on TPS minus the year's minimum (10-point steps); reproduced for 91,681 of 91,681 facility-years | 57,328 | 0.46 (0.25 within year) | cuts of 0.5-1% for facilities 0-30 points below the minimum | TPS two years later: -0.4 to +1.5 points for a 0.5-1% cut (sd of Y 16.5; rotation sd 0.3-1.4), i.e. no detectable effect; 1.5-2% cuts too rare (768 and 451 facility-years) | 0.04 (se 0.06): not rejected | weak: clean rule and no bunching, but a null effect and rare high doses. Another session has a duplicate `esrd_qip/` (NBER mirror) |
| `cms_hrrp` (new): readmissions penalty | continuous cut, capped at 3%, from six readmission ratios (multi-score) | 10,608 | 0.14 | not run | not run | not run | weak fit: several scores, 84% already penalised at the heart-failure kink |
| `nhanes_lunch` (in hand) | free / reduced / paid lunch on the income-poverty ratio | 12,730 | 0.31 | wide | noisy, signs change across bins | not run | does not stick (small n, eligibility not take-up) |

Partly built: `chile_sep` (Art. 16 tiers verified against the law text for 2008, 2011 and 2012; SIMCE school files and
covariates in hand). The score Q is blocked: datosabiertos.mineduc.cl and the other MINEDUC hosts did not answer
from this machine on 2026-10-28, so `load_dose()` raises FileNotFoundError and the registry skips it.

Figures and a short evaluation: `simulations/dose_schedule/dose_applications_eval.tex` (alpha, psi and U
for UI, DSH, LCFF, taxi and Lending Club), made by `experiments/scripts/dose_schedule_figures.py`.

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
| CWBH UI microdata, 5 states 1976-84 (Landais 2015, openICPSR 114581) | the main UI application: about 165,000 spells, five state schedules | sign in at https://www.openicpsr.org/openicpsr/project/114581/version/V1/view, "Download this project", accept the terms; save `Data/rawdata/*.dta` and `Readme.pdf` (about 2.3 GB) in `experiments/datasets/ui_cwbh/data/raw/` |
| Freddie Mac single-family loan-level data | fee grid on credit score x LTV, tens of millions of loans, 2023 grid change | free registration at https://freddiemac.com/research/datasets/sf-loanlevel-dataset; a few origination years into `experiments/datasets/freddie_mac/data/raw/` |
| Kentucky KEES scholarship (KYSTATS) | award rises with GPA and ACT; latent ability score | data request with an MOU at https://kystats.ky.gov/Reports/IndividualDataRequest (up to 4 months) |
| Texas ERC (Pell) | Pell award = max - EFC | proposal and fee (https://texaserc.utexas.edu) |
| HRRP FY2023 supplemental file | the cms.gov FY2023 page links the FY2024 ZIP by mistake | download the real FY2023 IPPS final-rule supplemental file by hand into `experiments/datasets/cms_hrrp/data/raw/supp/hrrp_fy2023.zip` |
| ESRD QIP minimum TPS for PY2020, 2021, 2023 (59, 56, 83, read from the data) | no CMS fact sheet for those years | optional: confirm in the ESRD PPS final rules |
| Medicare DSH financial outcomes | margins and uncompensated care | the HCRIS Hospital-2010 cost reports are public on cms.gov; say if a parser should be written |
| LCFF 2016-17 to 2018-19 (more years at c = 0.50) | a second schedule with more data | save the Principal Apportionment pages (e.g. https://www.cde.ca.gov/fg/aa/pa/pa1819.asp) from a browser; cde.ca.gov blocks scripts |
| SEDA district test scores (for Title I) | achievement outcomes | entering an e-mail at https://edopportunity.org/opportunity/data/ accepts a data use agreement; download district achievement CSVs into `experiments/datasets/title1/data/raw/seda/` |
| Chile SEP priority-student files | the score Q for `chile_sep` | when datosabiertos.mineduc.cl answers (it may need a Chilean IP), run `experiments.datasets.chile_sep.download`; else ask estadisticas@mineduc.cl, or decide whether Internet Archive copies of the official files are acceptable |
| Spain MCVL (UI with dependent-based caps) | large administrative UI data | institutional application (https://www.seg-social.es/descarga/es/120853); eligibility of non-EU institutions unclear |
