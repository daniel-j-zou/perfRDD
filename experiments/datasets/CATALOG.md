# Dataset catalog

Written 2026-11-11 (Claude). Index of every folder in `code/experiments/datasets/`, built
from each folder's README.md and adapter.py and from the five project notes in this folder
(listed at the end). Nothing was run; data presence was counted with `find <folder>/data -type f`.

Each dataset folder is one data source plus an adapter. The contract (see `../README.md`) is
that `adapter.py` exposes `load() -> RDDSample`: a score Q (one column, or several for
multi-score rules with a `treatment_rule`), the treatment D, covariates X and outcome Y.
Extra loaders (`load_two_score`, `load_dose`, other outcomes) are listed per dataset below.
`data/raw` and `data/processed` are git-ignored, so a count of 0 means the data are not on
this machine. Verdicts are quoted from the notes with the note's date; the notes are
snapshots, and a folder may have changed since.

59 folders are listed: 58 datasets plus `simulations/`, which holds simulation notes only.
In the table, "n" is the main sample size in the README (units are rows of `load()` unless said).

## Summary table

| folder | setting | score and rule class | outcome | n | data on this machine | status |
|---|---|---|---|---|---|---|
| taxi | NYC taxi tip suggestions at a $15 fare | single threshold (also a dose schedule) | continuous | about 1.5M trips per month (30,000 pilot subsample) | partial (January 2009 only) | paper application |
| nhanes | HbA1c diabetes line 6.5 and blood pressure | single threshold | continuous | about 5k | yes | early screening dataset, no verdict in the notes |
| nhanes_lunch | school lunch eligibility on income ratio 1.85 | single threshold | both | 14,307 | yes | does not stick as a dose (2026-10-28) |
| oulad | Open University first-assignment pass mark 40 | single threshold | continuous | about 21,800 | yes | early screening dataset, no verdict in the notes |
| lending_club | loan DTI trigger 30, interest rate | single threshold | continuous | about 890k | yes | sticks locally as a dose (2026-10-28) |
| lending_roi | same loans, realized lender return | single threshold | continuous | not stated | yes (reads lending_club files) | boundary at full n (adapter docstring) |
| fl_grades_2002 | Florida F grade 2002-2011 | single threshold (also two scores) | continuous | 26,713 | yes | edge: too few near the threshold (2026-10-29) |
| la_school_grades | Louisiana A-F school grades | single threshold, grade bands (two-score view) | continuous | 5,839 | yes | two-score sample too small (2026-10-28) |
| in_school_grades | Indiana A-F school grades | single threshold, grade bands (two-score view) | continuous | 1,716 | yes | too small (2026-10-28) |
| tx_school_ratings | Texas A-F campus ratings | single threshold, grade bands (two-score view) | continuous | 38,075 | yes | cleanest school result, threshold below deployed (2026-10-28) |
| nyc_progress_reports | NYC school progress report grades | single threshold, bands (two-score view) | continuous | 4,663 | yes | NYC A/B only interior estimate, not distinguishable from deployed (2026-10-28) |
| cms_hacrp | Medicare hospital-acquired condition penalty | single threshold (worst quartile; two-score view) | continuous | 14,657 | yes | no effect, edge (2026-10-28) |
| epa_pm25 | PM2.5 nonattainment designation | single threshold (also either-or on two scores) | continuous | 403 counties | yes | not usable as is: fuzzy, no local jump (2026-10-29) |
| il_school_designations | Illinois ESSA Comprehensive designation | single threshold (either-or for high schools) | continuous | 7,116 | yes | too small for four folds (2026-10-28) |
| gpa | university academic probation (Lindo et al.) | group thresholds (three campuses) | binary and continuous | 44,362 (40,582 with next GPA) | yes (in `Dep_Data/`) | boundary optimum on every outcome (2026-11-10) |
| chile_admission | Chile PAES university-application eligibility at 458 | single threshold (fuzzy) | binary | 267,341 | partial (PAES files yes, DEMRE application files no) | boundary optimum (2026-11-10); DEMRE files waiting for the author |
| vlbw | very low birth weight at 1500 g | single threshold | binary | 285,610 used (4,027,475 births) | yes | interior optimum under the logit (2026-11-11) |
| lending_default | same loans, repayment indicator | single threshold | binary | not stated | yes (reads lending_club files) | no note; adapter docstring claims an interior optimum |
| chile_teacher_scholarship | teacher scholarship, PAES 625 or 595 | group thresholds | binary | 258,843 | yes (processed cache; raw in chile_admission) | not used for binary routes: 5% outcome (2026-11-10) |
| chile_top10 | Chile top-10% class route | group thresholds in an either-or rule | binary | 21,479 (two-score 225,050) | yes (processed cache; raw in chile_admission) | not used for binary routes: 5% treated (2026-11-10) |
| chile_two_score | Chile PAES weighted sum of reading and math | two scores | binary | 223,335 | yes (processed cache; raw in chile_admission) | ready for testing; binary Y dropped by author decision (2026-10-28) |
| fdic_capital | US bank "well capitalized" categories | two scores, both-rule (ratios) | both | 248,741 | yes | edge in the estimator; failure outcome 0.5% (2026-10-28) |
| gainful_employment | US gainful-employment debt-to-earnings rates | two scores, either-or and both | continuous (also binary closure) | 7,065 | yes | flat effect, edge (2026-10-28) |
| hmda | 2024 mortgages, CLTV 80.1 and loan limit | two scores, both-rule | continuous | not stated | yes | no verdict in the notes |
| fl_school_grades | Florida A-F school grades | group thresholds (school type by year) | continuous | 23,373 | yes | no effect found at 53.5, weights point lower (2026-10-28) |
| nc_school_grades | North Carolina school grades and low-performing label | two scores | continuous | 12,425 | yes | low-performing rule would label more schools (2026-10-28) |
| ca_dashboard | California Dashboard Red color | two scores, both-rule | continuous | ELA 49,451 | yes | no effect of Red, edge (2026-10-28) |
| wi_report_cards | Wisconsin report-card stars | two scores, school-specific weights | continuous | 15,161 | yes | flat effect, edge (2026-10-28) |
| nursing_homes | CMS nursing-home star ratings | group thresholds (state by month), two-score view | continuous | 68,431 | yes | no effect, edge (2026-10-28) |
| esrd_qip | CMS dialysis quality program, NBER copy | single threshold with bands (two-score view) | continuous | 11,477 | yes | selective attrition, edge (2026-10-28) |
| hospital_340b | 340B hospital eligibility | group thresholds (two-score view) | continuous | 25,831 | yes | edge: nearly constant effect (2026-10-29) |
| us_tracts_lic | NMTC and Opportunity Zone low-income tracts | two scores, either-or | continuous | 70,320 | yes | edge: job effect near zero (2026-10-27) |
| us_tracts_ciia | CDFI Investment Areas | three scores, either-or | continuous | 76,961 | yes | edge: job effect near zero (2026-10-28) |
| tx_index_2013 | Texas Improvement Required label | two scores, either-or | continuous | 11,543 | yes | edge, rotations disagree (2026-10-29) |
| nc_ayp | North Carolina NCLB adequate yearly progress | two scores, both-rule over many cells | continuous | 1,723 (all years) | yes | edge, small (2026-10-29) |
| ui_upjohn | unemployment insurance bonus experiments, WA and IL | dose schedule | continuous | 27,590 | yes | sticks, small n (2026-10-28) |
| ui_cwbh | unemployment insurance, five states 1976-84 | dose schedule | continuous | 128,315 | yes | sticks, main UI application (2026-11-02) |
| ca_lcff | California LCFF concentration grant | dose schedule | continuous | 10,645 | yes | sticks, small n (2026-10-28) |
| medicare_dsh | Medicare DSH payment adjustment | dose schedule | continuous | 19,061 | yes | sticks (2026-10-28) |
| medicare_ma_benchmarks | Medicare Advantage county benchmarks | dose schedule | continuous | 31,181 | yes | weak: near-null effects (2026-10-28) |
| title1 | Title I Targeted Grants formula | dose schedule | continuous | 43,205 | yes | does not stick (2026-10-28) |
| cms_esrd_qip | CMS dialysis payment cuts, official reports | dose schedule | continuous | 57,328 | yes | weak: null effect, rare high doses (2026-10-28) |
| cms_hrrp | CMS readmissions penalty | dose schedule (six scores) | continuous | 10,608 | partial (FY2023 file missing) | weak fit (2026-10-28) |
| erate | E-Rate discount bands | dose schedule (also several thresholds) | continuous | 73,434 | yes | partly sticks, bunching and low R^2 (2026-10-28) |
| chile_sep | Chile SEP concentration subsidy | dose schedule | continuous | not stated (score not built) | partial (score files missing) | on hold, MINEDUC host down (2026-10-29) |
| freddie_mac | mortgage credit score by LTV grid | two scores, decision matrix | binary (default) | 925,000 sampled loans | yes (sample files) | does not stick as a dose (2026-11-02) |
| hrsa_hpsa | HRSA shortage-area scores and NHSC funding | single threshold by year | none (placeholder) | 8,174 | yes | placeholder: no credible outcome |
| cejst_tracts | CEJST disadvantaged tracts | two scores, both-rule | none (placeholder) | 73,273 | yes | placeholder: no outcome, designation retired |
| calenviroscreen | California SB 535 communities | composite score | none | about 8,000 | no | waiting for the author download |
| us_tracts_qct | HUD qualified census tracts | two scores, either-or | none yet | about 74,000 | no | waiting for the author download |
| fannie_mae | Fannie Mae loan performance | two scores, decision matrix | none yet | not stated | no | waiting for the author download |
| yelp | Yelp half-star rounding | several thresholds on one score | none yet | about 150,000 | no | waiting for the author download |
| kenya_tracking | Kenya school tracking (Duflo et al.) | group thresholds (school medians) | none yet | not stated | no | waiting for the author download |
| mimic | MIMIC-IV ICU severity score | single threshold | none (stub) | not stated | no | stub, DUA-restricted |
| nlsy | NLSY AFQT and wages | single threshold | none (stub) | about 12,700 | no | stub |
| pop_eleches | Pop-Eleches and Urquiola Romanian school thresholds | group thresholds | none (no adapter) | not stated | yes (raw replication files, 888 MB) | not usable: no individual covariates (2026-09-01) |
| adams | Adams et al. replication files | not stated | none (no adapter) | not stated | yes (do-files and PDF, 128 KB) | no note; no README and no adapter |
| simulations | simulation notes, not a dataset | not applicable | none | not applicable | not applicable | notes folder |

## (a) The paper's application

### taxi
- Treatment: percentage tip suggestions (20/25/30%) instead of fixed amounts ($2/$3/$4) at a $15 fare, Vendor VTS. Score: fare in dollars, threshold 15. D = 1{Q >= 15}.
- Y: tip in dollars. X: trip distance, passengers, tolls, surcharge, hour of day, day of week.
- Source: NYC TLC trip records (Haggag and Paci 2014 design). In hand: January 2009 raw file and the processed VTS credit-card file; `download.py` can add the other months (the README says a full year is about 5.5 GB).
- Loaders: `load`, `load_haggag_paci_vendor`, `load_haggag_paci` (the published restrictions: before November 2009, no tolls or surcharges, daytime windows, standard meter increments, fares $5 to $25).
- Notes: the README treats the 30,000-trip pilot as exploratory and the restricted sample as the paper-facing one. The dose note (2026-10-28) also uses taxi as a dose schedule: VTS alone (541,318 trips) and VTS plus CMT (1,019,330), verdict "sticks" with a linear baseline in distance, with no power in the modifier test for VTS alone.

## (b) Single-threshold candidates with a continuous outcome

### nhanes
- Treatment: HbA1c at or above 6.5 (ADA diabetes line). Score: HbA1c. Y: systolic blood pressure. X: age, sex, race, BMI, poverty index.
- Source: CDC NHANES 2017 public files; in hand (4 files), about 5k complete cases. Loader: `load`.
- Notes: no verdict in the five notes. This is one of the early (August) datasets.

### nhanes_lunch
- Treatment: eligibility for free or reduced-price school lunch, income-to-poverty ratio at or below 1.85 (Schanzenbach 2009 design). Score: the ratio. Y: lunch receipt, lunches per week, BMI. X: age, sex, race, household characteristics, cycle dummies.
- Source: CDC NCHS, cycles D to I, 14,307 children; in hand (21 files). Loaders: `load`, `load_lunch_days`, `load_subsidized`, `load_bmi`.
- Notes (2026-10-28, dose note): "does not stick" as a dose schedule (12,730 children, noisy, signs change across bins, eligibility not take-up).

### oulad
- Treatment: passing the first tutor-marked assignment, score 40 or above (UK pass mark). Score: first assignment score. Y: mean score on later assignments in the module. X: previous attempts, credits, ordinal codes for gender, education, deprivation band, age, disability.
- Source: Open University OULAD via a UCI mirror; in hand (9 files), about 21,800 student-module pairs. Loaders: `load`, `load_rich`.
- Notes (2026-10-27, multi-rule note): the two-cutoff "pass" rule was checked and not usable (exam scores exist only for two modules, and the rule matched the final result for 88%). The single threshold loader is unchanged.

### lending_club
- Treatment: debt-to-income ratio at or above 30 (underwriting trigger). Score: DTI. Y: originated interest rate. X: loan amount, income, delinquencies, accounts, inquiries, ordinal codes.
- Source: Lending Club public archive 2007-2015, about 890k loans; in hand (9 files). Loader: `load`.
- Notes: the README does not say that the interest rate is set by the lender from the score, but the `lending_default` docstring does. In the dose note (2026-10-28) the same data, with the rate on the sub-grade rank as a dose, "sticks locally" (171,175 loans, default probability up 0.6 to 4.3 pp at higher rates).

### lending_roi
- Same score, treatment and X as `lending_club`; Y is realized lender return, (total payment - funded amount) / funded amount, on completed loans only. n not stated in a README (the folder has no README).
- Data: reads `lending_club/data/raw/loans.csv`. Loader: `load`.
- Notes: the adapter docstring says the screened result is a boundary optimum at full n, citing the research log. No dated note in this folder.

### fl_grades_2002
- Treatment: an F grade from the Florida A+ plan, 2002-2011 (vouchers to 2005, state intervention). Score: grade points minus the year's D minimum; the treated side is below 0. Y: next year's percent of possible points. X: CCD school characteristics (96 columns).
- Source: FLDOE spreadsheets 2001-2014 and NCES CCD; in hand (74 files), 26,713 school-years, 437 graded F. Loaders: `load`, `load_next_proficiency`, `load_next_status`, `load_next_growth`, `load_next_grade`, `load_closed`, `load_two_score`.
- Notes (2026-10-29): published F matches the rule for every school, local jump +3.1 (1.1), but the estimator goes to the edge because only 7% of the evaluation fold is inside the trim.

### la_school_grades
- Treatment: letter grade B or better rather than C (other boundaries via `load_ab`, `load_cd`, `load_df`). Score: published school performance score minus its threshold. Y: next year's score. X: October 1 enrollment shares and charter flag.
- Source: Louisiana DOE files; in hand (20 files), 5,839 school-years, two-score view 1,089 (0.75 assessment plus 0.25 progress). Loaders also include `load_next_assessment`, `load_next_grade`, `load_closed`, `load_two_score`.
- Notes (2026-10-28): grade matches the rule 99.9%. The two-score sample is too small for the four-role split.

### in_school_grades
- Treatment: a D or better rather than an F (other boundaries via `load_ab`, `load_bc`, `load_cd`); weighted sum of domain points (grades 3-8: 0.5 and 0.5). Y: next year's overall points (`load`) or pass rate (pooled loaders). X: lunch, race, English learner and special education shares, enrollment.
- Source: Indiana IDOE, 2016-17 and 2017-18 grades; in hand (25 files). n 1,716 schools, pooled 2,590 school-years. Loaders: `load`, `load_ab`, `load_bc`, `load_cd`, `load_df`, `load_two_score`, `load_pooled`, `load_two_score_pooled`.
- Notes (2026-10-28): too small for four folds; sd of the estimated threshold 6.7 to 7.6 points.

### tx_school_ratings
- Treatment: overall A rather than B (other boundaries via `load_bc`, `load_cd`, `load_df`). Score: 0.7 max(achievement, progress) plus 0.3 closing the gaps, minus the threshold. Y: next rated year's overall score. X: disadvantaged and bilingual shares, enrollment, district variables.
- Source: TEA 2026 multi-year ratings spreadsheet; in hand (12 files), 38,075 campus-years. Loaders also: `load_next_achievement`, `load_next_ctg`, `load_closed`, `load_two_score`, `load_three_score`.
- Notes (2026-10-28): rule matches 100% outside four local-accountability districts. "Texas is the cleanest school result": with weights fixed the A/B threshold is estimated about 1.9 sd below the deployed 89.5; value gains are small.

### nyc_progress_reports
- Treatment: grade C or better rather than D or F (`load`); A/B and B/C via `load_ab`, `load_bc`, D/F via `load_df`. Score: overall score (Environment 15 plus Performance 25 plus Progress 60) minus the threshold for school type and year. Y: next year's overall score. X: school demographics, peer index, prior percentile, district dummies.
- Source: NYC Open Data and Internet Archive copies of DOE guides; in hand (29 files), 4,663 school-years at C/D. Loaders also: `load_cd`, `load_next_progress`, `load_no_grade_next`, `load_two_score`.
- Notes (2026-10-28): rule reproduced at A/B and B/C; 13% of D/F school-years have no grade next year, so Y is missing selectively below the threshold. A/B is "the only interior estimate this round" and is not distinguishable from the deployed threshold.

### cms_hacrp
- Treatment: a 1% Medicare payment cut for hospitals in the worst quartile of the Total HAC Score; the treated side is higher. Score: (Total HAC Score minus the year's threshold) over that year's sd. Y: the standardized score three program years later (`load`). X: hospital size, teaching, DSH, case mix, ownership.
- Source: CMS Hospital Compare archives via NBER's mirror (data.nber.org), 14,657 hospital-years; in hand (53 files). Loaders: `load`, `load_psi90`, `load_all_years`, `load_next_year`, `load_next_year_psi90`, `load_two_score`, `load_two_score_psi90`.
- Notes (2026-10-28): matches CMS penalty flags for 100% of FY2016-2026; estimator shows no effect and goes to the edge. Download source is the NBER mirror whose robots.txt disallows everything; the notes (2026-10-29) record that the author was asked what to do.

### epa_pm25
- Treatment: PM2.5 nonattainment designation of a county (1997 round and later rounds). Score: county maximum design value minus the standard plus half a rounding unit; two-score loader adds the 24-hour value for an either-rule. Y: change in annual mean PM2.5 3 to 5 years later. X: population, density, manufacturing share, region, earlier PM levels.
- Source: EPA design-value reports, Green Book, AQS; in hand (48 files). n 403 counties (73 treated); pooled 1,226 county-rounds. Loaders: `load`, `load_1997`, `load_2006`, `load_2012`, `load_1997_dvsite`, `load_two_score`, `load_pooled`.
- Notes (2026-10-29): rule reproduces for 89 to 98% of counties and the design is fuzzy below the threshold; local jump -0.12 (se 0.30); "EPA fails (3)" (a sharp rule); estimator not run.

### il_school_designations
- Treatment: not designated Comprehensive support (`load`); high schools are also Comprehensive by graduation rate (an either-rule). Score: the summative index minus the bottom-5% threshold. Y: next year's index. X: enrollment, demographic shares, school and district type.
- Source: ISBE Report Card Service (3,698 school files, paced requests); in hand (3,717 files). n 7,116 school-band-years, 146 within 2 points of the threshold. Loaders: `load`, `load_es`, `load_hs`, `load_not_in_support`, `load_exemplary`, `load_two_score`, `load_two_score_indicators`, `load_two_score_indicators_no_elptp`.
- Notes (2026-10-28): the rule reproduces every official designation 2023-2025; too small for four folds, estimator not run; support lasts four years so many units at the threshold are already treated.

## (c) Binary-outcome settings

Author decision of 2026-10-28 (multi-rule note): the two-score work keeps to continuous Y. Binary outcomes are outside the paper (binary note, 2026-11-10). The exploratory run on gpa and chile_admission, and an addendum on vlbw, are in `BINARY_OUTCOME_ROUTES_20261110.md`.

### chile_admission
- Treatment: eligibility to apply to universities, PAES best reading/math-1 average at or above 458 (top-10% students are exempt, so the design is fuzzy). Score: that average. Y: university enrollment 2024, any enrollment 2025, university enrollment 2025. X: NEM, class ranking, school type, track, sex, years since graduation, family income group, Santiago.
- Source: MINEDUC open data (PAES and higher-education enrollment, linked by MRUN), 267,341 registrants; the PAES files are in hand (27 files, 2.6 GB). The DEMRE application files (program weights and selection) are not: the portal runs a reCAPTCHA and the README lists browser download steps for the author.
- Loaders: `load` plus `load_univ_2024`, `load_any_2025`, `load_univ_2025`, flexible-covariate variants (`*_flex`), `load_sel_2024`, `load_sel_2025`, `load_acred_2024`, `load_acred_2025`, `load_duration_2024`.
- Notes (2026-11-10): all four routes put the optimum at the edge (threshold 420, treat most); tau stays positive across the policy set; the logit does not close the gap to local estimates (univ_2024 logit +0.147 against local +0.077). Access routes for the rest of the Chile data are in `CHILE_DATA_ACCESS.md` (2026-09-29).

### gpa
- Treatment: academic probation (Lindo, Sanders and Oreopoulos 2010), first-year GPA below the campus threshold (1.5, 1.5, 1.6 by campus). Score: distance from the threshold. Y: next GPA (selected sample), graduation within 5 or 6 years, return for year two, staying. X: high-school grade percentile, credits, campus, sex, birthplace, age, English.
- Source: AEJ Applied replication package; in hand under `Dep_Data/` (the `data/` folder does not exist, so the count rule gives 0 here). n 44,362 (40,582 with next GPA). The processed CSV is git-LFS tracked. Loader: `load` (with `extras["group"]` for campus).
- Notes: probation matches the rule except for 51 students (2026-10-27). Binary note (2026-11-10): every route puts the optimum at c = -0.6 (fewest on probation), tau falls with the threshold, so the graduation crossing near c = -0.3 is a minimum. The README warns that the next-GPA sample is selected after treatment. The README also lists two results files, `HARD_TRIM_RESULTS.md` and `WELFARE_RESULTS.md`.

### vlbw
- Treatment: very-low-birth-weight care below 1500 g (treat below; scripts mirror the score). Score: reported birth weight in grams. Y: infant death within one year (binary, survival used in the note). X: gestation, plurality, sex, mother's age, race, education, prenatal care, tobacco, others.
- Source: NCHS 2002 birth-cohort linked birth/infant-death file, 4,027,475 births, sample 500-2500 g with known gestation 285,610; in hand (9 files, 1.1 GB). Loader: `load`. The README records that ftp.cdc.gov has robots.txt `Disallow: /` and the author instructed the download on 2026-11-11.
- Notes (2026-11-11 addendum): the case where the link matters. Under the logit the effect grows as weight falls, so the optimum is interior once a per-infant cost is set (1512 g at cost 0.02, bootstrap interval 1476 to 1543); the linear probability model fails (effect extrapolates to +23 pp at 1800 g, optimum at the edge for every cost). The 1500 g line is a guideline and no cost is derived from the data.
- Disagreement: README says infant death is about 0.7% overall; the note says survival is 96.0% in the 500-2500 g sample. These are different samples, not a conflict.

### lending_default
- Same score, treatment and X as `lending_club` (DTI at or above 30); Y = 1{fully paid} (binary repayment). No README; n not stated.
- Data: reads `lending_club/data/raw/loans.csv`. Loader: `load`.
- Notes: the adapter docstring claims the effect changes sign across the overlap window so the criterion has an interior optimum, and `lending_roi` says the continuous version is a boundary at full n. No dated note in the five notes mentions it.

### Other binary or rare-event settings (summaries in section d)
`chile_two_score` and `chile_top10` have binary enrollment outcomes; `chile_teacher_scholarship` has a 5% outcome; `fdic_capital` has bank failure (0.5% of bank-years); `gainful_employment` has program closure (n 1.8k to 7k); `freddie_mac` would use default. The binary note (2026-11-10) did not use chile_top10 (21k, 5% treated), chile_teacher_scholarship, fdic_capital failure or gainful_employment.

## (d) Group-threshold, either-or and two-score rules (the multi-rule side track)

Source note: `MULTI_RULE_DATASETS_20261027.md` (rounds dated 2026-10-27 to 2026-10-29). This is groundwork outside `PROJECT_SCOPE.md`; multiple scores are listed there as future work. Estimator results are in `simulations/TWO_SCORE_ESTIMATOR_20261027.md`. The note's lesson (2026-10-29): an interior optimum needs an effect that varies and crosses the cost inside the supported region, enough units near the threshold, and a sharp rule.

### chile_two_score
- Treatment: eligibility to apply to universities, (reading + math-1) / 2 at or above 458. Scores: best reading and best math-1 PAES scores. Y: any enrollment 2025 (`load`), university enrollment 2024 or 2025 (`load_univ_2024`, `load_univ_2025`). X: chile_admission covariates without the top-10% flag.
- Source: chile_admission raw files (MINEDUC open data); processed cache in hand (3 files). n 223,335. Other loaders: `load_2023_univ_2024`, `load_2025_univ_2025` (2023 and 2025 cohorts).
- Notes (2026-10-27/28): rule matches the published flag for 100%; the weakest direction of T (reading against math) is weak (corr(T1, T2) = 0.94), so rules that trade the tests against each other are weakly identified. With weights fixed the estimated threshold is 416 to 436 at cost 0.1 and 476 to 504 at cost 0.2 (costs are illustrative). Binary Y dropped by the author's 2026-10-28 decision.

### chile_top10
- Treatment: eligible through the top-10% route: high-school grade average minus the class's threshold at or above 0 (`load`); `load_two_score` is the either-rule PAES >= 458 or class route. Y: university enrollment 2024. X: chile_two_score covariates without NEM and ranking (p = 11).
- Source: chile_admission raw files; processed cache in hand (6 files). n 21,479 below 458 (1,104 treated); two-score 225,050. Loaders: `load`, `load_mid`, `load_2025`, `load_2023_univ_2024`, and the matching `load_two_score*` versions. Class thresholds are bracketed from registrants, not published; `load_mid` uses the midpoint.
- Notes (2026-10-28): eligibility matches the published flag for every registrant. The PAES threshold is estimated at 397 (cost 0) and 448 (cost 0.1); the grade route is pushed to the edge (R^2 0.03).

### chile_teacher_scholarship
- Treatment: academic eligibility for the teacher-education scholarship, threshold 625, or 595 for recent top-10% graduates of non-private schools (group-specific). Score: PAES average centred at the student's own threshold. Y: enrollment in a university education program 2024 (`load`) or 2025 (`load_ped_2025`). X: chile_admission covariates plus group.
- Source: chile_admission raw files plus the 2024 scholarship file; processed cache in hand (1 file). n 258,843 (group 1: 21,574).
- Notes (2026-10-27): 96% of the 1,368 award holders meet the rule; the first stage is small (1.2% of eligible students). Not used in the binary-routes run because the outcome is 5% (2026-11-10).

### fdic_capital
- Treatment: "well capitalized" by ratios, a both-rule (total risk-based at or above 10, leverage at or above 5, plus tier 1 and CET1 from 2015). Score: the binding margin (`load`) or (total risk-based, leverage) (`load_two_score`). Y: log loan growth over four quarters; `*_fail`: bank failure within eight quarters. X: bank size and balance-sheet ratios, charter, state, year.
- Source: FDIC BankFind API; in hand (599 files). n 248,741 bank-years; failure sample 259,430 with 1,274 failures. Loaders: `load`, `load_fail`, `load_quarterly`, `load_two_score`, `load_two_score_fail`, `load_adequate`, `load_adequate_fail`.
- Notes (2026-10-28): ratios recomputed exactly (no published bank-level category); estimator goes to the edge and banks bunch just above both thresholds.

### gainful_employment
- Treatment: not passing the 2015 debt-to-earnings test (annual rate above 8 and discretionary rate above 20, a both-rule; passing is the either-rule); fail is above 12 and 30. Scores: the two rates. Y: no IPEDS completions in 2019-20, or change in log completions (`*_dlog`). X: program and institution characteristics.
- Source: ED final rates file and IPEDS; in hand (23 files). n 7,065 programs for the two-score loaders (`load` 1,798; `load_annual` 3,329). Loaders: `load`, `load_dlog`, `load_annual`, `load_annual_dlog`, `load_two_score`, `load_two_score_dlog`, `load_two_score_fail`, `load_two_score_fail_dlog`.
- Notes (2026-10-28): rule matches 8,528 of 8,528 programs without an approved appeal; no program lost aid (rule rescinded 2019); effect on closure about 1 point (sd 0.8), flat, 100% edge.

### hmda
- Treatment: both CLTV above 80.1 and loan amount above 766,550 (conforming limit). Scores: (combined loan-to-value, loan amount). Y: interest rate plus 0.8 times the CLTV-over-80 indicator (PMI-adjusted financing cost). X: about 26 borrower, contract and tract columns.
- Source: HMDA 2024 LAR (CFPB), public; in hand (51 files, 2.2 GB). The adapter reads `data/raw/dftest_export.csv`. n not stated. Loader: `load`.
- Disagreement: the multi-rule note (2026-10-27) lists `hmda/` as "in hand, the author decides whether to build it", but the folder already has an adapter. No verdict in the notes.

### fl_school_grades
- Treatment: grade B or better rather than C, with thresholds by school type and spring year (group-specific); the score is a fixed-weight sum of 11 or 12 components. Y: next grading year's unrounded percent of points. X: disadvantaged and minority shares, enrollment, Title I, school type, district.
- Source: FLDOE CDN files; in hand (34 files). n 23,373 school-years, two-score 6,940. Loaders: `load`, `load_ab`, `load_bc`, `load_cd`, `load_df`, `load_next_ela`, `load_next_math`, `load_next_grade`, `load_closed`, `load_two_score`.
- Notes (2026-10-27/28): published grade matches the rule for 34,343 of 34,343. Moving only the threshold gives no reason to leave 53.5; the chosen achievement weight 0.31 is below the deployed 0.43; gains are +0.04 to +0.05 points per school.

### nc_school_grades
- Treatment: grade B or better rather than C on 0.8 achievement plus 0.2 growth (`load`); `load_low_performing` is the both-rule (index below 54.5 and growth below 85). Scores: (achievement, growth). Y: next grading year's score. X: disadvantaged, race, English learner shares, enrollment, teacher experience, charter, Title I.
- Source: NC DPI School Report Card; in hand (5 files). n 12,425 school-years. Loaders: `load`, `load_ab`, `load_bc`, `load_cd`, `load_df`, `load_next_ach`, `load_next_growth`, `load_closed`, `load_low_performing`, `load_low_performing_next_lp`.
- Notes (2026-10-28): grade 100%, low-performing 100% of 4,055 designations. B/C gives no reason to move; the low-performing rule would label more schools (score 59.6 against 54.5, about 2 sd; gain +0.08).

### ca_dashboard
- Treatment: Red color on the California Dashboard, a both-rule (status Very Low and change below +3.0); Red-or-Orange is a staircase. Scores: (status, change) in distance-from-standard points (`load_two_score`, `load_math_two_score`). Y: next Dashboard year's status. X: enrollment shares from the Census Day file.
- Source: CDE research files (www3.cde.ca.gov); in hand (30 files). n ELA 49,451 school-years, math 49,442. Loaders: `load`, `load_math`, `load_two_score`, `load_math_two_score`, `load_status_2022`.
- Notes (2026-10-28): every color reproduced 2017-2026 after documented exclusions. No effect of Red on next status (-0.05 and 0.05, sd 0.04), 100% edge.

### wi_report_cards
- Treatment: rating of 4 stars or better rather than 3 (`load`; other boundaries via `load_stars5`, `load_stars3`, `load_stars2`). The weighted sum's achievement weight falls with the disadvantaged share, so the rule has school-specific weights and thresholds. Scores: (achievement, growth). Y: next report card year's overall score. X: demographic shares, enrollment.
- Source: Wisconsin DPI report-card downloads; in hand (45 files). n 15,161 school-years, two-score 13,710. Loaders: `load`, `load_stars5`, `load_stars4`, `load_stars3`, `load_stars2`, `load_two_score`.
- Notes (2026-10-28): every published score from 2017-18 reproduced. The rating has no formal consequence in these years, so D is the published rating itself. The estimator is flat (the four rotations spread over 45 to 90 at cost 2).

### nursing_homes
- Treatment: health-inspection rating of 2 or more stars rather than 1, with state-by-month percentile thresholds (group-specific); `load_two_score` is the 5-star staffing both-rule (RN hours and total hours). Y: log(1 + deficiency score) of the next standard survey. X: beds, ownership, chain, occupancy, state, year.
- Source: CMS Nursing Home Compare via NBER's mirror, state tables from cms.gov and the Internet Archive; in hand (207 files). n 68,431 inspection events (staffing 66,322). Loaders: `load`, `load_points`, `load_lagged`, `load_residents`, `load_23`, `load_34`, `load_45`, `load_two_score`, `load_two_score_lagged`, `load_two_score_residents`.
- Notes (2026-10-28): every check 99.99 to 100%; no effect, edge. Disagreement: the multi-rule note's "Waiting for the author" table (2026-10-27) still lists nursing_homes with CMS access denied; the same note's fifth round says it was a stub and was built from the NBER mirror. The README matches the fifth round.

### esrd_qip
- Treatment: no Medicare payment reduction for dialysis facilities, TPS at or above the year's minimum (`load`); `load_bands` counts the four reduction cut points; `load_two_score` uses (clinical, reporting) domain scores with weights 0.75 and 0.25. Y: TPS two payment years later. X: ownership, chain, stations, lagged measures.
- Source: CMS QIP files from NBER's mirror (data.nber.org); in hand (14 files). n 11,477 facility-years (PY2015 and PY2017). Loaders: `load`, `load_clinical`, `load_next_py`, `load_bands`, `load_two_score`.
- Notes (2026-10-28): 100% of published reductions PY2014-2019; selective attrition in dialysis. The dose note records a duplicate session folder, `cms_esrd_qip/` (section e). Same NBER mirror caveat as `cms_hacrp`.

### hospital_340b
- Treatment: 340B eligibility on the DSH test, DPP (SSI plus Medicaid fractions) at or above 27.325, or 22.765 for rural referral and sole community hospitals (group-specific). Score: DPP minus the group threshold; `load_two_score` gives (SSI %, Medicaid %). Y: log outpatient drug charges two cost reports later. X: hospital characteristics (33 columns).
- Source: CMS HCRIS cost reports plus `medicare_dsh` impact files; in hand (32 files). n 25,831 hospital-reports; DSH group two-score 18,640; `load_rrc_sch` 7,191. Loaders: `load`, `load_two_score`, `load_rrc_sch`.
- Notes (2026-10-29): rule matches the cost report's own line 33 for 99.7 to 99.9%; local jump +0.24 (se 0.07), 340B billing 39% to 73%. Estimator: edge, because the effect is nearly constant so every rule treats all or none.

### us_tracts_lic
- Treatment: low-income community (NMTC and Opportunity Zone eligibility): poverty at or above 20% or income ratio at or below 80, an either-rule on two scores. Y: change in log jobs 2017-2019 (LODES7); placebo 2015-2017; LIHTC placement. X: 2010 decennial counts only.
- Source: CDFI Fund, Census ACS and Planning Database, LEHD, HUD; public, in hand (366 files, 491 MB). n 70,320 tracts. Loaders: `load`, `load_placebo`, `load_lihtc`.
- Notes (2026-10-27): rule matches CDFI's flag for 100%; the two scores are correlated -0.85. The job effect is about zero, so every rule class goes to the edge.

### us_tracts_ciia
- Treatment: CDFI Investment Area, an either-rule on three scores (poverty at or above 20, income ratio at or below 80, unemployment ratio at or above 1.5), ACS 2016-2020. Y: change in log jobs 2022-2023 (LODES8). X: 2020 decennial counts.
- Source: CDFI Fund, Census, LEHD; in hand (155 files). n 76,961 tracts (two-score 50,028). Loaders: `load`, `load_placebo`, `load_all`, `load_two_score`.
- Notes (2026-10-28): 100% outside the exception routes. Edge, effect on jobs about zero.

### tx_index_2013
- Treatment: Texas Improvement Required label, an either-rule on index margins (Index 1 or 2 below target, or Index 3 or 4); the treated side is below 0. Y: next rated year's Index 1 (`load`), its change (`load_gain`), or the 2018 A-F achievement score (`load_2017`). X: disadvantaged and bilingual shares, mobility, enrollment, district.
- Source: 34,532 static TEA summary PDFs converted with pdftotext; in hand (34,557 files). n 11,543 campus-years (1,061 IR). Loaders: `load`, `load_gain`, `load_two_score`, `load_three_score`, `load_2017`. 2016 files are missing (TEA form).
- Notes (2026-10-29): rule matches 99.94%; local jump +1.1 (0.6); estimator at the edge and rotations disagree.

### nc_ayp
- Treatment: missing adequate yearly progress, a rule over many school-subgroup cells (both-rule for passing, either-rule for missing it); the treated side is below 0. Scores: minimum margin over all cells, or (reading, math). Y: next year's math (`load`) or reading proficiency. X: lunch share, enrollment, teacher variables.
- Source: Internet Archive copies of NC DPI report-card data; in hand (12 files). n 607 school-years in the main years (2004, 2005, 2010); 1,723 in all years 2004-2010. Loaders: `load`, `load_two_score`, `load_reading`, `load_restructuring`, `load_entering`, `load_first_threat`, `load_all`, `load_all_two_score`.
- Notes (2026-10-29): rule match 88% (97% in 2004, 2005, 2010); effect +2 to +4 in improvement; edge, with a density bump just above 0. The note quotes 1,723 as the improvement sample, while the README gives 1,723 for all years of `load(years=ALL_YEARS)`; I did not resolve which is right.

## (e) Dose schedules (the intensity side track)

Source note: `DOSE_SCHEDULE_DATASETS_20261028.md` (2026-10-28, with entries to 2026-11-02). This is groundwork for a possible next paper, outside `PROJECT_SCOPE.md`. Method: `experiments/methods/dose_schedule.py`; runs via `python -m experiments.scripts.dose_schedule_apply <name>`. Every dose folder exposes `load_dose()`, and `load()` gives D = 1{score above the threshold}.

### ui_cwbh
- Dose: weekly unemployment benefit as a share of the cap, set by a kinked schedule on highest-quarter earnings (19 state-period schedules in LA, ID, MO, NM, caps $85 to $205); reference dose is the cap (45% of claims). Y: weeks of UI paid. X: age, sex, race, education, dependents, industry, weeks worked.
- Source: Landais (2015) data, openICPSR 114581 (author's login); in hand (39 files). n 130,935 claims (128,315 after trimming); WA adds 42,247 with an imputed score. Loader: `load_dose` only (no `load`).
- Notes (2026-11-02): "sticks": the main UI application. Weeks paid down 2.3 to 3.5 at 60% of the cap; the modifier test rejects in 1984 dollars. The data-use agreement (contact Meyer or Anderson) is still to be settled.

### ui_upjohn
- Dose: weekly benefit relative to the cap on average of two highest quarters of earnings; WA 1988 and IL 1984 bonus experiments. Y: weeks with UI paid (also dollars, spell, later earnings). X: age, sex, race, lagged earnings, state, claim month.
- Source: W.E. Upjohn Institute public files; in hand (86 files). Loaders: `load_dose`, `load`. n 27,590 in the README.
- Notes (2026-10-28): "sticks"; small n, two states, nominal dollars; weeks paid down 2.7 to 3.2 at 60% of the cap. The note's table gives n 27,038 against the README's 27,590 (the note's count is after its own trimming, as far as I can tell; not confirmed).

### ca_lcff
- Dose: California concentration grant, 0.50 (then 0.65) times the base grant times max(UPP - 0.55, 0), dollars per funded ADA. Score: unduplicated pupil share. Y: CAASPP percent meeting standard next spring (ELA; math optional). X: charter flag, enrollment, race shares, lagged outcome, location.
- Source: CDE LCFF Summary Data, CALPADS counts and CAASPP files; in hand (46 files). n 10,645 LEA-years. Loaders: `load_dose`, `load`.
- Notes (2026-10-28): rule reproduced to the dollar for districts; "sticks" with caveats (small n, persistent score, COVID years, 41% of charter-years have a binding cap); ELA +0.7 to +2.3 pp at $1,000 per ADA.

### medicare_dsh
- Dose: operating DSH adjustment on the disproportionate patient percentage (0 to 15% then 2.5% plus 65% of the excess, a kink at 20.2, a cap for some hospitals; scaled by 0.25 from FY2014). Y: log Medicare cases next year; `outcome="exit"` for exit. X: hospital characteristics (69 columns).
- Source: CMS IPPS Impact Files (NBER CSV copies for FY2005-2024, CMS files after); in hand (55 files). n 19,061 hospital-years (FY2008-2013). Loaders: `load_dose`, `load`.
- Notes (2026-10-28): "sticks": exit down 0.9 to 2.6 pp, cases up 3 to 10%. Financial outcomes would need HCRIS cost reports (author to say if a parser is wanted).

### medicare_ma_benchmarks
- Dose: Medicare Advantage applicable percentage (0.95 to 1.15) by quartile of the county's FFS rank, blended with last year (40 groups). Score: percentile rank of the FFS rate. Y: MA penetration in July. X: lagged penetration, demographics, income, risk score.
- Source: CMS ratebooks and penetration files, Census, SAIPE; in hand (441 files). n 31,181 county-years (2017-2026). Loaders: `load_dose`, `load`.
- Notes (2026-10-28): rule reproduced for 50,296 of 50,298 county-years; "weak": near-null noisy effects, the pre-ACA cap binds for 23%.

### title1
- Dose: Title I-A Targeted Grant formula weight on the district's SAIPE child-poverty share (jump at 5%, convex kinks at 15.58, 22.11, 30.16, 38.24%). Y: current spending per pupil (also first-stage outcomes). X: enrollment, spending, race shares, locale, state.
- Source: Census SAIPE, F-33 finance survey, NCES CCD; in hand (32 files). n 43,205 district-years (FY2016-2019). Loaders: `load_dose`, `load`.
- Notes (2026-10-28): "does not stick": effect -$60 to -$160 with rotation sd $190 to $500, formula predicts actual Title I money with R^2 0.52. SEDA test scores would need an author download.

### cms_esrd_qip
- Dose: Medicare payment cut 0, 0.5, 1, 1.5, 2% by 10-point steps of TPS below the year's minimum. Y: TPS two payment years later (also SHR, SMR). X: lagged QIP results, chain, facility age, state (95 columns).
- Source: CMS QIP Performance Score Summary Reports and Dialysis Facility Compare snapshots (cms.gov); in hand (280 files). n 57,328 facility-years. Loaders: `load_dose`, `load`.
- Notes (2026-10-28): rule reproduced for 91,681 of 91,681; "weak": no detectable effect, 1.5 to 2% cuts rare. Duplicate of `esrd_qip/` (section d) on a different source.

### cms_hrrp
- Dose: Medicare readmissions penalty, a continuous cut capped at 3% from six condition ratios (a kink on a multi-score). Score: heart-failure excess ratio minus the peer-group median (any condition selectable). Y: the same ratio in the FY t+3 file. X: dual proportion, discharges, peer group, state (70 columns).
- Source: CMS IPPS final-rule supplemental files; in hand (117 files); the FY2023 file is missing (the cms.gov page links the FY2024 zip by mistake, a hand download is on the author list). n 10,608 hospital-years. Loaders: `load_dose`, `load`.
- Notes (2026-10-28): "weak fit": R^2 0.14, and 84% already penalized at the heart-failure kink.

### erate
- Dose: E-Rate Category 1 discount (20 to 90%) by bands of the school-lunch share, separate urban and rural matrices; reference 50%. Y: log requested dollars next year (`load`), committed dollars (`load_committed`). X: enrollment, school counts, locale flags, state, year.
- Source: USAC Open Data; in hand (24 files). n 73,434 entity-years (FY2022-2025). Loaders: `load`, `load_c20`, `load_c35`, `load_c75`, `load_committed`.
- Notes (2026-10-28): "partly sticks": log dollars +0.37 to +0.44 at 80% against 50%; R^2 0.29 limits transport; bunching just above 49.5 (ratio 1.77). The multi-rule note lists it as a rule check, 309,351 applications matching the matrix.

### chile_sep
- Dose: Chile SEP concentration subsidy, a step schedule by the school's share of priority students (tiers 15, 30, 45, 60). Y: school mean SIMCE 4th grade (reading plus math over 2); also math, reading, promotion, attendance. X: dependency, rural, enrollment, lagged SIMCE, region and year.
- Source: Ley 20.248, MINEDUC open data, Agencia de Calidad SIMCE files; partial (459 files). The score (priority students by school) is blocked: datosabiertos.mineduc.cl did not answer from this machine, so `load_dose()` raises FileNotFoundError. n is marked TODO in the README. Loaders: `load_dose`, `load_dose_math`, `load_dose_reading`, `load_dose_promotion`, `load_dose_attendance`, `load`.
- Notes (2026-10-29): put on hold by the author ("keep chile pending"); tiers verified against the law for 2008, 2011 and 2012.

### freddie_mac
- Dose: up-front credit fee on the credit-score by LTV grid (fee not in the data); mortgage insurance above 80% LTV. Scores: (credit score, LTV), DTI third. Y: 90-day delinquency or default. X: loan purpose, property, occupancy, borrowers, state, year.
- Source: Freddie Mac single-family sample files 2008-2026 (author download 2026-10-29; 38 files, 50,000 loans per origination year). The folder has no `adapter.py`, only `screen.py`, so there is no `load()` yet.
- Notes (2026-11-02): screened, "does not stick": R^2 of the score on X is 0.12, about one 20-point band supported. Disagreement: the multi-rule note's waiting table (2026-10-27) lists freddie_mac as waiting for a free account; the dose note and README say the sample files are in hand.

## (f) Placeholders, not usable, or waiting for the author

### hrsa_hpsa
- Treatment: HPSA score at or above the NHSC loan repayment funding threshold of the year (13 in FY2025, 17 in FY2024, 16 in FY2023 and FY2020). Score: integer 0 to 25. Y is a placeholder (NHSC FTE per 10,000 residents in 2024, which predates the 2025 scores). X: 84 columns.
- Source: data.hrsa.gov and AHRF; in hand (36 files), 8,174 HPSAs. Loaders: `load`, `load_fy`, `load_sites_2026`, `load_area`, `load_with_mcta`, `load_dental`, `load_mental`, `load_components`.
- Notes (2026-10-28): "no credible outcome"; the threshold is a funding priority, not an eligibility rule.

### cejst_tracts
- Treatment: CEJST v2.0 core rule, low-income percentile at or above 65 and a burden percentile at or above 90 (both-rule). Y is a placeholder of zeros. X: race, age, enrollment, state indicators.
- Source: CEQ CEJST 2.0 files from Internet Archive captures (January 2025); in hand (9 files), 73,273 tracts. Loader: `load`.
- Notes (2026-10-28): the rule matches 93.75% (the rest come from other routes). README verdict "weak": no outcome and the designation is no longer in use.

### calenviroscreen
- Treatment: SB 535 disadvantaged community, top 25% of the CalEnviroScreen 4.0 score (pollution burden times population characteristics). Y planned: Climate Investments dollars per tract. About 8,000 tracts. OEHHA serves a bot check, so the author must download the results spreadsheet into `data/raw/`.
- Data: none on this machine; no adapter loader. Notes (2026-10-27): waiting for the author.

### us_tracts_qct
- Treatment: HUD qualified census tract (share of households under 60% of area median income at or above 50, or poverty at or above 25; a population cap makes it fuzzy). Y planned: LIHTC units placed in service. About 74,000 tracts. huduser.gov blocks scripts.
- Data: none; no adapter loader. Notes (2026-10-27): waiting for the author.

### fannie_mae
- Fannie Mae loan performance data, a decision matrix on credit score and LTV like `freddie_mac`. Requires a free account. No data and no adapter loader; README says waiting for the author. The multi-rule note (2026-10-27) groups it with freddie_mac.

### yelp
- Displayed half-star rating (thresholds at x.25 and x.75 of the average review rating). Y planned: reviews or check-ins in later months; about 150,000 businesses. The Yelp Open Dataset needs a terms form. No data, no adapter loader; waiting for the author.

### kenya_tracking
- Duflo, Dupas and Kremer (2011): students below the school's median baseline score go to the lower section (group thresholds, one per school). Y: endline test scores. Source: Harvard Dataverse doi:10.7910/DVN/LWFH9U (guestbook). No data, no adapter loader; waiting for the author.

### mimic
- Stub: MIMIC-IV ICU severity score (SOFA, SAPS-II or APACHE-IV; threshold to be chosen), Y ICU length of stay or ventilator-free days. PhysioNet data-use agreement needed. `adapter.py` defines `load`; no data on this machine, n not stated.

### nlsy
- Stub: NLSY AFQT percentile with natural thresholds (10, 30, 50), Y log hourly wage or earnings (about 12,700 in NLSY79). `adapter.py` defines `load`; no data on this machine. Public extracts are free per the README.

### pop_eleches
- No README and no adapter. Holds the Pop-Eleches and Urquiola raw replication files (AER, six `.dta` files, do-file, readme PDF; 888 MB).
- Notes (2026-10-27, multi-rule note, citing the log of 2026-09-01): checked and not usable, no individual covariates.

### adams
- No README and no adapter. Holds only a license, two do-files and a README PDF from a replication package (128 KB, no data files). Not mentioned in any of the five notes; I could not tell which paper it is from without opening the PDF.

### simulations
- Not a dataset. Holds simulation and estimator notes (`TWO_SCORE_ESTIMATOR_20261027.md`, `TWO_SCORE_POPULATION_20261027.md`, `DS_*` checks, `dose_schedule/`) and no README or adapter.

## Project-level notes in this folder

| file | what it holds |
|---|---|
| `EDUCATION_ASSESSMENTS_SCAN_20260923.md` | Desk scan of NAEP, TIMSS and PISA (nothing downloaded): they are low-stakes, so none gives a score-to-treatment design; useful as Y or X, or for semi-synthetic benchmarks. Ranks school-meal Community Eligibility (with SEDA outcomes) first, and lists free-meal designs, ECLS-K:2024 and rejected options. The five-point screen (real threshold, covariates that move the score, later outcome, policy reason and cost, enough units) comes from here. |
| `CHILE_DATA_ACCESS.md` | (2026-09-29) Which Chile education sources are open (MINEDUC datos abiertos, DEMRE portal) and which need a request (PSU-era DEMRE files with MRUN, SIMCE student scores, subject grades, CAE loans, program earnings), with the route and Ley 20.285 timing. |
| `MULTI_RULE_DATASETS_20261027.md` | (2026-10-27 to 10-29) Datasets for group-threshold, either-or and two-score rules, in six rounds, with rule checks, readiness numbers, the 2026-10-28 decision to keep continuous Y, a waiting-for-the-author list and a not-usable list. Some of its waiting and "stub" entries are out of date (see nursing_homes, freddie_mac, hmda above). |
| `DOSE_SCHEDULE_DATASETS_20261028.md` | (2026-10-28, edited to 2026-11-02) Datasets for the continuous-dose extension: required features, method, one results table with a verdict per dataset, lessons, and an author to-do list of downloads. |
| `BINARY_OUTCOME_ROUTES_20261110.md` | (2026-11-10, addendum 2026-11-11) Exploratory run of four binary-outcome routes on GPA and Chile PAES, and the VLBW addendum. Paper and scope file unchanged; binary outcomes are outside the paper. |
