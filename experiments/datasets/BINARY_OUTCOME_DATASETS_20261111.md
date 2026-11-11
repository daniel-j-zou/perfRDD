# Binary-outcome settings for the threshold estimator: candidate scan (2026-11-11)

Written by Claude at the author's request, from web searches and page reads only. Nothing
was downloaded. The purpose is to find settings, beyond the ones already in the tree or
ruled out, in which the paper's model (score Q = gamma'X + eta, treatment D = 1{Q on one side
of a threshold}, a binary outcome Y measured after treatment) can be estimated from open
microdata, so that "where should the threshold be?" can be answered with a per-unit cost.

The six criteria, as agreed:

1. A threshold on a continuous score that a decision maker chooses (guideline, eligibility
   line, program rule), so the policy question is meaningful.
2. A treatment with a documented, meaningful effect and a plausible per-unit cost.
3. A binary outcome measured after treatment, base rate above about 3 percent.
4. Covariates that explain the score (R-squared of Q on X at least 0.2) and are plausibly
   independent of the residual eta.
5. Public unit-level microdata (no application or a light registration), tens of thousands
   of units, enough near the threshold.
6. An effect that plausibly fades or changes sign across the score.

Already in hand or ruled out before this scan (not repeated below): NCHS linked birth and
infant death files at 1500 g (`vlbw/`), Chile PAES 458 (`chile_admission/`), GPA probation
(`gpa/`), Lending Club, HMDA, OULAD, NHANES HbA1c, school accountability grades, FDIC capital
ratios, close elections, Hansen's BAC data, Cook and Kang birthdays, Colombia SPP.

Judgments on criterion 4 (R-squared) are my estimates from the literature, not computed;
each needs a check on the data. "Fuzzy" means the rule moves the treatment probability by
less than one.

## Ranked summary

| rank | candidate | rule (score, threshold, who sets it) | binary outcome (base rate) | n | access | criteria met / doubtful |
|---|---|---|---|---|---|---|
| 1 | SEER breast cancer with Oncotype DX Recurrence Score | RS 0-100, chemotherapy advised above 25 (TAILORx; guidelines), earlier lines 18 and 31 | death within 5 years (all-cause about 5-10 percent; breast-cancer death 2-3 percent) | 293,668 tumors with a score, 2004-2019 | SEER Research Plus (eRA Commons or HHS account, institutional email, 2 business days) plus a specialized-database agreement; US-based researchers only | 1, 2, 4, 5, 6 met; 3 doubtful (rare if breast-cancer death); treatment is the oncologist's decision, so fuzzy |
| 2 | Chile PSU 475 line for the state-guaranteed loan (CAE) and scholarships | PSU reading-math average, 475 points, set by law 20.027 (2006) | university enrollment (about 15 percent just below, 35 percent just above, Solis 2017); persistence to year 2 and 3 | about 200-250k test takers a year, 2007-2022, open files since the 2004 process | DEMRE open data portal (csv, no request; ID_aux links stages) and MINEDUC Datos Abiertos | all six plausible; 6 to check (effect may be flat in the score) |
| 3 | SBA 7(a) loans at the $150,000 guarantee notch | loan amount, $150,000; guarantee 85 percent at or below, 75 percent above (SBA rule) | charge-off (7-15 percent of resolved loans by cohort) | about 1.5 million loans 1991-2026 | data.sba.gov FOIA csv, public domain | 1, 3, 5 met; 2 (effect on default is small: Bachas, Kim and Yannelis find default smooth at the notch), 4 (loan size vs. X), 6 doubtful; bunching at the notch |
| 4 | NCHS linked files at the 5000 g macrosomia line | birth weight, 5000 g, rule of thumb in neonatal care (Brilli and Restrepo 2020) | NICU admission (12-14 percent, this is the treatment), death in 28 days (about 0.5 percent), 5-minute Apgar under 7, assisted ventilation | 55,581 singleton births within 227 g over 2007-2013; 2007-2022 now available | CDC NCHS public-use files (as for `vlbw/`) | 1, 4, 5 met; 2 (effects 1-2 pp on care, 0.2 pp on death), 3 (death is rare), 6 doubtful |
| 5 | Medicaid eligibility at 138 percent of the poverty line (expansion states) | income as a share of the poverty line, 138 percent, set by the ACA and state choice | Medicaid coverage (30 percent near the line), uninsured (about 15 percent), private non-group coverage | CPS ASEC: 16,582 adults 27-64 within 50 points, 2014-2016; ACS: ten times more; SIPP: monthly income | Census public-use files (free download, IPUMS registration) | 1, 2, 3, 5 met; 4 (income measured with error, annual vs. monthly), 6 doubtful; income is manipulable (SIPP evidence) |
| 6 | THEOP: Texas top 10 percent automatic admission | high-school class rank percentile, top 10 percent (Texas law 1998, later 6-7 percent at UT Austin) | enrollment at the applied university, graduation | several hundred thousand applications to 9 universities, 1990s-2002 | Princeton OPR archive public-use files (registration); ICPSR 29841 is restricted | 1, 2, 3, 5 met; 4 and the score itself doubtful: rank is coarsened in the public files (to verify); rank is a within-school percentile |
| 7 | PROGRESA/Oportunidades eligibility index | household poverty score from the 1997 census, regional threshold (program rule) | child enrolled in school (secondary about 65 percent), child works | 24,000 households, 506 villages, 1997-2000 | Harvard Dataverse record points to the old Oportunidades site, now offline; IFPRI copy marked "no longer available" | 1, 2, 3, 6 met; 4 fails in spirit (the score is a function of X); 5 doubtful (availability) |
| 8 | SNAP gross-income test at 130 percent of the poverty line | income as a share of the poverty line, 130 percent (federal rule, states raise it under BBCE) | SNAP receipt (40-60 percent among eligible), food insecurity (25-35 percent among low-income households) | CPS Food Security Supplement, about 40,000 households a year | Census public-use (free) | 1, 3, 5 met; 2 (no RD evidence at 130 found), 4, 6 doubtful; fuzzy (net-income and asset tests) |
| 9 | NYC restaurant inspection score, A grade at 13 points | inspection points, A if 0-13 (DOHMH rule, 2010) | closed by DOHMH within a year, next inspection below 14 | about 25,000 restaurants, 294,700 violation rows, rolling 3-year window | NYC Open Data (public) | 1, 3, 5 met; 2 (revenue effect small or nil, Meltzer et al. 2017), 4 and manipulation (bunching at 13, Ho 2012) doubtful |
| 10 | SEER lung cancer, adjuvant chemotherapy at 4 cm | tumor size in mm, 4 cm (CALGB 9633 subgroup; stage IB/IIA boundary) | death within 5 years (40-50 percent) | about 10,000 resected node-negative cases a year | SEER Research (email registration) | 1, 2, 3, 5 met; 4 fails (size barely explained by X); 6 plausible |
| 11 | ACA premium subsidy line at 400 percent of the poverty line | income share of poverty line, 400 percent (ACA, suspended 2021-2025) | uninsured, direct-purchase coverage | ACS, CPS | Census public-use | 1, 3, 5 met; 2 weak (effects small and imprecise), 4, 6 doubtful |
| 12 | Bankruptcy means test at the state median income | current monthly income relative to the state median (BAPCPA 2005, DOJ tables) | Chapter 13 instead of 7 (about 30 percent), discharge | about 1 million cases a year in the FJC Integrated Database | FJC public download | 1, 3, 5 met; 2 (Parra 2016 used hand-collected PACER data), 4 (household size and the means-test income may be missing from the IDB) doubtful |
| 13 | PPP loans at $150,000 (simplified forgiveness form 3508S) | loan amount, $150,000 (SBA rule, 2020-2021) | loan not forgiven (about 5-7 percent) | 11.5 million loans | data.sba.gov FOIA csv | 1, 3, 4, 5 met; 2 and 6 doubtful (no documented effect; loan size is 2.5 times payroll) |
| 14 | AFQT 31 enlistment line in the NLSY79 and NLSY97 | AFQT percentile, 31 (DoD quality floor) | ever served in the military (about 10 percent) | 12,686 and 8,984 respondents | NLS Investigator (free registration) | 1, 2 (Greenberg et al. 2023, Army records), 3, 4 met; 5 fails (small; the NLSY test is not the enlistment test) |

Top three to try first: 1 (SEER Oncotype), 2 (Chile 475), 3 (SBA 7(a)). Candidate 4 reuses
data already on this machine and is the cheapest check.

## 1. SEER breast cancer with the Oncotype DX Recurrence Score

- **Rule.** The 21-gene Recurrence Score (RS, 0-100) is reported for hormone-receptor
  positive, HER2-negative early breast cancer. After TAILORx (Sparano et al., NEJM 2018,
  [doi:10.1056/NEJMoa1804710](https://www.nejm.org/doi/full/10.1056/NEJMoa1804710)),
  guidelines advise chemotherapy plus endocrine therapy above RS 25 and endocrine therapy
  alone at RS 11-25 for women over 50; before 2018 the lines were 18 and 31. The threshold is
  a clinical choice and the question "where should it be?" is live: TAILORx itself moved the
  high-risk line from 31 to 26, and women 50 or younger with RS 16-25 showed some benefit.
- **Treatment and effect.** Chemotherapy. TAILORx: no benefit of chemotherapy at RS 11-25
  overall (9-year distant recurrence about 5 percent in both arms); benefit above 25; an
  exploratory age interaction at 16-25 ([NIH summary](https://www.nih.gov/news-events/nih-research-matters/test-predicts-whether-chemotherapy-will-help-early-stage-breast-cancer-patients)).
  The benefit rises with the score, which is criterion 6 in its cleanest form. Per-unit cost:
  chemotherapy's monetary and toxicity cost per patient.
- **Outcome.** Death within 5 years (all-cause 5-10 percent in SEER-linked cohorts of tested
  patients; breast-cancer-specific 2-3 percent), or distant recurrence (not in SEER).
  Criterion 3 is met for all-cause death, doubtful for cancer death.
- **Covariates.** Grade, tumor size, PR status, age, histology, nodal status, ER level. Grade
  and PR are strongly associated with RS; an R-squared of 0.2-0.3 is plausible. Independence
  of the residual from X is the usual question.
- **Data.** SEER specialized database "Invasive Breast Cancer with Oncotype DX Recurrence
  Score", 2004-2019 diagnoses, 293,668 tumors linked to a test result in the 22-registry
  file, follow-up through 2020; versions with county or census-tract attributes.
  [Request page](https://seer.cancer.gov/data/specialized/available-databases/breast-oncotype-dx-rs-request).
  Access: hold SEER Research Plus (eRA Commons or HHS account, institutional email, "the
  SEER program will process your request within 2 business days",
  [access page](https://seer.cancer.gov/data/access.html)), then submit a study plan and a
  signed specialized-database data use agreement; "available only to U.S. based
  researchers affiliated with U.S. institutions". Chemotherapy is in the treatment fields
  (first course only; a separate treatment-data limitations agreement). This is more than a
  light registration but needs no review of merit.
- **Judgment.** Criteria 1, 2, 4, 5, 6 met; 3 doubtful if the outcome is cancer death.
  Treatment is fuzzy (chemotherapy use rises with RS rather than jumping), so the local
  check at the line is a first-stage question before anything else. Published SEER-RS
  survival studies: [Breast Cancer Research 2019](https://link.springer.com/content/pdf/10.1186/s13058-019-1190-4.pdf).

## 2. Chile: PSU 475 line for the CAE loan and scholarships (Solis 2017)

- **Rule.** The state-guaranteed loan (CAE, law 20.027 of 2006) and most MINEDUC
  scholarships require a PSU reading-math average of at least 475 points (the loan's rule
  since 2006; [UDD note on the 2012 rules](https://www.udd.cl/wp-content/uploads/2012/12/Informativo_CRAE_2012.pdf),
  [DIPRES 2019](https://www.dipres.gob.cl/597/articles-206584_doc_pdf.pdf)). The threshold
  is a legislative choice; the current rules restate it on the PAES scale, so pool PSU years
  (2007-2022) first and check the PAES-era rule before adding 2023 onward.
- **Treatment and effect.** Loan and scholarship eligibility. Solis (JPE 2017,
  "Credit access and college enrollment",
  [working paper](https://eml.berkeley.edu/~saez/course131/solis.pdf)): enrollment jumps
  by nearly 20 percentage points at the line (from about 15 percent just below), and the
  effect persists into years 2 and 3; it removes the family-income gradient. Card and Solis
  (NBER w27269, [pdf](https://nber.org/system/files/working_papers/w27269/w27269.pdf)) use
  the same 475 line for retakers and persistence. Per-unit cost: the loan subsidy or
  scholarship per eligible student.
- **Outcome.** Enrolled in a university the next year (binary, about 15 percent below the
  line and 35 percent above); enrolled in year 2 (persistence); any higher-education
  enrollment.
- **Covariates.** NEM (school grades), ranking, school type and dependency, region, family
  income bracket, parental education from the socioeconomic file. In `chile_admission/` the
  12 score covariates explain PAES well; the same should hold for PSU (R-squared about 0.4).
  The 2026-09-29 notes found X independent of eta fails in spread for PAES; expect the same.
- **Data.** DEMRE "Bases de Datos Abiertos" portal
  ([demre.cl/portales/portal-bases-datos](https://demre.cl/portales/portal-bases-datos), usage
  guide [pdf](https://portaldemre.demre.cl/portales/datos-abiertos/documentos/documento-uso-datos-20241128.pdf)):
  csv files for every stage (inscription, scores, application, selection, enrollment) of
  every admission process since 2004, linked by the anonymous ID_aux, no request needed
  (identified MRUN data need an email request). MINEDUC Datos Abiertos has PDT and PAES
  files 2021-2026 ([page](https://datosabiertos.mineduc.cl/pruebas-de-admision-a-la-educacion-superior/))
  and the higher-education enrollment files already used by `chile_admission/`. Size: about
  200-250k test takers a year; the 475 line sits near the median, so the window is dense.
  Both hosts refused automated fetches today (connection refused; the MINEDUC host was down
  on 2026-10-29 too), so the author should download by hand.
- **Judgment.** All six plausible. The adapter is mostly written (`chile_admission/`); the
  new work is the 475 rule, the earlier years and the loan take-up variable (MINEDUC's
  student-benefit files, with mrun, give take-up if the author wants a fuzzy design).

## 3. SBA 7(a) loans at the $150,000 guarantee notch

- **Rule.** Loans of at most $150,000 carry a maximum SBA guarantee of 85 percent; larger
  loans 75 percent (temporarily 90 percent on both sides in 2009-2010). The notch is an SBA
  program parameter.
- **Treatment and effect.** The higher guarantee. Bachas, Kim and Yannelis (JFE 2021,
  "Loan guarantees and credit supply", [pdf](https://cmepr.gmu.edu/wp-content/uploads/2019/08/Bachas-Yannelis-Loan-Guarantees-and-Credit-Supply-1.pdf),
  [publisher page](https://faculty.chicagobooth.edu/constantine-yannelis/research/loan-guarantees-and-credit-supply))
  find strong bunching just below $150,000 and a credit-supply elasticity near 5; they
  report that "interest rates and ex-post charge-off rates trend smoothly through the
  threshold". So the documented effect is on lending volume, not on default; for a binary
  default outcome the effect at the line is small by their account. Cost: the expected
  guarantee payout per loan.
- **Outcome.** Charge-off (LoanStatus CHGOFF). Third-party tallies of the FOIA files give
  6.9 percent of FY2010-2015 loans and 7.9 percent of resolved FY2010-2019 loans
  ([example](https://ctacquisitions.com/?p=56495)); 2000s cohorts are higher.
- **Covariates.** Term, NAICS, state, lender, business type and age, jobs supported, fiscal
  year, interest rate. Loan size is partly explained (R-squared 0.2-0.3 plausible, term and
  jobs carry most of it), but bunching means the score is manipulated near the line, which
  the paper's density assumptions do not allow without a donut.
- **Data.** [data.sba.gov 7(a) and 504 FOIA](https://data.sba.gov/dataset/7a-504-foia):
  four 7(a) csv files (FY1991-1999, 2000-2009, 2010-2019, 2020-present, "as of September 30,
  2026"), public domain, quarterly updates, data dictionary xlsx. Fields (from the
  dictionary as summarized by users): GrossApproval, SBAGuaranteedApproval, ApprovalDate,
  TermInMonths, NaicsCode, BusinessType, BusinessAge, LoanStatus (PIF, CHGOFF, CANCLD,
  EXEMPT, COMMIT), ChargeOffDate, GrossChargeOffAmount, JobsSupported, lender and location.
  About 1.5 million loans.
- **Judgment.** 1, 3, 5 met; 2, 4, 6 doubtful. Worth a quick look because the file is one
  download and the guarantee cost is a real per-unit cost, but the expected answer is "no
  effect on default at the line", which makes it a negative example rather than an
  application.

## 4. NCHS linked birth and infant death files at 5000 g (macrosomia)

- **Rule.** Birth weight at 5000 g. Brilli and Restrepo (Economics and Human Biology 2020,
  [doi:10.1016/j.ehb.2019.100825](https://doi.org/10.1016/j.ehb.2019.100825); working paper
  [Verona 01/2019](https://ideas.repec.org/p/ver/wpaper/01-2019.html)) show that neonatal
  care follows a rule of thumb at 5000 g and at no lower macrosomic line. Like 1500 g, the
  line is a clinical convention, not a statute.
- **Treatment and effect.** Extra neonatal care. From the paper's text: the sample is
  "U.S. singleton births in 2007-2013 with birth weight between 4773 and 5227 grams",
  55,581 observations; NICU admission rises by about 2 percentage points at the line (from
  about 12 percent), antibiotics by 1 point, and 28-day mortality falls by 0.2 points.
- **Outcome.** 28-day or one-year death (rare, around 0.4-0.6 percent in this weight range:
  criterion 3 fails), or the care outcomes themselves: NICU admission (12-14 percent),
  assisted ventilation, 5-minute Apgar below 7, which are measured at or just after birth
  and are partly the treatment.
- **Covariates.** As in `vlbw/`: gestation, sex, plurality, mother's age, parity, diabetes
  (a strong predictor of macrosomia), education, race, prenatal care. R-squared at least 0.2
  is plausible with gestation and diabetes.
- **Data.** CDC NCHS public-use linked birth and infant death files (birth-cohort files
  2007-2022, all public, as used for `vlbw/`). About 8,000 births a year within 227 g of
  5000 g, so 2007-2022 gives about 120,000.
- **Judgment.** 1, 4, 5 met; 2, 3, 6 doubtful. Cheap to try because the files are on this
  machine; the honest expectation is a small, noisy effect and an edge optimum.

## 5. Medicaid eligibility at 138 percent of the poverty line

- **Rule.** In expansion states, adults with modified adjusted gross income up to 138
  percent of the federal poverty line are eligible for Medicaid; just above, they get
  marketplace subsidies. The line is a legislative choice (and states chose whether to
  expand).
- **Treatment and effect.** Medicaid eligibility instead of subsidized private coverage.
  Barcellos, Jacobson and Levy (AJHE 2023, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11068085)):
  CPS ASEC 2014-2016, expansion states, adults 27-64, local linear with a 50-point bandwidth,
  Medicaid coverage falls by 5.5 points (base 30.4 percent) at the line and private non-group
  coverage rises by 3.6; uninsurance does not change; no heaping or manipulation found.
  Hinde (AJHE 2017, [RePEc](https://econpapers.repec.org/RePEc:tpr:amjhec:v:3:y:2017:i:3:p:346-369))
  finds a 5.4-point rise in private coverage above 138 percent. Cost: the public cost per
  Medicaid enrollee net of the subsidy.
- **Outcome.** Insured (binary, uninsured about 15 percent near the line), Medicaid
  coverage, private non-group coverage, or self-reported fair/poor health.
- **Covariates.** Education, age, hours and weeks worked, family size, occupation,
  state; income relative to poverty is explained with R-squared about 0.3-0.4 in the CPS.
  The residual's independence of X is doubtful because reported income carries error that
  correlates with work status.
- **Data.** CPS ASEC (free from Census or [IPUMS CPS](https://cps.ipums.org), registration),
  about 16,000 adults in the 88-183 percent window for three years; ACS PUMS gives about ten
  times more per year, but with annual income (noisier). The SIPP measures monthly income;
  a 2025 preprint ([arXiv 2506.12976](https://arxiv.org/abs/2506.12976)) shows childless
  adults in expansion states lower their lowest monthly earnings to qualify, so the SIPP
  shows manipulation that the annual surveys hide.
- **Judgment.** 1, 2, 3, 5 met; 4 and 6 doubtful. The documented effect is a change of
  coverage type rather than of being insured, so the binary outcome should be "Medicaid"
  or "any coverage" and the cost side carries the argument.

## 6. THEOP: Texas top 10 percent automatic admission

- **Rule.** Texas HB 588 (1997, in force from 1998) guarantees admission to any public
  university to seniors in the top 10 percent of their high-school class; UT Austin later
  capped it (top 6-8 percent). The score is class rank percentile; the threshold is set
  by the legislature and is a live policy question.
- **Treatment and effect.** Guaranteed admission (and the signal of it). Niu and Tienda
  (2010) and later work with THEOP show enrollment effects at the flagship; Black, Denning
  and Rothstein (2023, AEJ Applied) find effects on degrees and earnings using restricted
  state data.
- **Outcome.** Enrollment at the applied university, graduation (transcripts for enrollees).
- **Covariates.** SAT/ACT, high-school type and location, share of economically
  disadvantaged students, race, sex. Rank is a within-school percentile, so school fixed
  effects and test scores explain it moderately.
- **Data.** THEOP administrative public-use data at the Princeton OPR archive
  ([oprdata.princeton.edu/Archive/THEOP](https://oprdata.princeton.edu/Archive/THEOP)):
  application and transcript records from seven public and two private Texas universities,
  "coverage begins several years prior to the implementation of the Texas top 10% law in
  1998, and extends until at least 2002"; the Texas A&M file covers all first-time freshman
  applicants 1992-2002. Confidentiality measures include "collapsing small frequency cells,
  and setting values to missing". The ICPSR copy (29841,
  [page](https://www.icpsr.umich.edu/web/ICPSR/studies/29841)) is fully restricted. The
  Princeton pages refused automated reads today (403), so the registration terms and the
  coding of class rank (decile bins or percentile) must be checked by hand.
- **Judgment.** 1, 2, 3, 5 met; the score may be too coarse in the public files (4 and the
  continuity of Q doubtful).

## 7. PROGRESA/Oportunidades eligibility index

- **Rule.** Households in program villages were classified poor by a discriminant score
  built from the 1997 household census (ENCASEH), against a regional threshold chosen by
  the program.
- **Treatment and effect.** Cash transfers conditional on school attendance. Buddelmeyer
  and Skoufias (World Bank PRWP 3386, 2004,
  [pdf](https://openknowledge.worldbank.org/bitstreams/affbca72-21d9-5a2c-a365-95dafa7723a5/download))
  compare RD at the score threshold with the experimental estimates and agree in 10 of 12
  cases; Schultz (2004) gives the experimental enrollment effects (secondary enrollment up
  by 6-9 points). Cost: the transfer per household.
- **Outcome.** Child enrolled in school (secondary about 65 percent at baseline), child
  works, health visits.
- **Covariates.** The score is itself a function of household assets and demographics, so
  X explains Q almost exactly and the residual is tiny; one would have to hold back part of
  the index inputs, which breaks the model's spirit. R-squared is not the problem; the
  absence of a genuine eta is.
- **Data.** Harvard Dataverse record "Mexico, Evaluation of PROGRESA"
  ([doi:10.7910/DVN/05BMJY](https://doi.org/10.7910/DVN/05BMJY)) points to the former
  Oportunidades evaluation site, which no longer responds; the IFPRI/AGRIS copy is marked
  "no longer available" ([record](https://agris.fao.org/search/en/providers/122566/records/669fae3d4295623862502c8e)).
  24,000 households, 506 villages, waves 1998-2000.
- **Judgment.** 1, 2, 3, 6 met; 4 fails in spirit; 5 doubtful until a live mirror is
  found. Listed because it is the classic score-threshold program; not a first choice.

## 8. SNAP gross-income test at 130 percent of the poverty line

- **Rule.** Federal gross-income limit of 130 percent of poverty (net 100 percent, asset
  test); states raise the gross limit to up to 200 percent under broad-based categorical
  eligibility. Chosen by Congress and the states.
- **Treatment and effect.** SNAP eligibility. No RD at 130 percent with the CPS was found;
  the PSID fuzzy RD ([PMC9268983](https://pmc.ncbi.nlm.nih.gov/articles/PMC9268983)) and
  state-policy instruments ([NBER paper on older adults](https://conference.nber.org/conf_papers/f212923.pdf))
  are the closest. The food-security effect of SNAP is contested.
- **Outcome.** SNAP receipt (take-up), household food insecurity (25-35 percent among
  low-income households; USDA measure from the December CPS supplement).
- **Covariates.** As for Medicaid; the rule is fuzzier (net income, assets, BBCE).
- **Data.** CPS Food Security Supplement, about 40,000 households a year, public.
- **Judgment.** 1, 3, 5 met; 2, 4, 6 doubtful. Weaker than candidate 5 on every count.

## 9. NYC restaurant inspection score and the A grade

- **Rule.** Since 2010 DOHMH posts letter grades: A for 0-13 points, B for 14-27, C for 28
  or more. The score is violation points; the bands are the department's choice.
- **Treatment and effect.** Posting a B instead of an A. Meltzer, Rothbart et al. (Syracuse
  CPR WP 202, 2017, [pdf](https://www.maxwell.syr.edu/uploadedFiles/cpr/publications/working_papers2/wp202.pdf);
  Public Finance Review 2019) find no lasting revenue effect of grading; Rothbart, Silver and
  Bae use an RD on adjudication and find grades are contested and changed near the lines.
  Ho (2012, "Fudging the nudge") documents bunching just below 14, so the score is
  manipulated at the line.
- **Outcome.** Closed by DOHMH within 12 months (ACTION field), next cycle's grade.
- **Covariates.** Cuisine, borough, chain, previous score; moderate R-squared with the
  previous score.
- **Data.** [NYC Open Data 43nn-pn8j](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j):
  31 columns including SCORE, GRADE, ACTION ("closed/re-opened by DOHMH"), INSPECTION TYPE,
  CUISINE DESCRIPTION, BORO; 294,700 violation rows; only the last three years are kept, so
  older cycles need archived snapshots.
- **Judgment.** 1, 3, 5 met; 2 weak, manipulation at the line, rolling window. Low rank.

## 10. SEER lung cancer: adjuvant chemotherapy at 4 cm

- **Rule.** Adjuvant chemotherapy after resection of node-negative NSCLC is advised for
  tumors above 4 cm (CALGB 9633 subgroup; the 8th-edition staging moves 4-5 cm to IIA).
- **Treatment and effect.** CALGB 9633 found benefit above 4 cm and none below
  ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7499246)); NCDB analyses disagree on
  whether size predicts benefit ([JAMA Oncology 2016-type analyses via PubMed 27287414](https://www.pubmed.gov/27287414/)).
  So criterion 6 is the clinical question itself.
- **Outcome.** Death within 5 years (40-50 percent).
- **Covariates.** Age, sex, histology, grade, visceral pleural invasion, nodes examined.
  Tumor size is poorly explained by these (R-squared well under 0.2): criterion 4 fails.
- **Data.** SEER Research (email registration), tumor size in mm, chemotherapy yes/no,
  survival; about 10,000 resected node-negative cases a year.
- **Judgment.** 1, 2, 3, 5 met; 4 fails. Listed to record the check.

## 11. ACA premium subsidy line at 400 percent of the poverty line

- Same data and covariates as candidate 5. The subsidy stops at 400 percent (2014-2020 and
  again from 2026); RD evidence is weak: "weak evidence for changes at 400% FPL"
  ([AEA 2017 preliminary paper](https://aeaweb.org/conference/2017/preliminary/paper/Bn7KksRZ)),
  a small shift in state-exchange states ([APPAM 2019](https://appam.confex.com/appam/2019/webprogram/Paper31810.html)).
  Outcome: uninsured or direct-purchase coverage. Judgment: 1, 3, 5 met; 2, 4, 6 doubtful.

## 12. Bankruptcy means test at the state median income

- **Rule.** Since BAPCPA (2005) a filer whose current monthly income exceeds the state
  median for the household size faces the means test, which can bar Chapter 7; the medians
  are published by the DOJ ([CRS summary](https://www.everycrsreport.com/reports/RS22058.html)).
- **Treatment and effect.** Chapter 7 access. Parra (2016, "How does consumer bankruptcy
  protection impact household outcomes?", [AEA 2017 paper](http://www.aeaweb.org/conference/2017/preliminary/paper/ZkNEty7Y))
  runs a fuzzy RD at the thresholds on more than 40,000 hand-collected PACER petitions
  (2006-2009, 65 districts) and finds Chapter 7 raises later business creation, secured
  borrowing and homeownership; he notes that public survey data cannot show the
  discontinuity.
- **Outcome.** Filed under Chapter 13 rather than 7 (about 30 percent), case discharged.
- **Covariates.** Assets, liabilities, district, year, pro se; whether the public file holds
  current monthly income and household size is not confirmed.
- **Data.** FJC Integrated Database, bankruptcy snapshots FY2008-2026, SAS and text, free
  download ([page](https://www.fjc.gov/research/idb/bankruptcy-cases-filed-terminated-and-pending-fy-2008-present));
  the page names chapter, county and zip but not income fields; the codebook must be read.
- **Judgment.** 1, 3, 5 met; 2 and 4 doubtful. The outcome of interest (post-filing
  behavior) is not in the IDB.

## 13. PPP loans at $150,000

- **Rule.** Loans of $150,000 or less could use the one-page forgiveness form 3508S
  ([SBA form](https://www.sba.gov/document/sba-form-3508s-ppp-3508s-loan-forgiveness-application-instructions)).
- **Treatment and effect.** Less documentation. No published RD or bunching study on
  forgiveness at this line was found.
- **Outcome.** Loan not forgiven (about 5-7 percent) or forgiven late.
- **Covariates.** Jobs reported, NAICS, business type, lender, state: loan size is 2.5
  times monthly payroll, so R-squared is high.
- **Data.** [data.sba.gov PPP FOIA](https://data.sba.gov/dataset/ppp-foia): 13 csv files
  (one above $150k, twelve up to $150k, as of 2024-09-30) and a data dictionary, public
  domain; forgiveness amount and date are in the files.
- **Judgment.** 1, 3, 4, 5 met; 2 and 6 doubtful. A one-afternoon check at most.

## 14. AFQT 31 in the NLSY79 and NLSY97

- **Rule.** DoD requires at least 96 percent of recruits to score 31 or above and 60
  percent at 50 or above, so the Army seldom takes applicants below 31.
- **Treatment and effect.** Enlistment. Greenberg, Gudgeon, Isen, Miller and Patterson
  (QJE 2023, [RePEc](https://ideas.repec.org/a/oup/qjecon/v137y2023i4p2363-2418..html)):
  crossing 31 raises enlistment by 10 points and raises earnings by over $4,000 a year for
  19 years; Army applicant records linked to tax data, restricted.
- **Outcome.** Ever served (about 10 percent of NLSY79 men).
- **Covariates.** Schooling at test, parental education, family background: R-squared
  about 0.4-0.5.
- **Data.** NLSY79 (12,686) and NLSY97 (8,984), free at [NLS Investigator](https://www.nlsinfo.org/investigator).
- **Judgment.** 1, 2, 3, 4 met; 5 fails (a few hundred respondents near 31; the NLSY AFQT
  is from a norming test, not the enlistment test, so the rule is very fuzzy).

## Settings where a zero cost is reasonable (added 2026-11-11)

Written after the pooled VLBW run, at the author's request ("think if we have settings where
it is reasonable for the cost to be 0"). With c = 0 an interior optimum needs the effect of
treatment on Y to change sign across the score, so the harm of treating the wrong units must
already be inside Y. A transfer of money or care to units that can only gain (GPA, the Chile
lines, SNAP, Medicaid, VLBW) never does that and always needs a cost. Three families do.

**Treatments with their own toxicity** (the outcome nets the harm):

- Chemotherapy by the Oncotype score (number 1 above): TAILORx found no benefit below 25 and
  real toxicity, so survival to five years should cross zero inside 11-40. The cleanest case;
  waits for the author's SEER application.
- Adjuvant chemotherapy by tumor size, SEER lung (number 10): same logic, but the size is
  barely explained by X.
- Transfusion by hemoglobin threshold (7 versus 8 g/dL; restrictive-strategy trials): MIMIC-IV
  on PhysioNet, credentialed access; the rule is fuzzy and the score is a lab value.
- Induction or cesarean by estimated fetal weight (the 4000-4500 g macrosomia guideline): the
  maternal harm is visible only in a composite Y, and the NCHS file has the actual weight,
  not the estimate the decision used.

**Assignment between two arms that cost about the same** (which arm, not whether to spend):

- Remedial versus regular placement by a placement-test score (Texas and Florida studies;
  Scott-Clayton and Rodriguez 2015, Martorell and McFarlin 2011): remediation delays the
  students who did not need it and may help the weakest; Y = on-time degree. Data restricted
  in those states; a public variant is not known yet.
- Selective admission by class rank or exam score, THEOP (number 6): mismatch can hurt
  marginal admits; Y = graduation. Public-use files at the Princeton OPR archive with
  registration (Claude can register); the rank is coarsened, to verify.
- Vocational versus academic track by score (Chile: university versus technical enrollment
  could be recast this way from the files in hand, but that is not the money line, and the
  458 line shows a one-sided jump).
- Chapter 7 versus Chapter 13 by the bankruptcy means test (number 12): the public IDB file
  lacks household size; would need PACER-type data.

**Outcomes defined net of the treatment's downside:** a loan-approval rule with Y = no
default or bankruptcy afterwards; a UI extension with Y = reemployed at a better wage. The
SBA file (`sba_7a/`) has Y = repaid, but its treatment is the guarantee level, not approval,
so the effect never turns negative.

For the settings built so far the answer is no: none has a sign-changing effect. The two
candidates that remove the cost question are Oncotype (author's application) and THEOP
(registration).

## Scan 2 (2026-11-11, evening): education and social science, public data only

At the author's request ("there should be many education or social science settings with
binary Y"). Three subagents (Sonnet) searched education, social policy and open microdata
portals, with the rule: unit-level files a US researcher can get as a public download or a
light registration, no study-plan application. Nothing was downloaded. The education agent's
own conclusion ("no candidate fully meets the requirements") is too pessimistic: it missed
the Chilean per-student files, which the portal agent found, and it wrongly reported that
MINEDUC has no MRUN-level enrollment file (we hold two). Claude's compilation follows.

| # | Setting | Score, threshold (who sets it) | Treatment | Binary Y | Access | Sign change (c = 0 plausible)? | Verdict |
|---|---|---|---|---|---|---|---|
| S1 | **Chile grade retention at the 4.0 pass mark** (Diaz, Grau, Reyes and Rivera 2021 on crime; promotion rules in Decreto 67) | annual grade average `PROM_GRAL` (1-7); promoted if at least 4.0 with no failed subject, 4.5 with one, 5.0 with two (ministry rule) | repeat the grade | enrolled next year (dropout), on-time secondary completion, later higher-education enrollment | MINEDUC open data, `Rendimiento por estudiante`: the folder `chile_retention/` already exists with all years 2002-2025 downloaded on 2026-09-28 (no new download needed); linked by MRUN | yes: retention helps some and pushes others out; which year of school, not whether to spend | **built and run (2026-09-28 to 2026-11-11)**; n in the hundreds of thousands per grade-year; X = prior-year grades and attendance by MRUN; check the rule's subject-failure branches; earlier checks found local ITT effects on the binary outcomes near zero (-0.3 to -1.1 pp) with heaping and sorting at 4.5; the routes (setting `retention`, 4.45 mirrored) give boundary or small interior optima with global effects at the cutoff 4 to 7 pp negative against local jumps within +-0.5 pp, so not usable |
| S2 | Chile program-specific admission cutoffs (PAES `Postulantes` and `Matricula` files, in `chile_admission/data/raw`) | program-weighted score against the last admitted score of the program | admitted to the program applied to, else the next choice | enrolled; enrolled in year 2 | in hand (see `../MULTI_RULE_DATASETS_20261027.md`) | yes: which program, not whether | multi-rule setting; a single-rule version takes one large program |
| S3 | North Carolina structured sentencing: prior record level and offense class, active prison versus probation (Rose and Shem-Tov 2021 used the grid; NIJ study ICPSR 38817 used the public files) | prior record points (levels I-VI at 2, 6, 10, 14, 18 points) within an offense class, set by the legislature | active sentence versus intermediate or community punishment | reincarcerated or new commitment within 3 years, from the offender ID across files | NC DAC bulk files (12 tables, 35-235 MB), public; `opus.doc.state.nc.us/robots.txt` is `Disallow: /`, so a download needs the author's instruction; the field list is inside the zips (the structure PDF is a one-slide diagram) | yes: incarceration versus supervision | promising but unverified: prior record points may appear only as the level; score integer-valued |
| S4 | Transfusion at hemoglobin 7 g/dL in intensive care (Bosch et al. 2022, RD in MIMIC-IV and eICU) | lowest hemoglobin, 7.0 g/dL (guideline) | red-cell transfusion (fuzzy, +20 points) | in-hospital death | PhysioNet credentialing: CITI course plus a data use agreement, one to two weeks, no study plan; code release required with publications | yes: transfusion has its own harms | light credentialing, not an application; n about 11k (MIMIC) and 14k (eICU) in the RD samples; X explains hemoglobin weakly |
| S5 | FEMA Individuals and Households Program registrations (OpenFEMA v2) | verified property loss, FEMA eligibility lines | assistance eligibility or SBA referral | `ihpEligible`, `sbaApproved` (decisions at registration, not later outcomes) | public CSV or Parquet, 26 million rows, no registration | no: a transfer | fresh but the outcomes are decisions, not post-treatment; one-sided |
| S6 | GSE mortgage files at the 620 credit line (Keys et al. 2010 used non-agency data) | FICO at origination | pricing or securitization terms | serious delinquency | Fannie Mae / Freddie Mac loan-level files, free registration, 56 million loans | loosely | the files hold only purchased loans, so the 620 line is a selection margin; LTV lines are deterministic in X |
| S7 | Broward COMPAS (ProPublica) | decile score, bands at 5 and 8 | risk label | two-year recidivism | public GitHub file | no clear rule-based treatment | n about 10k; weak |
| S8 | Brazil ENEM 450 for FIES and ProUni; SISU cutoffs | ENEM score | aid or admission | enrollment | INEP public score files; no student-level link to enrollment after the LGPD re-release; SISU portal returned 403 | one-sided (aid) / two-arm (SISU) | no linkage; dead end unless the SISU file carries score and admission status |
| S9 | Colombia Saber 11 to Saber Pro linked file (Mendeley, CC BY) | Saber 11 global score | none in the file | module scores (continuous) | public | n/a | a linkage resource, not a design |
| S10 | US state merit aid (HOPE, Bright Futures, PROMISE), remedial placement (Texas, Florida, Tennessee, CUNY), retention and exit exams (Florida, Chicago, NYC, Massachusetts), exam schools (Boston, NYC, Chicago), gifted programs, COMIPEMS, Romania, Kenya, Ghana, Turkey, Korea, gaokao, Beca 18 | various | various | graduation, enrollment, dropout | all restricted (state, district or ministry agreements), including the Chicago exam-school replication archive (403, likely restricted) | remediation, retention and selective admission are two-sided | the famous designs; not obtainable without agreements |
| S11 | US Sentencing Commission datafiles | criminal history category, offense level | sentence | none after sentencing in the public file | public | n/a | no outcome |
| S12 | Florida OBIS public database (Mitchell et al. 2017 used the 44 sentence-point line) | total sentence points, 44 (statute) | prison versus non-prison | reconviction within 3 years | public Access database or text files; `fdc.myflorida.com/robots.txt` disallows generic clients; the scoresheet points are not known to be in the public file | yes | unverified; likely missing the score |

Portals that yielded nothing public: Brazil INEP linkage, Peru (Beca 18, ECE), Kenya KCPE, Romania,
Colombia SISBEN, Italy INVALSI, UK LSYPE and National Pupil Database, Bondora, NLSY97, ELS:2002,
HSLS:09, data.ny.gov parole, pretrial PSA, Italian and Czech procurement. Not checked for
budget: Mexico (COMIPEMS, PLANEA), Chile SIMCE and PACE, India, Indonesia, Turkey, the Nordic
registers (restricted), IPUMS, Add Health, ECLS, BPS, NPSAS, PSID, NSDUH, BRFSS, MEPS, SIPP,
Project Data Sphere, Vivli, BioLINCC, Fragile Families, World Bank microdata, DHS, LSMS.

**Reading.** Public unit-level files with a chosen threshold, a treatment with a two-sided
effect and a later binary outcome exist, but they are few and they are the ones already
near this project: the Chilean per-student files (S1, S2) are the only open education source
that links a score to a later outcome by a student ID; the NC offender files (S3) are the one
open social-policy source with a two-arm rule, pending the field check. S1 is the
recommendation: the retention rule is a legislative choice, the treatment is which grade the
student sits in next year (so c = 0 is the natural reading), the base rates are far from the
ceiling (dropout a few percent per year, completion about 80 percent), and the prior-year
grade average explains the current one well. Step one is two browser downloads
(`Rendimiento` for one year and `Matricula por estudiante` for the next) to confirm MRUN and
the columns `PROM_GRAL`, `ASISTENCIA`, `SIT_FIN`.

## Not usable (checked and rejected)

- **NCHS 2500 g low-birth-weight line.** Same files as `vlbw/`; Almond, Doyle, Kowalski and
  Williams (2010) find no mortality discontinuity at 2500 g, so criterion 2 fails.
- **Blood donor hemoglobin deferral (12.5 g/dL women, 13.5 men).** Clean rule and return
  outcome (Bruhin et al. 2020, Swiss Red Cross Zurich, 80,060 donors; Rosenman et al.,
  [arXiv 1910.02170](https://arxiv.org/abs/1910.02170)), but the data are not deposited and
  staff manipulate readings at the line. US REDS data need a BioLINCC application.
- **Army applicant AFQT records (Greenberg et al. 2023).** Restricted linked tax data.
- **Florida Criminal Punishment Code (44 points, prison presumptive).** Scoresheets are
  not a public download; recidivism needs DOC linkage.
- **COMPAS (ProPublica, 7,214 cases) and the NIJ recidivism challenge (Georgia, 25,835).**
  Scores are integer deciles (1-10), not continuous; COMPAS is small.
- **US Sentencing Commission drug-quantity thresholds.** Quantities are continuous and
  public, but the file has no post-sentence outcome (no recidivism); prison vs. no prison
  is the sentence itself.
- **NIH paylines (Jacob and Lefgren 2011).** Scores of unfunded applications are not
  public.
- **Spain MCVL (UI eligibility at 360 contribution days), German SIAB, Austrian ASSD,
  UK NPD GCSE lines (Machin, McNally and Ruiz-Valenzuela 2020), Massachusetts MCAS 220,
  Texas and Florida grade retention, Tennessee and West Virginia merit scholarships,
  Texas TASP and Florida CPT remediation lines.** Administrative data by application only.
- **UNOS/SRTR (MELD, KDPI), PLCO (PSA 4), BioLINCC trials, MIMIC-IV, NSQIP, NCDB.** Data
  use agreements with review or credentialing; MIMIC is already a stub in the tree.
- **Brazil ENEM 450 for FIES, Colombia Saber 11.** No outcome after assignment in the
  public files (as recorded on 2026-10-27).
- **Prosper loan listings.** Credit score given in 20-point bins; the grade bands are the
  firm's, and funded listings are a selected sample.
- **NHANES with linked mortality (LDL 190, blood pressure 140, eGFR 60).** Treatment is
  the physician's downstream choice, the score is barely explained by survey covariates,
  and a few hundred people sit near each line.
- **Canada EI variable entrance requirement (420-700 hours).** The EICS public file is
  about 10,000 a year; too small near each regional threshold.
- **Immigration points tests (Canada CRS draws, Australia 65 points).** Microdata are in
  restricted research centers only.
- **Macrosomia at 4000 or 4500 g (cesarean).** No RD evidence; Brilli and Restrepo find
  no change in care below 5000 g.
