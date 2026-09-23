# NAEP / TIMSS / PISA dataset scan for PerfRDD (Claude, 2026-09-23)

Desk scan only. Nothing was downloaded, ingested, or screened. Each candidate is judged
against what the differing-slopes model needs:

1. a real deployed cutoff on a scalar score, `D = 1{Q > phi_0}`;
2. covariates that move `Q` at fixed `eta`, so `T = gamma' X~` has enough spread for
   overlap in `eta`, with `X ⊥ eta` plausible;
3. an outcome measured after assignment;
4. a real reason to ask about a different cutoff, and a cost `c` we can explain;
5. enough units: the screens in `RESEARCH_LOG.md` needed roughly 250k rows before a flat
   utility curve stopped producing false interior optima.

## Bottom line

NAEP, TIMSS, and PISA are low-stakes, cross-sectional, matrix-sampled assessments. Their
scores are never used to assign anyone to anything, so none of them gives a
`Q -> D -> Y` design with the assessment score as `Q`. The reporting cutoffs (NAEP
achievement levels, TIMSS benchmarks, PISA proficiency levels) are labels, not
treatments. The assessments are useful in two other roles:

- **as `Y`**, when some other policy score assigns treatment to schools (NAEP, or SEDA,
  which puts state tests on a NAEP-linked scale);
- **as `X`**, the prior-achievement covariate that explains a later high-stakes score (the
  PISA-based LSAY cohorts).

They also give a public, realistic joint law for `(X, Q)` for semi-synthetic simulations.

## Ranked candidates

### 1. School-meal Community Eligibility Provision (CEP), with NAEP-linked outcomes (most promising)

| Role | Mapping |
|---|---|
| Q | school Identified Student Percentage (ISP, the share of students directly certified) |
| phi_0 | 40% (2014-15 to 2023); lowered to 25% by USDA final rule, effective 2023-10-26 |
| D | CEP eligibility (intent-to-treat; adoption is optional) |
| X | CCD/ACS school characteristics: race shares, urbanicity, enrollment, district SAIPE poverty, state |
| Y | school achievement: SEDA 5.0 school files (grades 3-8, 2008-09 to 2018-19, public), or restricted NAEP student data linked to schools |

Why it fits:
- The cutoff has actually moved, and moving it further is live policy. USDA lowered it from
  40% to 25% in 2023. A 2025 House reconciliation proposal would have raised it to 60%, but
  that change was left out of the enacted bill. The policy question is the paper's target:
  which eligibility cutoff maximizes net benefit?
- Demographics should predict ISP well, which gives the spread in `T` that overlap needs.
  `eta` is residual direct-certification poverty.
- The cost can be derived rather than chosen. The federal free-rate reimbursement share is
  `min(1, 1.6 x ISP)`, so the unreimbursed meal cost per student falls as ISP rises. This
  gives a structural reason for `W - c` to cross zero, which no dataset screened so far has
  had. Using it needs a small extension: a known cost function `c(Q)` instead of a scalar
  `c`. The extra utility term `E[c(Q) 1{Q > phi}]` is a plain sample mean.

Risks to check first:
- **Truncated running variable.** Published CEP lists (FRAC/CBPP databases, state lists) cover
  only eligible and near-eligible schools (15-25% counts as near-eligible after 2023). A sample
  cut on `Q` induces dependence between `X` and `eta`, as the taxi fare cut did. An untruncated
  proxy exists: CCD direct-certification counts from 2016-17 onward, reported by some states.
- **Group and district elections** blur school-level eligibility. Eligibility is still sharp as
  a function of the school's own ISP only when the school elects individually.
- **Clustering** by district and state; the sampling unit is the school.
- Evidence on effect sizes is small. Ruffini (JHR) finds about +0.02 SD in math in the most
  affected districts, scaling to about +0.07 SD per newly eligible student.

Related: Title I schoolwide eligibility at a 40% poverty rate (lowered from 75% to 50% in
1994 and to 40% under NCLB; ESSA lets states waive it). Same data pipeline, but the poverty
measure varies by district, the design is fuzzy, and prior RD evidence finds null or
negative effects, which points to a boundary optimum. This is a secondary candidate.

### 2. PISA-based semi-synthetic benchmark (immediately usable; for simulations, not an application)

Take `Q` = a PISA or TIMSS math plausible value and `X` = background-questionnaire
variables (ESCS, parental education, books at home, gender, immigrant status, home
language, grade), both from the public microdata. Simulate `D = 1{Q > phi_0}` and outcomes
with known `a`, `b`, and `beta_2`. Unlike the current Gaussian, t5, and mixture batteries,
this tests:
- a realistic `X`-`Q` relationship and overlap window;
- discrete and mixed covariates (the Bahadur lemma's mixed-covariate condition);
- heteroskedastic residuals by SES, a realistic departure from `X ⊥ eta`;
- school clustering (PISA samples about 35-42 students per school);
- survey weights, and plausible values as measurement error in `Q`.

It is public and free, and pooling countries gives hundreds of thousands of rows.

### 3. LSAY (Australia): PISA at age 15 -> ATAR -> adult outcomes (right structure, small n)

Since 2003, the LSAY cohorts have been PISA samples, followed for about 10 years; access is
free after registration with the Australian Data Archive. Mapping: `X` = PISA scores and
background at 15, `Q` = the university-entrance rank (ATAR/TER, self-reported), `Y` =
completion, employment, earnings. This is the textbook score-explained structure: the
prior test explains the later score, and `eta` is growth between 15 and 18. However:
- there is no single sharp cutoff, because course- and institution-specific ranks make the
  design fuzzy and multi-cutoff;
- n is small: after attrition, probably only a few thousand respondents per cohort report
  an ATAR (an estimate; the NCVER linked Y15 analysis used 2,310 records). That is far
  below the n the screens needed.

A policy hook exists (minimum-ATAR rules for teaching degrees), but it covers a small
subsample.

### 4. NAEP x state proficiency cut scores (motivation for the performative extension only)

NCES maps each state's proficiency cut score onto the NAEP scale (the latest report uses
2022 NAEP). States moved these thresholds a lot, especially when they adopted new
assessments. NAEP is a low-stakes audit of the same population, so it can show whether
state score distributions respond to where the cutoff sits. That is evidence for Future
Work IV, not a dataset for the current estimator: the threshold is state-level (a few
hundred state-grade-subject-years), and within-state distributions near the cut need
restricted NAEP.

## Considered and rejected

- **School-entry birthdate cutoffs** (relative age; TIMSS/PISA, NAEP Long-Term Trend age
  samples): covariates cannot predict birth date, so `T` barely varies and there is no
  overlap in `eta`. The identification strategy fails.
- **Class-size caps (Maimonides' rule) in TIMSS/PISA**: the rule is a multi-cutoff
  sawtooth, compliance is fuzzy, there are only about 150-300 schools per country, and
  effects are small. Interior optima are unlikely.
- **Achievement levels, benchmarks, and proficiency levels as `D`**: nothing is assigned by them.
- **TIMSS 2023 Longitudinal** (grade 4->5 in 9 systems; grade 8->9 in Jordan, Korea, and
  Sweden): useful as `X` (prior score), but no threshold assignment happens between waves.

## Access

- NAEP student data are restricted-use: an IES license applied for through the Standard
  Application Process (ResearchDataGov). School IDs link to CCD from 2000 onward.
- SEDA, CCD, FRAC/CBPP CEP lists, TIMSS/PIRLS (IEA), and PISA (OECD) are public.
- LSAY: free after registration with the Australian Data Archive.

## Suggested next steps

1. CEP: build the school-level CCD + SEDA + ISP file for one or two states that report
   untruncated direct-certification counts. Check the spread of `T` (R-squared of ISP on
   school characteristics), overlap at 40%, and the first stage (adoption given
   eligibility). Then run `screen_candidate` with the derived cost function.
2. PISA benchmark: fit `Q ~ X` on one large country to measure the realistic signal share
   and the residual heteroskedasticity, then add it as a scenario in the distributional
   battery.

## Sources

- USDA FNS final rule lowering the CEP minimum ISP to 25%: https://www.fns.usda.gov/cn/fr-092623
- 2025 proposal to raise the threshold to 60%: https://www.k12dive.com/news/house-republicans-float-plan-to-cut-community-eligibility-provision/739212/ ; left out of the final bill: https://www.k12dive.com/news/access-to-free-school-meals-under-threat-think-tank-warns/824629/
- FRAC CEP database (ISP by school; eligible and near-eligible): https://frac.org/research/resource-library/community-eligibility-cep-database
- CCD direct-certification counts from 2016-17: https://nces.ed.gov/ccd/quickfacts.asp
- Ruffini, CEP and achievement (JHR): https://edopportunity.org/papers/Ruffini_CEP_achievement_JHR.pdf
- SEDA 5.0 documentation: https://stacks.stanford.edu/file/druid:cs829jn7849/SEDA_documentation_v5.0.pdf
- Title I schoolwide threshold history: https://www.ed.gov/sites/ed/files/rschstat/eval/title-i/schoolwide-program/report.pdf
- NAEP restricted-use access and CCD linkage: https://nces.ed.gov/statprog/instruct.asp ; https://naep-research.airprojects.org/portals/0/edsurvey_a_users_guide/_book/dataAccess.html
- NAEP state mapping (2022): https://nces.ed.gov/nationsreportcard/studies/statemapping/
- TIMSS 2023 Longitudinal: https://www.iea.nl/studies/iea/timss/2023-Longitudinal
- LSAY data access (ADA): https://ada.edu.au/lsay/ ; LSAY and self-reported TER/ATAR: https://www.ncver.edu.au/research-and-statistics/publications/all-publications/the-impact-of-schools-on-young-peoples-transition-to-university
- Maimonides' rule (Angrist & Lavy): https://www.nber.org/papers/w5888
