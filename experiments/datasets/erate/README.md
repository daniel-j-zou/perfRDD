# E-Rate: discount bands on the NSLP share, by urban/rural status

| Field | Value |
|---|---|
| **Q** | NSLP share of the billed entity (school district or independent school), in percent: 100 x NSLP-eligible students / full-time students, from its FCC Form 471 for funding year t |
| **Threshold** | `load()`: 49.5, the edge of the 50-74% band. Effective cutoffs on Q: 0.995, 19.5, 34.5, 49.5, 74.5 (`CUTS`), the same for urban and rural entities |
| **Treatment** | `load()`: `1{Q >= 49.5}`, the 80% discount band (Category 1 urban 60 -> 80, rural 70 -> 80). `load_c20`, `load_c35`, `load_c75`: the other edges |
| **Rule class** | several cutoffs on one score; the discount at each band differs by group (urban/rural); a matrix of price subsidies, not a yes/no |
| **X** | log enrolment, log number of schools, rural, independent school, private / charter / tribal / ESA flags, latitude and longitude of the billed entity (with a missing flag), 55 state/territory and 3 year indicators (69 columns; 4 continuous) |
| **Y** | `load()`: log(1 + pre-discount eligible dollars requested on all the entity's FY t+1 Forms 471, as filed); 0 if it files none. `load_committed()`: log(1 + dollars committed on its FY t+1 funded requests) |
| **n** | 73,434 entity-years (21,137 entities), FY 2022-2025; `load_committed`: 54,915 (FY 2022-2024) |
| **Source** | USAC Open Data (Socrata, opendata.usac.org): Form 471 Basic Information, FRN Status, Discount Calculations, Supplemental Entity Information; 47 CFR 54.505; USAC "Discount Matrix" |

**Fit: weak.**
- **X predicts Q poorly.** R^2 is 0.29, and sd(eta) is 26.4 points on a 0-100 scale.
- **The score is reported by the applicant and is sorted around the cutoffs.**
  - Entity-years in [49.5, 50.5) outnumber those in [48.5, 49.5) by 1.77 to 1. The
    ratio is 1.32 at 34.5, 1.16 at 74.5 and 1.12 at 19.5.
  - At placebo points (24.5, 29.5, ..., 69.5) the same ratio is 0.89-1.11.
  - The excess is largest for independent and private schools (2.45 and 2.87 at 49.5).
    It is still 1.67 for public entities that use no alternative mechanism.
  - Q is therefore not a clean running variable at any edge.
- **The rule and the outcome are clean.** The deployed rule is reproduced exactly (see
  below), and USAC publishes outcomes for the following years.

## The rule

47 CFR 54.505 (eCFR text as of 2026-10-26, saved in `data/raw/docs/`) and USAC's
"Discount Matrix" (revised January 2024) set the following matrix.

| % of students eligible for NSLP | C1 urban | C1 rural | C2 urban | C2 rural |
|---|---:|---:|---:|---:|
| < 1 | 20 | 25 | 20 | 25 |
| 1-19 | 40 | 50 | 40 | 50 |
| 20-34 | 50 | 60 | 50 | 60 |
| 35-49 | 60 | 70 | 60 | 70 |
| 50-74 | 80 | 80 | 80 | 80 |
| 75-100 | 90 | 90 | 85 | 85 |

Category 2 for Tribal libraries is 90 in the top band from FY 2024 (54.505(g)). The
2026 amendment (91 FR 20375) removed only the old voice phase-down, 54.505(d).

- **District-wide share.** A school district divides its total NSLP-eligible students by
  its total students, and that one rate applies to all its schools (54.505(b)(1)).
  Independent schools use their own counts.
  - In the FY 2024 Discount Calculations file (41,933 district application versions),
    the child schools' NSLP counts sum to the Form 471 total in 99.97% of cases. Student
    counts match in 91.2%, and the implied band in 99.0%. The adapter uses the Form 471
    totals, which reproduce the discount.
- **Urban/rural.** A school is urban if it lies in a Census urbanized area or urban
  cluster of 25,000 or more people (54.505(b)(3)).
  - A district is rural only if more than 50% of its schools are rural; non-instructional
    facilities do not count (USAC, "Urban or Rural Status").
  - In FY 2024, this majority rule matches the published district status for 99.7% of
    21,051 district applications.
- **CEP schools.** For schools in the Community Eligibility Provision, the NSLP count is
  ceil(students x min(1, 1.6 x identified-student %)). It matches 98.4% of 165,076 CEP
  school rows (FY 2024).
  - The 1.6 multiplier is USAC/FCC guidance. It is checked here against the data, not
    against the order's text.
- **Alternatives.** Schools outside NSLP may use an alternative mechanism (survey,
  sibling match). 10.5% of sample entity-years do (`extras["alt_discount"]`).
- **Rounding, found in the data.** The documents do not say how a fractional share maps
  to a band. Applied to the exact Q, the published discount behaves as follows:
  - **At the 20/35/50/75 edges.** It follows round-half-up to a whole percent, so the
    cutoffs are 19.5, 34.5, 49.5 and 74.5, and ties go up. Shares in [x.495, x.5) go
    down (59 cases), so there is no rounding to two decimals first.
  - **At the 1% edge.** It follows the share rounded to two decimals, so the cutoff is
    0.995. The largest share given the bottom band is 0.988 and the smallest given the
    next is 0.995, so the cutoff is only bracketed.

## Rule check (2026-10-28)

`adapter.rule_check()` applies the matrix with `CUTS` to Q and the published urban/rural
status of every Form 471 of a school district or independent school. It covers both
versions: Original (as filed) and Current (after USAC review).

| FY | Rows (Original / Current) | C1 agreement | C2 agreement |
|---|---|---:|---:|
| 2022 | 30,792 / 30,564 | 1.000 | 1.000 |
| 2023 | 30,076 / 29,760 | 1.000 | 1.000 |
| 2024 | 30,904 / 30,613 | 1.000 | 1.000 |
| 2025 | 32,658 / 32,124 | 1.000 | 1.000 |
| 2026 | 32,426 / 29,434 | 1.000 | 1.000 |

- All 309,351 rows match. With bands on the unrounded share instead, 98.2% match.
- In the sample, `extras["eligible_published"]` (published C1 discount at or above the
  band above the cutoff) agrees with D for every row, at every edge.

## Sample and outcome

- **Unit.** A billed entity (BEN) x funding year. Only entities whose Form 471 applicant
  type is School District or School are included. Consortia (average discounts) and
  libraries (they borrow a district's share) are excluded.
- **Which application sets Q.** An entity files 1-5 Forms 471 a year. Q comes from its
  Current version of a Category 1 application, else the earliest certified. 1.1% of
  entity-years carry more than one Q across their applications
  (`extras["n_q_values"]`).
- **Outcome years.**
  - Y is measured in FY t+1. It sums the Original (as filed) version of every
    application, both categories.
  - 6.6% of entity-years file nothing in t+1 (Y = 0).
  - Mean Y is 10.0 (exp(10) is about $22,000) and sd(Y) is 3.0. corr(Y, Q) = 0.10.
- **Committed dollars.** `load_committed` sums `funding_commitment_request` over FY t+1
  FRNs whose Current status is Funded. It stops at FY 2024 because FY 2026 commitments
  are still being made.
- **Covariates.** X comes from the same application (enrolment, location) and from the
  Discount Calculations parent row (subtype, number of schools).
  - Left out of X because they are mechanically tied to the score: CEP participation
    (`extras["cep_parent"]`, `share_cep_schools`) and the alternative-mechanism flag.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness erate` reports, for the 49.5 edge:

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 73,434, 69 (4) |
| R^2 of Q on X | 0.289 (urban 0.300, rural 0.364) |
| sd(T), sd(eta) | 16.85, 26.40 |
| sd(eta) max/min across T deciles | 1.20 |
| Treated share | 0.528 (urban 0.507, rural 0.555) |
| Within 0.25 sd(eta) of the cutoff | 14.4% |
| Agreement with the published discount band | 1.000 |

| Loader | Cutoff | Treated share | Near cutoff | Entity-years within 1 point |
|---|---:|---:|---:|---:|
| `load_c20` | 19.5 | 0.815 | 10.7% | 1,141 |
| `load_c35` | 34.5 | 0.687 | 12.9% | 1,403 |
| `load` | 49.5 | 0.528 | 14.4% | 1,629 |
| `load_c75` | 74.5 | 0.297 | 10.5% | 1,208 |

- 11.3% of entity-years have Q = 100; 36% of these have CEP schools.
- 2,404 entity-years lie within 1 point of the 0.995 edge, and it has no loader.
  - 84% of them are independent schools, 77% private and 37% on an alternative
    mechanism.
  - The edge's location is only bracketed.

## Getting the data

    python -m experiments.datasets.erate.download          # idempotent, ~0.91 GB
    python -m experiments.datasets.erate.download --check  # sizes on disk
    python -m experiments.datasets.erate.download --only basic_info frn_status docs

- **Endpoint.** Every table uses Socrata's CSV endpoint
  `https://opendata.usac.org/resource/<id>.csv`. Each query has a `$select` of named
  columns, `$where=funding_year='<FY>'`, and `$limit=100000` / `$offset` paging.
- **Exact queries.** `download.table_url()` prints each one.
- **Left out.** Contact names, e-mails, phone numbers and free-text narratives.
- **Snapshot.** Downloaded 2026-10-28. USAC updates these tables daily (last modified
  2026-10-27), and FY 2025-2026 commitments are still changing.
- **Access.** No login, form or CAPTCHA; the portal answered every scripted request.
- **Cache.** The adapter caches `data/processed/erate_panel.csv.gz`, built in about 25
  s. Delete it after changing `adapter.py`.

| File | Socrata id / URL | Rows | Bytes | Use |
|---|---|---:|---:|---|
| `basic_info_fy{2022..2026}.csv` | `9s6i-myen` (FCC Form 471 Basic Information) | 364,509 | 88,655,762 | Q, urban/rural, published discount, dollars requested |
| `frn_status_fy{2022..2026}.csv` | `qdmp-ygft` (FRN Status) | 553,681 | 125,448,955 | commitments (`load_committed`) |
| `discount_calc_fy{2022..2026}.csv` | `upfy-khtr` (Discount Calculations), `applicant_type in ('School District','School')` | 3,094,543 | 642,683,609 | subtype, schools, CEP, urban/rural checks |
| `entities.csv` | `7i5i-83qf` (Supplemental Entity Information, current snapshot) | 249,885 | 53,388,794 | not used by the adapter |
| `docs/Discount-Matrix.pdf` | https://www.usac.org/wp-content/uploads/e-rate/documents/samples/Discount-Matrix.pdf | | 60,965 | the matrix |
| `docs/ecfr_47cfr54.505_2026-10-26.xml` | https://www.ecfr.gov/api/versioner/v1/full/2026-10-26/title-47.xml?part=54&section=54.505 | | 10,605 | 47 CFR 54.505 |
| `docs/meta/<id>.json` | `https://opendata.usac.org/api/views/<id>.json` | | 559,640 | column descriptions |

Rows by year: Basic Information 71,834 / 70,501 / 72,550 / 76,923 / 72,701; Discount
Calculations 782,091 / 569,775 / 572,065 / 632,499 / 538,113 (FY 2022-2026). Total
910,808,330 bytes.

## Caveats

- **Manipulation.** See the bunching figures under "Fit". The NSLP count, the
  enrolment, the data date and the alternative mechanism are the applicant's choices,
  within USAC rules this README does not document.
  - Review (PIA) changes them: among the 152,495 applications with both versions, Q
    differs in 11.3% and the C1 discount in 3.9%.
  - The adapter's Q is the Current (reviewed) one.
- **The treatment is a price.**
  - The discount is the federal share of eligible cost. Requested pre-discount dollars
    respond to the price, and committed dollars rise mechanically with the discount.
  - Bands are persistent: the FY t+1 discount depends on FY t+1's Q, which is close to
    FY t's.
  - 86% of entity-years are in the same band the next year, and 95% on the same side
    of 49.5. Y measures the effect of being in the higher band in t, partly through
    t+1.
- **Units.** Entities repeat over up to four years, so cluster by `extras["ben"]`.
  Mergers and BEN changes are not tracked.
- **Covariate timing.** Enrolment is the same count as the score's denominator, and
  location and subtype are reported in the same profile. None is the NSLP count itself.
- **Not covered.** Library and consortium discounts, the voice phase-down (ended FY
  2019) and Category 2 budgets. C2 budgets are per student and do not depend on the
  band.
