# Illinois ESSA summative designations: percentile cutoffs on a weighted index

| Field | Value |
|---|---|
| **Q** | `load()`: index - (T5 + 0.00005). The index is the All-Students index score (0-100, published to four decimals); T5 is the year-and-band Comprehensive threshold, rounded to four decimals. `load_exemplary()`: index - T10. `load_two_score()`: high schools, (index - (T5 + 0.00005), composite graduation rate %). `load_two_score_indicators()`: elementary/middle schools with all eight indicators, (mean growth score - c, weighted mean of the other six indicator scores - c), c = T5 + 0.00005 |
| **Threshold** | 0 on Q. On the index, T5 and T10 are set each year and band by rank: bottom 5% and top 10% of the ranked schools. T5 = 35.385 / 40.004 / 38.709 (ES, 2023 / 2024 / 2025) and 11.987 / 9.344 / 15.026 (HS). T10 = 81.449 / 84.036 / 81.994 (ES) and 91.235 / 91.265 / 93.018 (HS). Graduation: a composite rate of 66.67% (indicator score 0) |
| **Treatment** | `load()`, `load_es()`, `load_hs()`, `load_not_in_support()` and both two-score loaders: D = 1 = **not Comprehensive** (`1{Q >= 0}`; two-score loaders: the rule below). The untreated side, Comprehensive, means IL-EMPOWER support and state funding for a four-year improvement cycle. `load_exemplary()`: `1{Q >= 0}` = Exemplary |
| **Rule class** | percentile cutoffs on a fixed-weight sum of eight indicator scores. The weights depend on the grade band and are renormalised when an indicator is missing. For high schools, Comprehensive is an either rule: index <= T5 or graduation rate <= 66.67% |
| **X** | log enrollment; % low income, EL, IEP, homeless, Black, Hispanic, Asian and other race; student mobility rate; Title I program; school type, lowest and highest grade, charter, Chicago Public Schools; district type, log enrollment, % low income, % EL and log number of schools; ROE region (first two RCDTS digits), grade band and year indicators. 69 columns in `load()`, 14 of them continuous |
| **Y** | the next year's All-Students index score for the same school and grade band (2023 -> 2024, 2024 -> 2025) |
| **n** | `load()`: 7,116 school-band-years in 2023-2024 (5,764 ES, 1,352 HS; 3,670 schools), 357 Comprehensive. `load_exemplary()`: 6,076 (742 Exemplary). `load_two_score()`: 1,315 HS (74 Comprehensive, 8 of them by graduation rate alone). `load_two_score_indicators()`: 2,383 ES (74) |
| **Source** | ISBE: Summative Designation Business Rules 2023-2025; Descriptive Statistics 2023-2025; the official summative designation lists 2023-2025; index and indicator scores and the thresholds from ISBE's public Report Card Service (rcc.isbe.net, the data source of illinoisreportcard.com); Report Card Public Data Sets 2022-2025 |

ES is the elementary/middle band (grades 1-8) and HS the high-school band (grades 9-12).
A school serving both bands gets an index and a designation in each.

## The rule

The Summative Designation Business Rules (2023, 2024 and 2025 editions, `data/raw/rules/`)
build the designation in five steps. The indicator formulas and weights are the same in
all three editions.

1. **Indicator scores (Process 3).** Each indicator is scored 0-100 for the All-Students
   group and for every student group with at least 20 students:
   - ELA and math proficiency: % proficient / interim target, by grade group;
   - growth: 20/9 x mean SGP - 62.22 (0 at SGP 28, 100 at 73);
   - graduation: 3.7975 x composite rate - 253.16 (0 at a composite rate <= 66.67%, 100
     at >= 93%). The composite rate is 0.6 x 4-year + 0.3 x 5-year + 0.1 x 6-year;
   - chronic absenteeism: 100 - 2 x rate;
   - climate-survey participation: 20/9 x rate - 111.11;
   - 9th grade on track: 3 x (rate - 66.66);
   - ELPtP: mean of the students' progress points.
2. **Index (Process 4).** The index is the weighted sum of the indicator scores, divided
   by 100:

   | Indicator | ES weight | HS weight |
   |---|---:|---:|
   | ELA proficiency | 7.5 | 7.5 |
   | Math proficiency | 7.5 | 7.5 |
   | ELA growth | 25 | - |
   | Math growth | 25 | - |
   | Graduation rate | - | 50 |
   | Science proficiency | 5 | 5 |
   | English learner progress (ELPtP) | 5 | 5 |
   | *Academic total* | *75* | *75* |
   | Chronic absenteeism | 20 | 10 |
   | Climate survey participation | 5 | 6.67 |
   | 9th grade on track | - | 8.33 |
   | *SQSS total* | *25* | *25* |

   - When an indicator is missing (fewer than 20 students), its weight goes to the
     indicators of its own category (academic or SQSS) present at the school, in
     proportion to their weights.
   - A school needs 5 of the 8 indicators, at least one of them SQSS, to get an index.
3. **Thresholds (Process 5).** ISBE ranks the All-Students index scores of each band in
   descending order (a school with two bands is in both lists). T10 is the score at the
   top-10% rank and T5 the score at the bottom-5% rank. The 2025 rules give the ranks as
   round(0.1 n) and round(0.95 n) + 1. The 2023 and 2024 rules say "top 10%" and "lowest
   5%".
4. **Designation (Process 5), in this order:**
   - **Comprehensive**: index <= T5, reason "All Students". For high schools also a
     graduation indicator score of 0 (composite rate <= 66.67%), reason "Low Graduation
     Rate";
   - **Targeted**: not Comprehensive, and some student group's index <= T5;
   - **Exemplary**: none of the above, and index >= T10;
   - **Commendable**: all other schools.
5. **Funding-table steps.**
   - A Comprehensive school in year 3 of its IL-EMPOWER grant is published as
     **Intensive**.
   - A Targeted school in year 3 whose original group is still at or below T5 is
     published as **Comprehensive, "From Targeted"**.
   - ISBE's funding tables are not public, so these two steps are checked through the
     published reasons only.

**Consequence.**
- Comprehensive schools get IL-EMPOWER support and funding through a four-year
  improvement cycle.
- A school stays in support for the full four years, even if a later designation is
  Commendable or Exemplary.
- Targeted schools get "targeted" support. Exemplary is recognition only.

**Title I.** The definitions say "lowest-performing 5 percent of Title I-eligible schools".
In practice the Process 5 ranking takes every school with an index, and every school at or
below T5 is designated Comprehensive whatever its Title I status, 2023-2025. The rule check
below confirms this.

**Years.**

| Year | Designations | School-level index public? |
|---|---|---|
| 2018, 2019 | yes | no (lists only) |
| 2020, 2021 | none (COVID-19 waivers; 2019 designations reissued in 2020) | - |
| 2022 | yes, amended system | no |
| 2023-2025 | yes | yes (Report Card Service) |
| 2026 | "Profiles of Performance" (ESSA amendment approved 2026-09-11): performance levels per indicator, no index | - |

So 2023-2025 are the years in which the index rule can be reproduced. Only 2023 and 2024
have a next-year index as the outcome.

### Thresholds

ISBE publishes the thresholds in two places:
- the Descriptive Statistics PDFs, rounded to 0.01;
- the state `SummativeSchoolDistribution` report of the Report Card Service, at full
  precision (`adapter.state_thresholds()`).

The adapter uses the full-precision values, rounded to the four decimals of the
published index:

| Year, band | T5 (Comprehensive) | PDF | T10 (Exemplary) | PDF |
|---|---:|---:|---:|---:|
| 2023 ES | 35.385009 | 35.39 | 81.448531 | 81.45 |
| 2023 HS | 11.986575 | 11.99 | 91.235064 | 91.24 |
| 2024 ES | 40.003878 | 40.00 | 84.036129 | 84.04 |
| 2024 HS | 9.343947 | 9.34 | 91.264913 | 91.26 |
| 2025 ES | 38.709383 | 38.71 | 81.993536 | 81.99 |
| 2025 HS | 15.026447 | 15.03 | 93.017712 | 93.02 |

**The ranks cannot be fully recomputed from public data.** The service returns only the
high-school band for a school that has two bands, so the ES index of 74 / 84 / 84
dually designated schools (2023 / 2024 / 2025) is not public.
- Elementary/middle band:
  - Count the schools on ISBE's ES list (n = 3,060 / 3,056 / 3,049).
  - The top-10% rank round(0.1 n) then implies 3 / 3 / 1 dual schools above T10. That
    equals the number of dual schools published Exemplary.
  - The bottom-5% rank implies 5 / 6 / 10 dual schools at or below T5. The lists show
    5 / 3 / 11 dual schools Comprehensive by index.
- High-school band (all schools public):
  - The rank formula reproduces both 2023 thresholds and the 2024 T5 exactly.
  - The 2024 T10, and the 2025 T10 and T5, are each one rank away.
- `adapter.thresholds()` reports these ranks.

## Getting the data

    python -m experiments.datasets.il_school_designations.download          # everything (~100 min)
    python -m experiments.datasets.il_school_designations.download --static # the 14 static files
    python -m experiments.datasets.il_school_designations.download --check  # sizes and hashes

Everything goes to `data/raw/` (gitignored); all sources are public, with no login or form.

**Static files** (www.isbe.net, fetched 2026-10-28, 144,850,214 bytes in all; SHA-256 in
`download.py`).
- The Report Card data-library page
  (https://www.isbe.net/Pages/Illinois-State-Report-Card-Data.aspx) links them as
  `/_layouts/Download.aspx?SourceUrl=/Documents/...`.
- www.isbe.net/robots.txt disallows `/_layouts/`. So `download.py` fetches the same files
  from their direct `/Documents/` address, which serves identical bytes.

| File (https://www.isbe.net/Documents/...) | Bytes | Use |
|---|---:|---|
| `2022-Report-Card-Public-Data-Set.xlsx` | 21,161,046 | 2022 designation (previous support status of 2023 units) |
| `23-RC-Pub-Data-Set.xlsx` | 23,586,246 | X for 2023 ("General" sheet) |
| `24-RC-Pub-Data-Set.xlsx` | 55,577,890 | X for 2024 |
| `2025-Report-Card-Public-Data-Set.xlsx` | 40,314,938 | X for 2025 (rule check only) |
| `Summative-Designations.xlsx` | 214,864 | official 2023 designation list |
| `2024-Off-Summ-Desig.xlsx` | 203,350 | official 2024 list |
| `Official-Summ-Designation-List-ES-HS-Dually-Desig-Schools.xlsx` | 203,053 | official 2025 list |
| `Summ-Desig-Bus-Rules-2023.pdf` | 534,465 | rule 2023 |
| `Summative-Designation-Business-Rules-2024.pdf` | 585,631 | rule 2024 |
| `2025-Official-Summative-Designation-Business-Rules.pdf` | 608,754 | rule 2025 (ranks, weights, formulas) |
| `2026-Official-Summative-Designation-Business-Rules.pdf` | 813,271 | the 2026 redesign |
| `2023-Descriptive-Statistics.pdf` | 375,815 | thresholds to 0.01 |
| `2024-Summ-Desig-Descriptive-Statistics.pdf` | 219,980 | thresholds to 0.01 |
| `2025-Descriptive-Statistics.pdf` | 450,911 | thresholds to 0.01 |

Each list has three sheets (Dually Designated, Elementary/Middle, High Schools). A row
gives RCDTS, district, school, grade band, designation and reason.

**Report Card Service** (rcc.isbe.net). The index and indicator scores are not in the
Report Card Public Data Set. The Illinois Report Card shows them under School >
Accountability > Summative Designation, and loads them from ISBE's public service.

- **Thresholds**:
  `https://rcc.isbe.net/v1//ReportCardService/(En)/Domain(State)/Id(IL)/(SummativeSchoolDistribution)/(<year>)/Table/(Xml)`,
  saved as `data/raw/rcc_state/SummativeSchoolDistribution_<year>.xml`:

  | Year | Bytes | SHA-256 |
  |---|---:|---|
  | 2023 | 1,330,675 | c9d6f6243dfd03ba49cc2bb1de2cef95e39d75857adadfeb571381102d197126 |
  | 2024 | 1,330,722 | af7abe5e5fe3488c705f6293922f75c4b188e82a09840a173c578dd9fabcfa91 |
  | 2025 | 1,330,584 | cf9ed99848bae76424872163dea4f5dd0cc2838e89f2a34e3ae53dd451f4bcea |

- **Scores, one school at a time**:
  `https://rcc.isbe.net/v1//ReportCardService/(En)/Domain(School)/Id(<RCDTS>)/(SummativeWeightedIndicators)/(2023,2024,2025)/Table/(Xml)`.
  - The answer points to a generated file `https://rcc.isbe.net/v1//tempData/XML/File<n>.xml`.
  - The file has, per year, band, student group and indicator, the indicator score,
    effective weight and weighted score.
  - It also has one "Overall" row per year and band: the All-Students index to four
    decimals and the designation.
  - All 3,698 schools on the three lists were queried on 2026-10-28 (16:57-18:51 UTC),
    two at a time, with a 0.25 s pause between one worker's requests.
  - 3,698 files, 163,328,348 bytes, saved as `data/raw/rcc/<RCDTS>.xml`.
  - `data/raw/rcc_manifest.csv` records URL, size, time and SHA-256 for each (manifest
    SHA-256 99c51c18bfd1fb7fa41a0ac3964be707e9ef7fea5e7d34832c5e9bb734c4bc06).

Quirks of the service:
- **Cached files.** It names the generated file after the request and serves an
  existing file of that name. Some of these are in a bare `<Record>` layout without the
  "Overall" rows. `download.py` then repeats the request with the years in another
  order. 269 schools needed this, and every saved file has the full layout.
- **Years.** It answers "No Data Available" for 2018, 2019 and 2022.
- **Two bands.** It returns only the HS band for a school with two bands.
- **Transient failures.** Three schools came back without a status code on the first
  pass. `download.py` now retries such responses, and all three were fetched.

**Reading.**
- The adapter reads only the "General" sheet of each report card, with a streaming
  column reader (`_xlsx.py`): the sheets have 900-1,260 columns and up to 75 MB of
  XML. The lists use `fl_school_grades/_spreadsheet.py`.
- The panel is cached in `data/processed/il_school_designations_panel.csv.gz`, written
  atomically. Delete it after changing `adapter.py`.

## Rule check (2026-10-28)

`adapter.rule_check()` recomputes every designation from the published index and
group-index scores with ISBE's thresholds, and compares it with the official list:

| Year, band | On list | With index | Comprehensive by index | by graduation | Targeted | Exemplary | Exact agreement | With funding-table steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 ES | 3,060 | 2,986 | 148 | - | 418 | 303 | 0.939 | 1.000 |
| 2023 HS | 664 | 664 | 33 | 4 | 33 | 66 | 0.956 | 1.000 |
| 2024 ES | 3,056 | 2,973 | 147 | - | 378 | 303 | 0.978 | 1.000 |
| 2024 HS | 702 | 702 | 35 | 4 | 15 | 71 | 0.957 | 1.000 |
| 2025 ES | 3,049 | 2,965 | 142 | - | 394 | 304 | 0.988 | 1.000 |
| 2025 HS | 702 | 702 | 36 | 0 | 29 | 71 | 0.962 | 1.000 |

- **Every published designation is explained, 2023-2025.** The only differences from the
  rule are the two funding-table steps:
  - **Intensive** for a rule-Comprehensive school: 50 / 27 (ES / HS) in 2023, 39 / 30 in
    2024, 37 / 27 in 2025;
  - **Comprehensive "From Targeted"** for a rule-Targeted school: 132 / 2 in 2023, 25 / 0
    in 2024, none in 2025.
  - The designation in the service's "Overall" row equals the list's in every case but
    one: Unity Point Elem School, 2023, shows Commendable in the service but Targeted on
    the list. The rule gives Targeted (its students-with-disabilities group is at or
    below T5).
- **The index is reproduced from the indicator scores.** Using the Process 4 weights, with
  missing weights redistributed, the reconstruction is within 0.0043 of the published
  index for every unit. The gap is the rounding of the indicator scores to 0.01.
- **The 2-decimal PDF thresholds would not do.** With them, 7 school-years would be
  misclassified: for example, a school with an index of 81.4485 is Exemplary at
  T10 = 81.448531.
- **Two group-index ties.** Group index scores are published to two decimals. In two
  school-years a group's index rounds to the T5 PDF value. Compared with the
  full-precision T5, both come out as published.
- **Title I does not enter.** All 541 school-band-years at or below T5 in 2023-2025 are
  published Comprehensive (336) or Intensive (205). That includes 19 whose school is not
  served by Title I.

## Samples

`load()` uses the Comprehensive boundary, with ES and HS pooled. Q is centred by the
cutoff of each year and band (`extras["group"]`, `extras["group_cutoffs"]`).

| Step (2023 and 2024) | ES | HS |
|---|---:|---:|
| school-band-years on the official lists | 6,116 | 1,366 |
| with an index from the service (drops the ES band of 156 dually designated schools, and 3 others) | 5,952 | 1,365 |
| with a report-card row (X) | 5,952 | 1,365 |
| Y observed (next year's index; drops closures and schools without a designation) | 5,915 | 1,362 |
| not Comprehensive by graduation rate alone (HS) | 5,915 | 1,354 |
| not Comprehensive "From Targeted" (escalated Targeted schools) | **5,764** | **1,352** |

- The last two steps take out units that are Comprehensive through a rule other than
  the index: 8 by the graduation rule and 159 "From Targeted". With them out, D in
  `load()` is the index rule exactly. `extras["excluded"]` gives the counts.
- D agrees with the published Comprehensive/Intensive status for all 7,116 units
  (`extras["eligible_published"]`).

| Loader | n | D = 0 | Notes |
|---|---:|---:|---|
| `load()` | 7,116 | 357 Comprehensive (5.0%) | ES and HS |
| `load_es()` | 5,764 | 290 | |
| `load_hs()` | 1,352 | 67 | |
| `load_not_in_support()` | 6,623 | 145 | drops 493 units Comprehensive or Intensive the year before (already in a four-year cycle), so D = 0 is a new identification |
| `load_exemplary()` | 6,076 | 5,334 (D = 1: 742 Exemplary) | units the rule designates Exemplary or Commendable (no group at or below T5). Q = index - T10 |
| `load_two_score()` | 1,315 | 74 | HS with a graduation rate. Q = (index - (T5 + 0.00005), rate). D = 1{Q1 >= 0 and Q2 > 66.6661}. `extras["cutoffs"] = (0, 66.6661)`, `extras["rule"] = "both"` for D (Comprehensive is the either rule). 66 are below on the index, 8 on graduation alone |
| `load_two_score_indicators()` | 2,383 | 74 | ES with all eight indicators (nominal weights). Q = (G - c, R - c), G = mean of the ELA and math growth scores, R = weighted mean of the other six. `extras["weights"] = (0.5, 0.5)`, `extras["cutoff"] = 0` |
| `load_two_score_indicators_no_elptp()` | 2,350 | 150 | ES with every indicator but ELPtP. Weights (0.5357, 0.4643) |

**Graduation rate** (`load_two_score`, Q2):
- Where the graduation indicator score is between 0 and 100, Q2 is the rate implied by
  the score, (score + 253.16456) / 3.7975. That is the accountable-school rate, to
  +-0.002.
- Where the score is censored, Q2 is the report card's composite of the 4-, 5- and
  6-year rates, clipped to the censored side. That happened 427 times: 358 scores of 100
  and 69 of 0. No clip was needed: every report-card rate was already on the side the
  score implies.
- Where both exist, the two rates have a median gap of 0.008 points, and 96% are within
  0.05. The largest gap is 9.8 points: the report card counts the responsible school,
  the indicator the accountable school.

**Indicator loaders.** The two-score indicator loaders are exact:
- 0.5 G + 0.5 R equals the published index up to the rounding of the indicator scores
  (< 0.005);
- D agrees with the published designation for every unit;
- they share `load()`'s exclusions.
- The two ES missingness patterns cover 4,733 of the 5,764 ES units. The other patterns
  (missing science or climate) have other weights.

## X and fit

X comes from the "General" sheet of the same year's report card. Variables:
- enrollment (fall); % low income, EL, IEP, homeless; race shares; the student mobility
  rate;
- Title I program;
- school type, normalised across the years' labels; lowest and highest grade served;
  charter (RCDTS school code ending in C); Chicago Public Schools;
- the district's type, enrollment, % low income, % EL and number of schools;
- ROE region (38 indicators; Chicago, 15, is the base); band and year indicators.

Notes:
- The report card suppresses a group's share below 10 students. Such shares are set to
  4.5 / enrollment.
- None of these is test-based, and no indicator score is used. Attendance and graduation
  rates are left out because they are indicator inputs. Mobility and enrollment are for
  the same school year as the tests.

R^2 of Q on X (OLS with intercept):

| Loader / score | R^2 | sd(T) | sd(eta) | sd(eta) max/min over T deciles |
|---|---:|---:|---:|---:|
| `load()` index | 0.712 | 16.8 | 10.7 | 1.30 |
| `load_es()` | 0.558 | 11.1 | 9.9 | 1.57 |
| `load_hs()` | 0.774 | 20.5 | 11.1 | 2.72 |
| `load_exemplary()` | 0.484 | 9.5 | 9.8 | 1.68 |
| `load_two_score()` index / graduation rate | 0.794 / 0.752 | 20.8 / 9.1 | 10.6 / 5.2 | 2.66 / 4.74 |
| `load_two_score_indicators()` growth / rest | 0.263 / 0.778 | 7.3 / 15.6 | 12.2 / 8.3 | 1.18 / 2.81 |
| `load_two_score_indicators_no_elptp()` growth / rest | 0.205 / 0.825 | 7.2 / 20.7 | 14.1 / 9.6 | 1.16 / 3.45 |

- **By year and band**, the `load()` R^2 is 0.55-0.57 for ES and 0.77-0.79 for HS.
  - Without the region indicators: 0.697 overall.
  - With 859 district indicators: 0.799.
  - Adding last year's index (`extras["q_lag"]`, 2024 only): 0.857. It is not in X,
    because it depends on last year's designation.
- **Two-score views.**
  - High schools: corr(T) = 0.93, corr(eta) = 0.81. The weakest direction of T,
    (0.38, -0.92), has sd 3.1, 0.38 of sd(eta).
  - Elementary indicators: corr(T) = 0.68, corr(eta) = 0.41. The weakest direction has
    sd 0.49 of sd(eta). Growth is hard to predict from X (R^2 0.26), the rest easy
    (0.78).

## Outcome

`Y` is the next year's All-Students index score for the same school and grade band, as
published (four decimals):
- 2023 -> 2024 and 2024 -> 2025; `extras["y_year"]` records the year;
- mean 66.7, sd 16.4, corr(Y, Q) = 0.76 (0.82 ES, 0.94 HS);
- 40 eligible units (0.5%) have no Y (closure, or no designation the next year);
- `extras["y_designation"]` gives the next designation.

## Caveats

- **Support lasts four years, so many units near the cutoff are already treated.** A
  school identified Comprehensive stays in support for four years whatever its later
  designations.
  - Of the 146 units within 2 index points of T5, 61 were Comprehensive or Intensive the
    year before (`extras["designation_prev"]`; for 2023 it is the 2022 designation from
    the 2022 report card).
  - Of the 357 Comprehensive units in `load()`, 212 were Comprehensive or Intensive the
    year before.
  - So at the cutoff, D measures a new designation, not the presence of support.
  - `load_not_in_support()` keeps only units not in a Comprehensive cycle the year
    before. It is cleaner but thin: 145 new identifications, 11 of them HS.
- **The boundary is in the tail.**
  - Comprehensive is about 5% of each band. Within 2 index points of T5 there are 126 ES
    and 20 HS units (2.1% of `load()`).
  - The HS bottom 5% is almost all in Chicago: Youth Connection Charter School
    (dropout-recovery) campuses and CPS neighbourhood high schools. Most of them also
    have a composite graduation rate below 67%, so the two HS rules usually bind
    together, and only 8 units are Comprehensive by graduation alone.
  - The HS density is thin just above T5: in 2024, 5 high schools have an index between
    10 and 15, against 36 below 10.
  - The Exemplary boundary has more mass: 530 units within 2 points.
- **The rule changes between years.**
  - The weights and formulas are the same in 2023-2025, but the thresholds move (ES T5:
    35.4, 40.0, 38.7).
  - 2025 moved to new IAR performance levels: proficient is level 3-4, where it was
    4-5 through 2024. So the 2025 proficiency scores, and hence the 2024 -> 2025 outcome,
    are on a new basis.
  - The year indicators absorb level shifts only.
- **Timing.**
  - The year-t designation uses the spring-t tests and is published with the fall-t
    report card, in October.
  - Y, the year t + 1 index, uses spring t + 1 tests, taken during the first school year
    of support.
  - Comprehensive schools are funded through IL-EMPOWER for four years. Targeted schools
    get smaller targeted grants.
- **Two steps use non-public data.** Intensive and "From Targeted" depend on ISBE's grant
  tables. Both are identified from the published reasons, and "From Targeted" units are
  excluded from the main loaders.
- **Units.**
  - School-band-years from 3,670 schools. A school with two bands could appear twice a
    year, once per band, but only its HS band has a public index.
  - Cluster by school or district: district indicators raise the R^2 from 0.71 to 0.80.
  - eta is heteroskedastic for HS (sd ratio 2.7).
- **Missing data.**
  - The ES band of dually designated schools (74-84 a year) has no public index and is
    not in any sample.
  - ISBE ranks them, so they shift T5 and T10, which are taken from ISBE.
- **Shared assumptions.** As in every real dataset screened so far, X independent of
  eta holds only approximately.

## Suggested policy boxes

Cutoff shifts are in index points relative to the deployed cutoff (Q = 0).

| Loader | Parameter | Box | Deployed | Untreated or treated share over the box |
|---|---|---|---|---|
| `load_es()` | cutoff shift | [-10, +10] | 0 | Comprehensive 1.2% -> 13.4% |
| `load_hs()` | cutoff shift | [-10, +15] | 0 | 0.4% -> 7.8% (HS density thin above T5) |
| `load()` | cutoff shift, per band | [-10, +10] | 0 | about 1% -> 12% |
| `load_exemplary()` | cutoff shift | [-5, +5] | 0 | Exemplary 25% -> 4% |
| `load_two_score()` | index cutoff shift; graduation cutoff | [-10, +10]; [60, 75] % | 0; 66.67 | 5.3% of HS at or below 66.67; 9.5% below 75 |
| `load_two_score_indicators()` | weight on growth w (rest 1 - w); cutoff shift | [0.3, 0.7]; [-10, +10] | 0.5; 0 | untreated 4.4% (w = 0.3) -> 2.6% (w = 0.7) at the deployed cutoff |
| `load_two_score_indicators_no_elptp()` | weight on growth | [0.35, 0.75] | 0.536 | |
