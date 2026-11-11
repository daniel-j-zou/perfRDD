# Medicaid eligibility at 138 percent of the poverty line — CPS ASEC 2016-2020

| Field | Value |
|---|---|
| **Q** | family income as a percent of the family's poverty threshold, `100 * FTOTVAL / FPOVCUT` (the ratio behind the ASEC recodes `PERLIS` and `POVLL`; persons in related subfamilies carry the primary family's values, as the Census recodes do) |
| **Threshold** | 138: in ACA expansion states non-elderly adults with income up to 138 percent of the poverty line are income-eligible for Medicaid; above it they get marketplace subsidies |
| **Treatment** | `1{Q <= 138}` (treated = income-eligible; the rule treats below, so the estimator scripts mirror the score, see `mirrored()`) |
| **X** | age (and `(age-40)^2`), female, Hispanic, Black, Asian, other race (White base), less than high school, some college, bachelor's or more (high-school base), married, own children under 18 in the family (`FOWNU18`), family size (`FPERSONS`), employed, unemployed (not in labor force base), state indicators (California base) and income-year indicators (2015 base); p = 52 |
| **Y** | `load()`: covered by Medicaid at any time in the income year, binary. `load_uninsured()`: uninsured for the whole income year |
| **n** | 282,414 civilian adults 19-64 in full-year expansion state-years, poverty universe, not SSI recipients, not on Medicare, no work-preventing disability; 37,785 treated (13.4 percent, weighted 12.9) |
| **Y rates** | Medicaid 14.8 percent overall (weighted 14.3); 45.0 percent below the line, 10.2 above. Uninsured 9.7 percent (21.6 below, 7.9 above) |
| **Source** | Census Bureau CPS ASEC public-use files, survey years 2016-2020 = income years 2015-2019, no registration |
| **Extras** | `weight` (`MARSUPWT`, 2 implied decimals removed), `state`, `income_year`, `medicaid`, `uninsured`, `any_public`, `private`, `noncitizen`, `any_difficulty` (`PRDISFLG`), `fair_poor_health`, `hh_seq` |

## Getting the data

```
python -m experiments.datasets.cps_medicaid.download          # ~0.45 GB into data/raw/<year>/
python -m experiments.datasets.cps_medicaid.download --check
```

Sources, all under `https://www2.census.gov/programs-surveys/cps/datasets/<year>/march/`
(fetched 2026-11-11):

| Survey year | Data | Dictionary |
|---|---|---|
| 2016 | `asec2016_pubuse_v3.dat.gz` (29 MB) | `Asec2016_Data_Dict_Full.txt`, `asec2016_pubuse.dd.txt` |
| 2017 | `asec2017_pubuse.dat.gz` (30 MB) | `08ASEC2017_Data_Dict_Full.txt` |
| 2018 | `asec2018_pubuse.dat.gz` (29 MB) | `08ASEC2018_Data_Dict_Full.txt` |
| 2019 | `asecpub19csv.zip` (184 MB: `hhpub19`, `ffpub19`, `pppub19`, replicate weights) | `06_ASEC_2019-Data_Dictionary_Full.pdf`, `persfmt.txt`, `famlfmt.txt`, `hhldfmt.txt` |
| 2020 | `asecpub20csv.zip` (162 MB) | `ASEC2020ddl_pub_full.pdf`, `cpsmar20.pdf` (technical documentation), the three `*fmt.txt` |

The Census server ships CSV only from 2019; 2016-2018 are hierarchical fixed-width files
(household, family and person records, record type in column 1, 1076 characters) whose
layout is identical in the three dictionaries (`LAYOUT_*` in `adapter.py`). The IPUMS
route was not used. `build()` parses all five years in about 30 s and caches
`data/processed/cps_asec_2016_2020.parquet` (889,545 persons, 46 fields).

`www2.census.gov/robots.txt` (fetched 2026-11-11) is, in full:

```
User-agent: RavenCrawler        Disallow: /
User-agent: MegaIndex           Disallow: /
User-agent: MegaIndex.ru/2.0    Disallow: /
User-agent: BLP_bbot/0.1        Disallow: /
User-agent: BLP_bbot            Disallow: /
User-agent: TweetmemeBot        Disallow: /
User-agent: Applebot            Crawl-delay: 30
User-agent: Googlebot           Crawl-delay: 30
User-agent: Yahoo! Slurp        Crawl-delay: 30
User-agent: bingbot             Crawl-delay: 30
```

No rule applies to other clients, so the direct downloads are permitted; the script
pauses 3 s between files anyway. KFF's `robots.txt` is `User-agent: * / Disallow:` (open).

## Expansion states (KFF, "Status of State Action on the Medicaid Expansion Decision", as of November 2026)

A state-year is in the sample when the expansion was in force by January 31 of the
income year; states expanding later in the year are "partial" and dropped for that year
(the grace period decides only Maine, implemented 1/10/2019).

| Income year | Expansion states in the sample | Dropped as partial |
|---|---|---|
| 2015 (28) | AR AZ CA CO CT DC DE HI IA IL KY MA MD MI MN ND NH NJ NM NV NY OH OR PA RI VT WA WV | AK (9/2015), IN (2/2015) |
| 2016 (31) | + AK IN MT | LA (7/2016) |
| 2017, 2018 (32) | + LA | |
| 2019 (34) | + ME VA | |

Implementation dates used (`EXPANSION_DATES`): 1/1/2014 for AR AZ CA CO CT DC DE HI IA
IL KY MA MD MN ND NJ NM NV NY OH OR RI VT WA WV; MI 4/1/2014; NH 8/15/2014; PA 1/1/2015;
IN 2/1/2015; AK 9/1/2015; MT 1/1/2016; LA 7/1/2016; VA 1/1/2019; ME 1/10/2019. Later
expansions (ID, UT 1/2020; NE 10/2020; OK 7/2021; MO 10/2021; SD 7/2023; NC 12/2023)
fall after the last income year, so those states are never in the sample. Wisconsin
covers adults to 100 percent without the expansion and is dropped with the
non-expansion states.

## Variable construction

- **Medicaid.** 2019-2020: `CAID` ("Medicaid coverage last year"; `MCAID` in these
  files bundles CHIP and other means-tested coverage and is only kept raw). 2016-2018:
  `CAID == 1 | MCAID == 1 | CH_MC == 1` (the recode, the item and the child recode;
  the item flags about 2,000 persons a year that the recode misses).
- **Any coverage.** 2019-2020: `COV`. 2016-2018 have no such recode; it is built from
  `COV_GH`, `COV_HI`, `OUT`, `CH_HI` (private), Medicaid as above, `CARE`/`MCARE`,
  `CHAMP`, `PCHIP`, `OTYP_1/2/3/5`, `OTH` unless Indian Health Service only
  (`IHSFLG`), and `AHIPER`. Check against the published Census rates (all ages,
  weighted, percent):

  | Income year | 2015 | 2016 | 2017 | 2018 | 2019 |
  |---|---|---|---|---|---|
  | uninsured, here | 9.08 | 8.76 | 8.83 | 8.47 | 8.03 |
  | uninsured, Census report | 9.1 | 8.8 | 8.8 | 8.5 | 8.0 |
  | Medicaid, here | 19.56 | 19.45 | 19.34 | 16.72 | 16.20 |
  | Medicaid, Census report | 19.6 | 19.4 | 19.3 | 17.9* | 17.2* |

  \* the 2018-2019 report figures include CHIP and other means-tested coverage
  (`MCAID`), the file's `CAID` does not.
- **Disability exclusion.** SSI receipt (`SSI_YN`), Medicare coverage (under 65 this
  means a disability or ESRD determination) and a self-reported health problem that
  prevents work (`DIS_HP`). Keeping them gives n = 314,683 (`drop_disabled=False`).
  `PRDISFLG` (any of the six difficulty items; 3.4 percent of the sample) is not a
  restriction and is in `extras["any_difficulty"]`.
- **Pregnancy** is not asked in the ASEC; pregnant women have a higher Medicaid income
  limit in every state and cannot be removed. WIC receipt (`WICYN`) is kept in the
  cached frame as a weak proxy but not used.
- **Citizenship.** 11.3 percent of the sample are non-citizens, many of whom are not
  Medicaid-eligible at any income (five-year bar, undocumented). `load(citizens_only=True)`
  gives n = 250,628; `extras["noncitizen"]` carries the indicator.
- **Weights** are `MARSUPWT / 100`; the samples are unweighted, weights in `extras`.

## Fit of the score on the covariates

OLS of Q on X (p = 52, unweighted): R-squared **0.164** on the full sample, 0.077 within
38-238. Adding weeks and hours worked and the non-citizen flag gives 0.177; adding own
earnings 0.207; log Q on X gives 0.275 (0.336 with weeks, hours, non-citizen). The
0.3-0.4 quoted in `../BINARY_OUTCOME_DATASETS_20261111.md` needs the work variables and
the log scale.

## Manipulation check (adapter sample, income years 2015-2019 pooled)

| Q bin | n | weighted | Medicaid | uninsured |
|---|---|---|---|---|
| [108, 118) | 3,297 | 5.76 M | 0.448 | 0.201 |
| [118, 128) | 3,843 | 6.71 M | 0.417 | 0.214 |
| **[128, 138)** | **3,558** | 6.17 M | 0.404 | 0.189 |
| **[138, 148)** | **3,514** | 6.31 M | 0.350 | 0.183 |
| [148, 158) | 4,315 | 7.53 M | 0.318 | 0.181 |
| [158, 168) | 4,086 | 7.16 M | 0.304 | 0.199 |

No bunching below the line: 3,558 persons in 128-137 against 3,514 in 138-147 (the
density rises slowly with Q here). Local linear jump at 138 (triangular kernel, below
minus above): Medicaid +3.1 pp at h = 25, +3.4 pp at h = 50; private coverage -4.1 /
-5.0 pp; uninsured -0.1 / +1.1 pp. Barcellos, Jacobson and Levy (AJHE 2023, ASEC
2014-2016, ages 27-64, h = 50) report Medicaid -5.5 pp above the line on a base of 30.4
percent, private non-group +3.6, uninsurance unchanged, and no heaping; the signs and
the absence of a coverage effect are reproduced here, the Medicaid jump is smaller.

Known measurement issues with the score:

- The ASEC income is annual, reported in March for the previous calendar year, with
  Census family and official poverty thresholds; Medicaid uses current monthly modified
  adjusted gross income of the tax household against the HHS poverty guidelines. The
  two ratios differ by a few percent for the same person, so the 138 line in Q is the
  statistical, not the administrative, cutoff and the design is fuzzy on both sides.
- Rounded reports: among persons with Q in 100-180, 34 percent report a family income
  that is a multiple of $1,000, so Q heaps at the values round incomes map to for each
  threshold (the most common Q values near the line each hold 100-200 persons).
- Monthly manipulation that annual data hide: the SIPP shows childless adults in
  expansion states lowering their lowest monthly earnings to qualify (arXiv 2506.12976,
  2025), which the ASEC annual income cannot show; Barcellos et al. 2023 find no
  heaping in the ASEC.
- Imputed income: about a third of ASEC income items are imputed (hot deck); allocation
  flags (`I_*`) are in the raw files but not kept in the cache.

## Design notes

- The rule treats below the threshold. `mirrored()` returns `Q = -percent`, `phi0 = -138`.
- The treatment contrast is Medicaid eligibility against subsidized marketplace coverage
  (above 100 percent of the line everyone in the window is subsidy-eligible); the
  documented effect is a change of coverage type, not of being insured, so the cost
  side (public cost per Medicaid enrollee net of the subsidy) carries the policy
  question.
- `sample_frame(states="all")` returns every state-year with an `expansion` column
  (`expansion` / `partial` / `none`) for a difference-in-discontinuities check against
  non-expansion states.
- 2019 and 2020 use the Census's redesigned processing system; 2016-2018 the legacy
  one. The recodes are reconciled above; a year indicator is in X.
