# PM2.5 nonattainment designations (Clean Air Act): a cutoff on the county's maximum monitor design value, and an "either" rule on two standards

| Field | Value |
|---|---|
| **Q** | `load()`: county maximum of the complete 2001-2003 annual PM2.5 design values (DV, ug/m3) minus 15.05. `load_two_score()`: (max annual DV, max 24-hour DV) 2001-2003, shape (n, 2), in ug/m3. `load_2006()`: max 24-hour DV 2006-2008 minus 35.5. `load_2012()`: max annual DV 2011-2013 minus 12.05. `load_pooled()`: (DV - cutoff) / level |
| **Threshold** | 0 (single score; D = 1{Q >= 0}); `load_two_score()`: (15.05, 65.5). The published levels are 15.0 and 65 (1997), 35 (2006) and 12.0 (2012); a DV is rounded half up to 0.1 (annual) or 1 (24-hour), so "DV above the level" is exactly "DV >= level + half a unit" |
| **Treatment** | D = 1 is the county rule for nonattainment (a monitor in the county violates), the side above the cutoff. `extras["eligible_published"]` is EPA's designation (Green Book, first effective year of the round) |
| **Rule class** | single cutoff on the county maximum; `load_two_score()`: a cutoff on each of two scores with "either" logic. Cutoffs are the same for every county |
| **X** | log population and log density, manufacturing share of employment, % urban, centroid latitude and longitude, EPA region, earliest PM levels (1999-2000 PM2.5, 1996-1998 PM10, with missing indicators); later rounds also prior PM2.5 nonattainment; pooled also round indicators (20-25 columns) |
| **Y** | change in the annual mean PM2.5 at the county's monitors from the DV period to 3-5 years after designation (2008-2010, 2012-2014, 2018-2020); continuous |
| **n** | `load()` 403 counties (73 treated); `load_two_score()` 388 (71); `load_2006()` 414 (50); `load_2012()` 409 (24); `load_pooled()` 1,226 county-rounds, 570 counties (147) |
| **Source** | EPA design-value reports (www.epa.gov/air-trends/air-quality-design-values), EPA Green Book (www.epa.gov/green-book), EPA AQS annual summaries (aqs.epa.gov/aqsweb/airdata), Census population, Gazetteer, urban/rural and County Business Patterns (www2.census.gov); rule from 70 FR 944, 70 FR 19844, 74 FR 58688, 80 FR 2206, 80 FR 18535 |

## The rule

Under Clean Air Act section 107(d), EPA designates areas after each new or revised
NAAQS. A nonattainment area must adopt a state implementation plan with controls on
stationary, mobile and area sources (CAA part D, subpart 4 for PM). The data rule is 40
CFR Part 50, Appendix N. A monitor site's

- **annual DV** is the 3-year mean of annual means, each the mean of four quarterly means,
  rounded to 0.1 ug/m3;
- **24-hour DV** is the 3-year mean of annual 98th percentiles, rounded to 1 ug/m3.

A site violates a standard if its DV is above the level. The 1997 preamble (70 FR 944,
section VII) states the county rule: "if a PM2.5 monitor was violating the standard based
on the 2001-2003 data, at a minimum we designated the entire county where that monitor is
located as nonattainment". EPA then adds nearby counties that "contribute" to the
violation, starting from the metropolitan area. In a few large western counties it
designates only part of the county.

| Round | Standard(s) | DV period | Final rule | Effective | Green Book NAAQS |
|---|---|---|---|---|---|
| 1997 | annual > 15.0 or 24-hour > 65 | 2001-2003 | 70 FR 944 (2005-01-05), amended 70 FR 19844 (2005-04-14) | 2005-04-05 | PM-2.5 (1997) |
| 2006 | 24-hour > 35 (annual kept at 15.0) | 2006-2008 | 74 FR 58688 (2009-11-13); deferred areas 2011-02-03 | 2009-12-14 | PM-2.5 (2006) |
| 2012 | annual > 12.0 | 2011-2013 | 80 FR 2206 (2015-01-15), amended 80 FR 18535 (2015-04-07) | 2015-04-15 | PM-2.5 (2012) |

The levels, DV periods and dates are from the Federal Register notices in `data/raw/docs/`.
Two steps make the deployed designation differ from the county rule:

- **Changes before the effective date.** In both the 1997 and the 2012 rounds EPA let
  states submit the next year's certified data. It withdrew a nonattainment designation
  if every monitor in the area met the standard over the next 3-year period.
  - 1997 round (70 FR 19844): Athens GA, Elkhart IN, Lexington KY, Marion County WV, San
    Diego CA, Toledo OH and Youngstown OH-PA, on 2002-2004 data. Columbus GA-AL also
    changed, on 2002-2004 data with spatial averaging approved (15.04).
  - 2012 round (80 FR 18535): Canton OH, Allentown PA, Johnstown PA, Cincinnati-Hamilton
    OH-KY and Louisville KY-IN, on 2012-2014 data.
  - These counties are flagged in `extras["pre_effective_reversal"]`.
- **Deferrals.** In 2009 EPA deferred Pinal AZ, Plumas CA and Shasta CA, which violated on
  2006-2008 data but not on 2005-2007. It designated West Central Pinal in 2011
  (`extras["na_later"]`); the Green Book has no 2006-standard designation for the other
  two.

  In the 2012 round it designated Illinois unclassifiable (no valid 2011-2013 DV, so no
  Illinois county is in the sample). It also deferred Florida, Tennessee outside
  Chattanooga, and eight areas in Georgia with neighbouring Alabama and South Carolina
  counties, for insufficient data. Deferred counties that do have a valid DV (16 in
  Florida, 9 in Georgia, 1 in Tennessee) are in the sample. None violates, and they were
  later designated unclassifiable/attainment.

**Which DVs.** Q uses EPA's own site DVs, which include its collocation, substitution and
event rules:

- the 2001-2003 and 2006-2008 site listings in the 2006-2008 DV report (AQS as of
  2009-07-07);
- Table 5 of the 2011-2013 report (final 2014-08-28).

Only complete DVs count: status 1 (complete, violating) or 2 (complete, meeting) in the
2009 report, and the "valid" column in the 2014 report. Appendix N also accepts an
incomplete DV above the level (status 3), but EPA did not designate on those:

- 2006 round: none of the 11 counties that violate only through incomplete sites was
  designated;
- 1997 round: 5 of 13 were, all inside areas designated because of another county's
  complete violating DV.

Sites EPA marks as not comparable to the NAAQS (status 5) are dropped. For the 2012
round, the county DV is EPA's county summary (Table 4). It equals the maximum valid site
DV in 485 of 486 counties. The exception is Imperial, CA (14.3, from 2011 and 2013 data
"submitted to, but not currently in, AQS", the report's footnote 4).

**The 24-hour arm of 1997.** Only five counties had a complete 24-hour DV above 65
(Fresno, Kern, Kings, Riverside and Tulare, CA; 66-76). All five also violate the
annual standard, so the "either" rule and the annual rule designate the same counties.
`load_two_score()` is a genuine two-score rule, but its second arm never binds alone at
the 1997 cutoff.

## Rule check (2026-10-29)

`adapter.rule_check()`: every county with a complete DV for the round's standard, the
rule against the Green Book designation (first designated in 2005, 2009 or 2015).

| Round | Counties | Rule | Published | Agree | Rule and published | Rule only | Published only | P(pub \| rule) | P(pub \| no rule) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1997 | 525 | 81 | 119 | 0.897 | 73 | 8 | 46 | 0.901 | 0.104 |
| 2006 | 512 | 51 | 94 | 0.904 | 48 | 3 | 46 | 0.941 | 0.100 |
| 2012 | 486 | 24 | 20 | 0.984 | 18 | 6 | 2 | 0.750 | 0.004 |

**Above the cutoff the rule is exact up to documented changes.** Every "rule only" county
is explained:

- 1997: the 8 are the April 2005 changes. Russell AL (Columbus) 15.8, San Diego 15.8,
  Clarke GA 15.6, Fayette KY 15.6, Marion WV 15.4, Elkhart IN 15.2, Mahoning OH 15.2 and
  Lucas OH 15.1.
- 2006: the 3 are the deferred Pinal (48, designated 2011), Plumas (49) and Shasta (48).
- 2012: the 6 are the April 2015 changes. Butler OH 13.6, Hamilton OH 12.3, Cambria PA
  12.3, Northampton PA 12.2, Clark IN 12.1 and Stark OH 12.1.

So every violating county that was neither changed before the effective date nor deferred
was designated (73/73, 48/48, 18/18).

**Below the cutoff the designation is fuzzy.**

- 1997: all 46 "published only" counties lie in a designated area that contains a
  violating county. These are EPA's contributing counties. Their median Q is -1.05
  (10th-90th percentile -2.7 to -0.35).
- 2006: 42 of 46 are, and the other 4 are in Knoxville TN. EPA's 2009 county summary does
  not report a 2006-2008 DV for Knox County because a previously violating site had no
  valid DV.
- 2012: 2 of 2 are.
- A further 17 (1997), 6 (2006) and 0 (2012) designated counties have monitors but no
  complete DV, so they are not in the sample. Counties designated without any monitor are
  not in the sample either.

**Sharpness near the cutoff (1997, `load()` sample).** Share designated by 0.5 ug/m3 bin
of Q:

| Q bin | [-1.5,-1) | [-1,-0.5) | [-0.5,0) | [0,0.5) | [0.5,1) | [1,1.5) |
|---|---:|---:|---:|---:|---:|---:|
| Counties | 35 | 21 | 20 | 18 | 14 | 12 |
| Share designated | 0.26 | 0.33 | 0.50 | 0.83 | 0.79 | 1.00 |

The designated share rises steadily toward the cutoff from below (contributing counties in
violating metros). It jumps about 0.3 at the cutoff, and the April 2005 changes sit just
above it. The local-linear first stage is therefore modest: see the effect check below.

**AQS cross-check.** I recomputed 2001-2003 site DVs from the AQS annual summaries (one
POC per site, all data). They match EPA's published DV exactly at 53% of 752 complete
sites and within 0.1 at 86%, and they fall on the same side of 15.05 at 99.6%. The
differences come from collocated-monitor substitution, concurred-event exclusion and data
revisions. Q uses EPA's published values.

## Outcome and effect check

Y is built for each PM2.5 site (AQS parameter 88101, "PM25 Annual 2012" rows, i.e. the
mean of quarterly means; all data, including event-flagged days). Each site-year uses one
POC: filter-based 24-hour samplers first, then the most valid days. A site's change is
mean(post window) - mean(DV period), and it needs at least two annual means in each
window. Y is the county mean of its sites' changes. Sites per county average 1.5.

| Round | Y mean | Y sd | Mean pre | Mean post | Counties with complete DV | With Y |
|---|---:|---:|---:|---:|---:|---:|
| 1997 (2001-03 to 2008-10) | -2.33 | 1.41 | 12.3 | 9.9 | 525 | 403 |
| 2006 (2006-08 to 2012-14) | -2.55 | 1.46 | 11.2 | 8.7 | 512 | 414 |
| 2012 (2011-13 to 2018-20) | -1.49 | 1.51 | 9.1 | 7.6 | 486 | 409 |

Other outcomes in the cache:
- `y_change_dvsite`: the change at the county's DV site (`load_1997_dvsite()`);
- `y_change_early`: the change 1-3 years after designation;
- `y_pre`, `y_post`.

**Local jump at the cutoff.** `adapter.local_jump(sample, h, d=..., covariates=...)` fits
a local-linear regression with a triangular kernel and separate slopes, with HC1 standard
errors (clustered by county for the pooled sample). "First stage" is the jump in the
published designation.

| Loader | h | n in window | Jump in Y (se) | First stage (se) | Jump in Y with X (se) | First stage with X (se) |
|---|---:|---:|---:|---:|---:|---:|
| `load()` (1997) | 1.0 | 73 | -0.35 (0.43) | 0.12 (0.27) | -0.56 (0.50) | 0.16 (0.39) |
| | 2.0 | 148 | -0.12 (0.30) | 0.15 (0.17) | -0.09 (0.31) | 0.09 (0.20) |
| | 3.0 | 221 | 0.08 (0.24) | 0.24 (0.13) | 0.12 (0.24) | 0.18 (0.14) |
| `load_2006()` | 4 | 93 | 0.44 (0.59) | 0.52 (0.14) | 0.95 (0.23) | 0.47 (0.17) |
| | 6 | 160 | 0.15 (0.50) | 0.54 (0.11) | 0.78 (0.25) | 0.54 (0.13) |
| | 9 | 262 | 0.16 (0.44) | 0.56 (0.09) | 0.76 (0.22) | 0.55 (0.10) |
| `load_2012()` | 1.5 | 82 | -0.96 (0.65) | 0.08 (0.14) | -0.94 (0.50) | 0.02 (0.12) |
| | 2.0 | 128 | -0.75 (0.56) | 0.24 (0.17) | -0.76 (0.38) | 0.23 (0.14) |
| | 3.0 | 227 | -0.53 (0.49) | 0.32 (0.17) | -0.56 (0.35) | 0.32 (0.15) |
| `load_pooled()` | 0.10 | 251 | -0.05 (0.32) | 0.32 (0.13) | 0.30 (0.26) | 0.28 (0.11) |
| | 0.20 | 597 | -0.07 (0.23) | 0.41 (0.09) | 0.21 (0.19) | 0.37 (0.08) |

- **The local effect is not detected.**
  - In the main 1997 round the jump is -0.1 ug/m3 with se 0.3, and the first stage is
    0.15-0.24.
  - The 2012 round leans negative: -0.75 (0.56), or -0.76 (0.38) with X. Its early window
    (2016-2018) gives -0.79 (0.44), about -9% in logs (se 4%).
  - The 2006 round's jump is positive, +0.8 (0.25) with X. A designation is unlikely to
    raise PM2.5, so this probably reflects how western wood-smoke and wildfire counties
    near 35 differ from the rest (an inference, not tested).
  - The pooled Wald estimate at h = 0.2 is -0.17 (0.58).
- The 98th percentile as Y (2006 and 2012 rounds), longer windows (5-8 years) and log
  changes gave the same picture. These were checked in the session, not kept as loaders.
- The raw contrast is large: Y is -3.84 for 1997 rule-treated counties against -1.99 for
  the rest. It reflects the steady decline of Y in Q, about -0.35 per ug/m3, which is
  regression to the mean plus faster declines where PM was high. It is not a jump.
- **Literature (checked).** All three studies estimate average effects of designation,
  not the local contrast at the county cutoff.
  - Sanders, Barreca and Neidell (2020), "Estimating Causal Effects of Particulate Matter
    Regulation on Mortality", *Epidemiology* 31(2):160-167, doi:10.1097/EDE.0000000000001153.
    In a difference-in-differences design on the 2005 designations, PM2.5 fell 1.59
    ug/m3 (95% CI 1.39-1.80) in nonattainment counties relative to attainment counties,
    and mortality at 65+ fell 0.93% (abstract, EPA HERO record 6671796).
  - Zigler, Choirat and Dominici (2018), *Epidemiology* 29(2):165-174,
    doi:10.1097/EDE.0000000000000777. For the eastern US, the PM2.5 reductions after the
    2005 designations could not be conclusively attributed to the designations on
    average, because regional programs (e.g. CAIR) lowered PM2.5 everywhere. They found
    effects in a subset of areas.
  - Auffhammer, Bento and Lowe (2009), *JEEM* 58(1):15-26, doi:10.1016/j.jeem.2008.12.004
    (PM10, 1990 CAAA). Nonattainment status had no effect at the average monitor, but
    monitors above the annual standard fell 7-9 ug/m3 (11-14%).
  - Our DV-site outcome (`y_change_dvsite`) follows that last finding but shows no jump
    either: -0.15 (0.31) at h = 2.
  - A local effect of the size Sanders et al. report (-1.6 times a first stage of 0.2-0.4,
    so -0.3 to -0.6) lies inside our 95% intervals but is not detected.

## Readiness numbers

| Loader | n | p | Treated share | R^2 of Q on X | Without PM lags | Without PM lags and prior status |
|---|---:|---:|---:|---:|---:|---:|
| `load()` (1997 annual) | 403 | 20 | 0.181 | 0.839 | 0.426 | 0.426 |
| `load_two_score()` annual | 388 | 20 | 0.183 | 0.838 | 0.427 | 0.427 |
| `load_two_score()` 24-hour | 388 | 20 | | 0.733 | 0.388 | 0.388 |
| `load_2006()` | 414 | 22 | 0.121 | 0.580 | 0.425 | 0.325 |
| `load_2012()` | 409 | 23 | 0.059 | 0.600 | 0.420 | 0.235 |
| `load_pooled()` | 1,226 | 25 | 0.120 | 0.599 | 0.327 | 0.274 |

- In the two-score sample, corr(annual, 24-hour) = 0.81.
- **Boundary mass.** Counties with a complete DV within 10% of the level of the cutoff
  (`adapter.boundary_mass()`): 144 in 1997, 100 in 2006 and 68 in 2012. Within 5%: 70,
  56 and 29. The 1997 round is the main loader.
- **Score distributions** (percentiles 5/25/50/75/95, `load()` samples):
  - 1997 annual: 6.7/10.6/12.5/14.3/17.2 (range 4.4-27.8);
  - 2006 24-hour: 17.7/25/29/32/41 (range 9-70);
  - 2012 annual: 6.0/8.3/9.4/10.4/12.3 (range 3.9-18.1).
- **The PM lag carries most of R^2 in 1997.** The 1999-2000 PM2.5 level is the network's
  first two years, just before the DV period.

## Getting the data

    python -m experiments.datasets.epa_pm25.download          # idempotent, 194 MB
    python -m experiments.datasets.epa_pm25.download --check  # sizes and SHA-256

- **Access.** Every file came from a plain request with an honest user agent ("PerfRDD
  research download (python urllib)"): no login, form, CAPTCHA or click-through.
- **robots.txt (checked 2026-10-29).**
  - aqs.epa.gov has none (404).
  - www.epa.gov, www3.epa.gov (`/airquality/greenbook/`, `/pmdesignations/`),
    www2.census.gov and www.govinfo.gov do not disallow these paths. www3 asks for a
    crawl delay of 1 s, which the script observes.
  - BEA's county employment zips (apps.bea.gov/regional/zip/) are disallowed, so
    manufacturing employment comes from County Business Patterns instead.
- **Speed.** AQS served each file in 1-2 minutes.
- **Reading.** Nothing downloaded is executed. The adapter reads:
  - the zips with `zipfile`;
  - the Green Book `.dbf` files with a small standard-library dBASE reader (`read_dbf`);
  - the DV reports with `experiments/datasets/fl_school_grades/_spreadsheet.py`
    (imported, not copied).

  The Green Book `.xls` versions are not used: that reader fails on them, and the `.dbf`
  files hold the same tables.
- **Caches** (`data/processed/`, written atomically):
  - `aqs_pm_site_years.csv.gz` (0.5 MB): one PM2.5 or PM10 annual mean per site and year,
    1996-2020;
  - `epa_pm25_county_rounds.csv.gz` (0.2 MB): the county x round panel with DVs, rule,
    designations, outcomes and covariates.

  Building both takes about 15 s. Delete them after changing `adapter.py`.

| File (`data/raw/`) | Bytes | Source URL | SHA-256 (first 16) |
|---|---:|---|---|
| `aqs/annual_conc_by_monitor_<y>.zip`, y = 1996-2020 (25 files) | 2.8-5.8 MB each, 120.6 MB total | https://aqs.epa.gov/aqsweb/airdata/annual_conc_by_monitor_<y>.zip | in `download.py` |
| `dv/dv_pm25_2006_2008rev102809.xls` | 13,574,144 | https://www.epa.gov/sites/default/files/2016-09/dv_pm25_2006_2008rev102809.xls | 6f2ba7ed1d5ffbbf |
| `dv/pm25_designvalues_20112013_final_08_28_14.xlsx` | 1,374,635 | https://www.epa.gov/sites/default/files/2016-09/pm25_designvalues_20112013_final_08_28_14.xlsx | fb32c45b13f550ea |
| `greenbook/phistory.dbf` | 273,682 | https://www3.epa.gov/airquality/greenbook/downld/phistory.dbf | 9dbf0de07ee6facc |
| `greenbook/nayro.dbf` | 745,423 | https://www3.epa.gov/airquality/greenbook/downld/nayro.dbf | 5a22cf4b7750eb3a |
| `greenbook/greenbook_exportdoc.pdf` | 52,292 | https://www3.epa.gov/airquality/greenbook/downld/greenbook_exportdoc.pdf | 6057bc2d7e8599eb |
| `census/co-est00int-tot.csv` | 370,855 | https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv | 273e56203b61c2cc |
| `census/co-est2019-alldata.csv` | 3,644,730 | https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv | 654032fbb805bc9a |
| `census/Gaz_counties_national.zip` | 167,168 | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip | 592fd19e5027658a |
| `census/PctUrbanRural_County.txt` | 455,024 | https://www2.census.gov/geo/docs/reference/ua/PctUrbanRural_County.txt | c9b83a06ff642cf6 |
| `census/cbp00co.zip`, `cbp05co.zip`, `cbp10co.zip` | 16,024,897; 14,931,003; 16,025,092 | https://www2.census.gov/programs-surveys/cbp/datasets/<y>/cbp<yy>co.zip | a57c70e9561c475e; 0e1f410399367e96; c2d1b4b3e7de69e5 |
| `docs/fr_2005-01-05_pm25_1997_designations.pdf` (70 FR 944) | 427,079 | https://www.govinfo.gov/content/pkg/FR-2005-01-05/pdf/05-1.pdf | 262f3b0a3ac65189 |
| `docs/fr_2005-04-14_pm25_1997_supplemental.pdf` (70 FR 19844) | 188,173 | https://www.govinfo.gov/content/pkg/FR-2005-04-14/pdf/05-7227.pdf | 8d112ab7d7208835 |
| `docs/fr_2009-11-13_pm25_2006_designations.pdf` (74 FR 58688) | 448,999 | https://www.govinfo.gov/content/pkg/FR-2009-11-13/pdf/E9-25711.pdf | e7844ea14e423b58 |
| `docs/fr_2015-01-15_pm25_2012_designations.pdf` (80 FR 2206) | 547,249 | https://www.govinfo.gov/content/pkg/FR-2015-01-15/pdf/2015-00021.pdf | d4e85e8182c8d44d |
| `docs/fr_2015-04-07_pm25_2012_additional.pdf` (80 FR 18535) | 295,144 | https://www.govinfo.gov/content/pkg/FR-2015-04-07/pdf/2015-07948.pdf | a71020d4108fb449 |
| `docs/pm25_1997_preamble.pdf`, `pm25_1997_tsd_ch2.pdf`, `pm25_1997_apr2005_tsd.pdf`, `pm25_1997_finaltable.htm` | 290,993; 150,975; 4,288,913; 32,914 | https://www3.epa.gov/pmdesignations/1997standards/documents/ (final/preamble.pdf, final/TSD/Ch2.pdf, Apr05/tsd/tsd.pdf, final/finaltable.htm) | e145045a308f0a12; dac1682a9d736763; 203c96d0b6d5c76d; 1f6483f730de0b54 |

AQS annual summaries (`aqs/annual_conc_by_monitor_<y>.zip`, from https://aqs.epa.gov/aqsweb/airdata/; full hashes in `download.py`):

| Year | Bytes | SHA-256 (first 16) |
|---|---:|---|
| 1996 | 2,803,197 | 114acf9c2c45dc99 |
| 1997 | 2,825,329 | 97d6e2b5275d1817 |
| 1998 | 2,965,838 | c94e062fb4f8fe20 |
| 1999 | 3,777,596 | e432444e15b3ea86 |
| 2000 | 4,350,920 | cffda9d8fec29ed9 |
| 2001 | 4,954,729 | bc1a195bc6bcd614 |
| 2002 | 5,566,297 | 8d18b46991d969b1 |
| 2003 | 5,694,693 | 2677eeeee4575a07 |
| 2004 | 5,751,150 | a8d0a4f0ee92ea6b |
| 2005 | 5,833,760 | e7d2388b372a259e |
| 2006 | 5,660,486 | 307974a00f152208 |
| 2007 | 5,536,284 | c7d077f2ce87b6e4 |
| 2008 | 5,406,728 | 1a12cf508398db53 |
| 2009 | 5,503,861 | d317ab54710846e2 |
| 2010 | 5,343,023 | 1e4899a8eee34d5b |
| 2011 | 5,270,616 | e8b5e20186106ddb |
| 2012 | 5,209,527 | 993967a684bbf317 |
| 2013 | 5,075,450 | 1d3ba698a7f0326b |
| 2014 | 5,012,002 | b5f98e45fe57e999 |
| 2015 | 4,743,738 | ce365d493ac5de3f |
| 2016 | 4,627,151 | e693e5f87409cf22 |
| 2017 | 4,465,459 | 76046e5278961416 |
| 2018 | 4,526,482 | 0a6ab33afcfb48fa |
| 2019 | 4,356,274 | 09808acc29300c80 |
| 2020 | 4,308,901 | b2c0c6b89dfc125e |

All files were downloaded on 2026-10-29. EPA regenerates the AQS files twice a year (these
are "as of 2026-06-25") and the Green Book monthly (export date 2026-09-30), so
`--check` reports a changed hash rather than failing.

- **Not available.** EPA's design-value page has no 2001-2003 report. The 2001-2003 site
  DVs come from the site listing inside the 2006-2008 report, which uses AQS as of July
  2009 rather than the data of December 2004.
- **Still reproduces exactly.** With those DVs, the 1997 rule reproduces every designation
  above the cutoff once the April 2005 changes are accounted for.

## Caveats

- **The designation is fuzzy below the cutoff and moderately sharp at it.**
  - EPA designates areas, not counties, so contributing counties just below the cutoff are
    often designated. Near the 1997 cutoff the designated share is 0.3-0.5 below and 0.8+
    above.
  - The local first stage is 0.1-0.25 in 1997, 0.5 in 2006 and 0.2-0.3 in 2012.
  - D is the county rule. The Green Book designation is `extras["eligible_published"]`.
- **The effect check is flat** (see above). The treatment is a real regulation with
  average effects in the literature, but the local reduced-form jump is small and
  imprecise with a few hundred counties.
- **Score construction.**
  - Q is a maximum over 1-13 listed sites, so counties with more monitors have mechanically
    higher Q. `extras["n_sites_listed"]` counts them; it is not in X.
  - The DV is EPA's rounded value, so Q lies on a 0.1 (annual) or 1 (24-hour) grid.
- **Outcome.**
  - Y mixes sites. A county's sites differ in level and some close or open between
    windows; Y uses only sites present in both windows.
  - Y includes event-flagged days, including wildfire smoke (2018 and 2020 matter for the
    2012 round). Excluding flagged days would let the outcome depend on agencies'
    flagging, which nonattainment areas have more reason to request.
  - Y overlaps with the next round's DV period (the 1997 round's 2008-2010 against the
    2006 round's 2006-2008).
- **Covariate timing and construction.**
  - Population and manufacturing share are from the year before the DV period (2000,
    2005, 2010). % urban is the 2010 Census for every round. Land area is the 2010
    Gazetteer.
  - CBP suppressed cells use the midpoint of their size class (share clipped to 0-100).
  - County FIPS changes (e.g. Broomfield CO) leave a few covariates missing; they get
    median imputation and a missing indicator.
- **Rounds are not independent.**
  - Of the counties with a complete DV, 25% (2006 round) and 23% (2012 round) were already
    in a 1997 area, and 20% of the 2012 counties in a 2006 area. Their controls predate
    the round.
  - `prior_na_1997` and `prior_na_2006` are in X for those rounds.
  - Pooling repeats counties: 570 counties in 1,226 county-rounds. Cluster on
    `extras["county_id"]`.
  - The pooled score is (DV - cutoff) / level, a share of the standard, so annual and
    24-hour rounds share one scale; round indicators are in X.
- **Other rules not modelled.** Partial-county designations (`extras["na_part"]`) count as
  designated. Classification (moderate/serious) and later redesignations to maintenance
  are not used.

## Suggested policy boxes

- `load()`: the annual cutoff in [13.05, 17.05] ug/m3, i.e. Q in [-2, 2].
  - 148 counties lie within 2 of the cutoff in the sample.
  - The later levels, 12.0 (2012) and 9.0 (2024), lie in the lower half of the 2001-2003
    distribution. A "stricter standard" box [12.05, 15.05] is also supported by the data,
    but has no treatment variation below 15.05.
- `load_two_score()`: annual in [13.05, 17.05] x 24-hour in [35.5, 65.5].
  - At the deployed 65.5 the 24-hour arm never binds alone.
  - At 35.5, the 2006 level, it would add 60 counties with annual < 15.05. At 45.5 it
    would add 10.
  - So the box spans the 1997-to-2006 tightening of the 24-hour standard.
- `load_2006()`: the 24-hour cutoff in [30.5, 40.5].
- `load_2012()`: the annual cutoff in [10.05, 14.05]. Only 24 counties are treated.
- `load_pooled()`: Q in [-0.15, 0.15] (relative to the level).
