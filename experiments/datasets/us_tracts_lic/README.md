# US census tracts: the low-income-community test as an either-cutoff rule (NMTC / Opportunity Zones, ACS 2011-2015)

| Field | Value |
|---|---|
| **Q** | `(pov, mfi_ratio)`: tract poverty rate (%) and tract median family income as % of the benchmark, ACS 2011-2015 5-year, as published by the CDFI Fund |
| **Threshold** | 20 on the poverty rate, 80 on the MFI ratio |
| **Treatment** | `1{Q1 >= 20 or Q2 <= 80}`: New Markets Tax Credit low-income community (LIC), hence eligible for Opportunity Zone (OZ) designation in 2018. Eligibility is the sharp, intent-to-treat treatment; designation is in `extras` |
| **Rule class** | deployed: a cutoff on each score, either one suffices (union of two half-planes); counterfactual: other cutoff pairs, both cutoffs, weighted sums |
| **X** | 2010 decennial counts from the Census Planning Database 2019 (`*_CEN_2010` columns only): log population, log density, urbanized / urban-cluster shares, five age shares, Hispanic / NH Black / NH Asian / NH other shares, institutional and non-institutional group quarters, family / married-couple / female-headed / single-person / with-children household shares, household size, renter share, vacancy, 2010 mail-return rate (+ missing flag), OMB metro flag, 50 state indicators. p = 76. No ACS estimate enters X |
| **Y** | `load()`: change in log(1 + workplace jobs), LODES7 WAC all jobs, 2017 to 2019; `load_placebo()`: the same change 2015 to 2017; `load_lihtc()`: any HUD LIHTC property placed in service 2019 or later |
| **n** | 70,320 tracts (50 states + DC) with both scores and the 2017-2019 outcome; 71,655 for the placebo, 71,820 for LIHTC |
| **Source** | CDFI Fund (cdfifund.gov), Census ACS summary file and Planning Database (www2.census.gov), LEHD LODES7 (lehd.ces.census.gov), HUD LIHTC FeatureServer; all public, no registration |

## Getting the data

    python -m experiments.datasets.us_tracts_lic.download     # idempotent, about 0.5 GB

Everything lands in `data/raw/` (gitignored). The first `load()` builds
`data/processed/us_tracts_lic.csv.gz` (one row per tract, 19 MB) in about 35 s.
Downloaded 2026-10-27; sizes are bytes on disk.

| File | URL | Size |
|---|---|---:|
| `nmtc-2011-2015-lic-nov2-2017-4pm.xlsx` (the scores and CDFI's LIC flag, 74,133 tracts) | https://www.cdfifund.gov/system/files/documents/nmtc-2011-2015-lic-nov2-2017-4pm.xlsx | 7.8 MB |
| `ozone-information-resource.2.27.18-locked2.xlsb` (OZ-eligible LIC and contiguous tracts, tract renumbering) | https://www.cdfifund.gov/system/files/documents/ozone-information-resource.2.27.18-locked2.xlsb | 0.86 MB |
| `designated-qozs.12.14.18.xlsx` (8,764 designated tracts, tract type, ACS vintage) | https://www.cdfifund.gov/system/files/documents/designated-qozs.12.14.18.xlsx | 0.28 MB |
| `opportunity-zones-9-10-2019.zip` (designated-QOZ shapefile; same 8,764 tracts, not read by the adapter) | https://www.cdfifund.gov/system/files/documents/opportunity-zones=8764.-9-10-2019.zip | 46.7 MB |
| `acs2015/20155{st}0048000.zip`, 52 files (B17001, tracts) | https://www2.census.gov/programs-surveys/acs/summary_file/2015/data/5_year_seq_by_state/{State}/Tracts_Block_Groups_Only/ | 26.8 MB |
| `acs2015/20155{st}0064000.zip`, 52 files (B19113, tracts) | same folders | 26.6 MB |
| `acs2015/g20155{st}.csv`, 52 files (geography; county to CBSA from summary level 322) | same folders | 84.2 MB |
| `acs2015/allgeo/20155{st}0064000.zip`, 52 files (B19113, statewide and Puerto Rico CBSAs) | .../5_year_seq_by_state/{State}/All_Geographies_Not_Tracts_Block_Groups/ | 55.4 MB |
| `acs2015/20155us0064000.zip`, `acs2015/g20155us.csv` (B19113 for the 917 CBSAs) | .../5_year_seq_by_state/UnitedStates/All_Geographies_Not_Tracts_Block_Groups/ | 12.3 + 5.0 MB |
| `pdb2019trv6_us.zip` (Planning Database 2019, tracts) | https://www2.census.gov/adrm/PDB/2019/pdb2019trv6_us.zip | 64.3 MB |
| `lodes7/{st}_wac_S000_JT00_{2015,2017,2019}.csv.gz`, 149 files | https://lehd.ces.census.gov/data/lodes/LODES7/{st}/wac/ | 157.1 MB |
| `lihtc_hud.csv` (50,566 projects, 17 fields, paged REST query) | https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/LIHTC/FeatureServer/0/query | 4.1 MB |

Total 491.5 MB. LODES7 has no 2017 or 2019 file for Alaska and no 2019 file for Arkansas
or Mississippi, and no workplace files for Puerto Rico. LODES7 ends in 2019, so there is
no later year. The ACS sequence and column of each table come from
`ACS_5yr_Seq_Table_Number_Lookup.txt` (B17001: sequence 0048, cells 1-2 at columns 7-8;
B19113: sequence 0064, column 75). The adapter reads the CDFI spreadsheets with the
standard-library readers in `_excel.py`, since the environment has no openpyxl or pyxlsb.

## The rule

26 USC 45D(e): a tract is a low-income community if its poverty rate is at least 20%, or
its MFI is at most 80% of the benchmark: the greater of statewide and metro-area MFI for
a metro tract, statewide MFI otherwise. CDFI publishes both scores for every 2010 tract,
from the 2011-2015 ACS, with its own yes/no flag. Rev. Proc. 2018-16 made these tracts
eligible for OZ nomination in 2018.

Checks over all 74,133 tracts in CDFI's file (`adapter.rule_check()`):
- **Rule against CDFI's LIC flag:** applied to CDFI's published scores, the rule
  reproduces the flag for all 74,133 tracts (agreement 1.000). A score shown as N/A
  never qualifies. In the sample, `extras["eligible_published"]` is this flag.
- **Rule against the OZ eligible-LIC list** (information resource of Feb 27 2018, 31,848
  tracts): agreement 0.9977. Every rule-eligible tract is on the list. The 168 listed
  tracts that fail the rule are CDFI's technical-correction additions: high-migration
  rural tracts with an MFI ratio of 80-85%, low-population tracts in Empowerment Zones,
  and territories.
- **Scores recomputed from the ACS summary file.**
  - Poverty is 100 x B17001_002 / B17001_001, rounded half up to 0.1. It equals CDFI's
    value for every tract with a value.
  - The MFI ratio uses B19113 for the tract, the state and the county's CBSA (OMB metro
    flag from CDFI's file). It is within 0.01 of CDFI's for all 72,697 tracts where both
    exist.
  - The rule on the recomputed scores agrees with CDFI's flag for 0.9997 of the 74,001
    ACS tracts. All 20 disagreements are tracts where CDFI shows the MFI ratio as N/A
    although B19113 has an estimate, and the recomputed ratio is <= 80. CDFI shows N/A
    for 251 such tracts; the reason is not documented, and most have poverty >= 20.
- **Benchmark** (`extras["benchmark_type"]`, `["benchmark_mfi"]`): the metro MFI for
  37,353 tracts and the state MFI for 35,344. In Puerto Rico CDFI used the greater of
  Puerto Rico and metro MFI (San Juan-Carolina-Caguas, 25,617 > 22,976), not the
  possession-wide MFI the statute names. The adapter follows CDFI.
- Designation: 8,764 tracts were designated; 8,488 of them pass the rule. Of the other
  276: 198 are non-LIC contiguous tracts, 50 are LICs that states qualified with the
  2012-2016 ACS (allowed by the IRS), and 28 are on CDFI's corrected LIC list (the
  high-migration, Empowerment Zone and similar additions).

Sample: drop the four island territories (132 tracts) and Puerto Rico (945, no LODES
workplace files), then 1,236 tracts with a score shown as N/A (mostly empty or tiny
tracts), then 1,500 tracts without the outcome (Arkansas 681, Mississippi 654, Alaska
165). No tract with both scores lacks covariates. n = 70,320.

## Readiness (2026-10-27)

`python -m experiments.scripts.multi_rule_readiness us_tracts_lic`

| Quantity | Poverty rate | MFI ratio |
|---|---:|---:|
| R^2 of Q on X | 0.750 | 0.684 |
| sd(T), sd(eta) | 10.7, 6.2 | 35.0, 23.8 |
| sd(eta) max/min across T deciles | 2.89 | 2.69 |

- corr(T1, T2) = -0.845 and corr(eta1, eta2) = -0.261.
- In natural units the weakest direction of T is about (-0.97, -0.26), mostly poverty
  because its scale is smaller. Its sd is 5.55, 0.32 of the root mean sd(eta). Scaled
  by sd(eta_j), the weakest direction is about (-0.64, -0.77), both scores moving
  together, with sd 0.63 sd(eta). The trade-off between the scores is much better
  identified than in `chile_two_score` (0.18).
- 42.2% are eligible: 25.6% on both criteria, 4.4% on poverty only, 12.2% on MFI only.
  The script's near-boundary share (0.00001) assumes a weighted-sum rule; ignore it.
  13.0% lie within 0.25 sd(eta_j) of the L-shaped boundary of the either rule.
- State indicators add about 0.025 to each R^2 (0.724 and 0.659 without them).
- On the log scale (log(1 + poverty), log ratio) the sd(eta) ratios fall to 1.66 and
  1.35, with R^2 0.73 and 0.76.
- Outcome: mean change in log jobs 2017-2019 is 0.024 (sd 0.25), 0.030 for ineligible
  and 0.015 for eligible tracts. 0.02% of tracts have zero jobs in 2017.

## Caveats

- **Geography.** The ACS 2011-2015, LODES7 blocks, the Planning Database's 2010 counts
  and HUD's LIHTC tract codes all use 2010 tracts. Twenty-six tracts were renumbered
  after 2010 without boundary changes (CDFI's "Tract Number Changes" tab). Every source is
  mapped to the 2010 number (LIHTC projects placed in service 2019 or later match a 2010
  tract 99.9% of the time). LODES7 block codes are 2010 blocks, so jobs aggregate
  exactly.
- **Heteroskedastic noise.** In natural units sd(eta) changes by a factor of 2.7-2.9
  across T deciles for both scores, so X independent of eta fails, as in every real
  dataset screened so far. The noise is mostly ACS sampling error, which grows with the
  level of each score.
- **Discrete score.** CDFI's poverty rate is rounded to 0.1; 159 tracts sit exactly at
  20.0 and are eligible. The MFI ratio is continuous.
- **5% contiguous exception.** A state could designate a non-LIC tract contiguous to a
  designated LIC if its MFI was at most 125% of that LIC's, for up to 5% of its
  designations. These tracts (198 designated; `extras["oz_contiguous_listed"]` marks the
  10,312 eligible ones, 24% of ineligible sample tracts) are treated by OZ but not by
  the rule.
- **Fuzzy designation.** Governors picked up to 25% of each state's LICs; 24.6% of
  eligible sample tracts were designated (`extras["oz_designated"]`). Eligibility also
  opens NMTC investment, which predates 2018, so D bundles NMTC and OZ eligibility, and
  NMTC eligibility under earlier ACS vintages is correlated with D.
- **Other vintages.** 51 designated tracts were qualified with the 2012-2016 ACS. NMTC
  eligibility is updated with each ACS release; only the 2011-2015 vintage (CDFI's
  November 2017 file) is here.
- The 2010 mail-return rate is missing for 0.6% of sample tracts (update / leave areas).
  It is set to the median, with a flag.
- LIHTC outcomes are rare (1.9% of tracts have a project placed in service in 2019 or
  later). HUD's file covers placements through 2020, plus 55 in 2021-2022.
