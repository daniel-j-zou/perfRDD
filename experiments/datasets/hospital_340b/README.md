# 340B hospital eligibility: a group cutoff on the DSH percentage (SSI + Medicaid fractions)

| Field | Value |
|---|---|
| **Q** | `load()`: disproportionate patient percentage (DPP = SSI fraction + Medicaid fraction, in percent, Worksheet E Part A lines 30 + 31 of cost report t) minus the hospital's group cutoff. `load_two_score()`: (SSI %, Medicaid %), shape (n, 2) |
| **Threshold** | 0 on Q. In DPP units: 27.325 for DSH hospitals (DSH adjustment percentage > 11.75) and 22.765 for rural referral centers and sole community hospitals (adjustment >= 8) |
| **Treatment** | 340B eligibility under the DSH test, D = 1{Q >= 0}: the eligible side is above. Intention to treat: enrollment is not observed directly (see "First stage") |
| **Rule class** | Group-specific cutoffs on one score, which is a fixed-weight sum (1, 1) of two scores. The statutory cutoff is on the DSH adjustment percentage, a piecewise-linear increasing function of DPP; both cutoffs lie on its top segment, so the rule is a cutoff on DPP |
| **X** | 33 columns: log beds, resident-to-bed ratio and teaching flag, case-mix index (+ missing flag), large-urban and other-urban flags, RRC/SCH flag (all Impact File FY t except RRC/SCH); 6 government control types and church nonprofit (base: other nonprofit); 8 census-region and 11 FY indicators. `lagged_y=True` adds Y on report t-1 |
| **Y** | log outpatient charges in the "drugs charged to patients" cost center (Worksheet C Part I, line 73, column 7) on cost report t+2. Other outcomes via `outcome=`; `lead=1` for report t+1 |
| **n** | 25,831 hospital-reports (2,564 hospitals), cost reports beginning in FY 2011-2022; 15,745 eligible. `load_two_score()` (DSH group): 18,640 (10,042 eligible). `load_rrc_sch()`: 7,191 (5,703 eligible) |
| **Source** | CMS HCRIS hospital cost reports (form CMS-2552-10) from downloads.cms.gov; IPPS Impact Files via `experiments/datasets/medicare_dsh`; CMS-1793-F Addendum AAA (340B remedy); 42 U.S.C. 256b(a)(4); HRSA 340B OPAIS registration help |

**Fit: good.**
- **The rule.** It is reproduced from the two fractions: 99.7-99.9% agreement with the
  cost report's own allowable DSH percentage in every year.
- **The effect is large.** Y jumps by 0.24 log points (se 0.07) at the cutoff, which is
  about 27% more outpatient drug charges two reports later.
- **First stage.** 340B drug billing jumps from 39% to 73% across the cutoff (DSH group,
  CY 2018-2022).
- **Covariates.** X predicts Q modestly (R^2 = 0.27).
- **Weaknesses.**
  - The ownership test is only partly observable.
  - Participation is fuzzy around the cutoff.
  - Y is gross charges, at prices the hospital sets.

## The rule

**Statute.** 42 U.S.C. 256b(a)(4) (section 340B(a)(4) of the PHS Act), read 2026-10-29 at
law.cornell.edu.
- **(L)(i) Ownership.** A subsection (d) (IPPS) hospital is a covered entity if it is one of:
  - owned or operated by a unit of state or local government;
  - a public or private nonprofit formally granted governmental powers;
  - a private nonprofit with a state or local government contract to serve low-income
    patients.
- **(L)(ii) DSH test.** It also needs, "for the most recent cost reporting period that ended
  before the calendar quarter involved", a disproportionate share adjustment percentage
  (determined under 1886(d)(5)(F) of the Social Security Act) greater than 11.75 percent.
  Pickle hospitals (1886(d)(5)(F)(i)(II)) qualify regardless.
- **(L)(iii)** It must not buy covered outpatient drugs through a group purchasing
  organization.
- **(M) Children's and free-standing cancer hospitals:** the same 11.75 test.
- **(N) Critical access hospitals:** (L)(i) only, no DSH test.
- **(O) Rural referral centers (RRC) and sole community hospitals (SCH):** (L)(i) and a DSH
  adjustment percentage of at least 8 percent.

**HRSA guidance.** From the OPAIS help page "Qualification Information"
(`data/raw/docs/hrsa_qualification_information.htm`):
- Registration pre-fills the DSH adjustment percentage and the control type "per HCRIS"
  from the most recently filed Medicare cost report.
- Proprietary control types (3-6) are ineligible.
- The RRC flag is Worksheet S-2 line 116, and SCH status is S-2 line 35.
- RRC and SCH need a DSH adjustment percentage "equal to or greater than 8 percent".
- The 11.75 figure was not found in that page's text; it is from the statute.

**From the cost report to Q.** CMS-2552-10 Worksheet E Part A:

| Line | Item | Stored as |
|---|---|---|
| 30 | SSI fraction: Medicare SSI days / Medicare Part A days (CMS's SSI ratio) | fraction, 4 decimals |
| 31 | Medicaid fraction: Medicaid, non-Medicare days / total days | fraction, 4 decimals |
| 32 | DPP = line 30 + line 31 | fraction, 4 decimals |
| 33 | allowable DSH percentage (the "DSH adjustment percentage" HRSA uses) | fraction, 4 decimals |

The adjustment percentage on line 33 comes from the operating DSH formula (42 CFR
412.106(d); see `medicare_dsh`):

    adj = 0 (DPP < 15);  2.5 + 0.65 (DPP - 15) (15 <= DPP < 20.2);  5.88 + 0.825 (DPP - 20.2) (DPP >= 20.2),
    capped at 12 for small urban and most rural hospitals.

- **Why the cap does not matter.** The cap (12) is above both cutoffs.
- **DPP thresholds.** `adj > 11.75` is DPP > 27.3152, and `adj >= 8` is DPP >= 22.7697.
- **Rounding.** Line 33 is rounded to 4 decimals. At DPP = 27.32 it reads 0.1175, which is
  not above 11.75 (10 of 10 reports at that value are ineligible by line 33). At 27.33 it
  reads 0.1176.
  - The rule on the 0.01 grid is therefore DPP >= 27.33 (DSH) and DPP >= 22.77 (RRC/SCH;
    11 of 11 reports at 22.77 read 8.00).
  - The adapter uses the midpoints 27.325 and 22.765, so ties cannot occur.
- **Groups.** RRC/SCH means S-2 line 116 = "Y" or S-2 line 35 > 0 on the same report;
  everyone else is in the DSH group.
- **Eligibility.** D = 1{DPP >= group cutoff}, among nonprofit (control 1-2) and state or
  local government (8-13) hospitals.

**What is not observed.**
- **Ownership.** Whether a private nonprofit has the government contract or governmental
  powers that (L)(i) requires is not in the data. All nonprofits are kept, as in Desai and
  McWilliams (2018) and as HRSA's control-type screen implies.
- **GPO use.** The group purchasing prohibition (L)(iii) is not observed.

## Rule check (2026-10-29)

`adapter.rule_check()` covers every short-term acute report with both fractions (50
states and DC, all control types, Pickle hospitals excluded): 40,296 reports in FY
2011-2024.

| Check | Result |
|---|---|
| line 32 = line 30 + line 31 (to 0.01 point) | 100% every year |
| line 33 = formula on line 32 (to 0.0001), or 0.12 under the cap | 96.7-98.9% by year |
| eligibility from Q vs eligibility from line 33 (> 11.75 or >= 8 by group) | 99.67-99.93% by year; 80 disagreements in all |
| same, in the `load()` sample (`extras["eligible_published"]`) | 99.85%; 40 of 25,831 disagree |

- **The 80 disagreements** (`rule_exceptions()`):
  - 29 reports have line 33 = 0 although DPP is above the cutoff. The hospital did not
    claim DSH on that report.
  - The rest have a line 33 that does not follow from line 32. Examples: a split-period
    report, or a manually entered value (e.g. CCN 360089, FY 2018: DPP 26.29, line 33 =
    13.19, formula 10.90).
- **Line-33 mismatches** that do not change eligibility are mostly small, up to about 1
  point.
- **Group classification** (`group_check()`): S-2 RRC/SCH status against the Impact File
  provider type agrees in 94-98% of IPPS reports a year.
  - The RRC count roughly doubles over the period (S-2: 585 RRC/SCH reports in FY 2011,
    1,090 in FY 2024), mostly urban hospitals reclassified as rural that then qualify as
    RRCs.
- **The published designation is not used.** HRSA's covered-entity list (OPAIS) is not
  downloadable by a plain request (see "Getting the data"). `eligible_published` is
  therefore the line-33 eligibility that HRSA's registration pre-fills, not enrollment.

## First stage (CMS 340B remedy file)

- **What the file is.** CMS-1793-F Addendum AAA lists, for each CCN, the 2018-2022 Medicare
  OPPS underpayment on drugs billed as 340B-acquired.
  - In those years Medicare paid ASP - 22.5% for them, and the 2023 remedy repaid the
    difference.
  - The amount is positive only if the hospital billed 340B-acquired drugs that year.
  - Rural SCHs, children's and cancer hospitals were exempt from the cut.
- **The measure.** `extras["remedy_cy"]` is the amount for calendar year t + lead, when
  that year is in 2018-2022. The first stage is 1{remedy > $1,000}.
- **Result.** The jump is clear at lead 2 and weaker at lead 1, consistent with
  eligibility from report t applying about one to two years after period t.

| Group (lead 2: CY t+2, 10,762 rows) | Below, within 2 points | Above, within 2 points | Local-linear jump (h = 5) | Q < -10 | Q > 10 |
|---|---:|---:|---:|---:|---:|
| DSH | 0.385 | 0.733 | 0.200 (0.045) | 0.042 | 0.817 |
| RRC, not SCH | 0.366 | 0.736 | 0.163 (0.095) | | |
| SCH (exempt from the cut) | 0.093 | 0.095 | uninformative | | |

At lead 1 the DSH-group jump is 0.07 (0.046); the raw means are 0.39 and 0.65.

## Outcome choice and the local jump

- **Main Y.** Log outpatient charges for drugs charged to patients, on report t+2.
- **Why this outcome.** 340B lowers the price at which a hospital buys outpatient drugs,
  while insurers keep paying the same.
  - The documented response is more outpatient drug administration in hospital
    departments.
  - Desai and McWilliams (2018) found more hospital-billed parenteral drug claims in
    hematology-oncology and ophthalmology at the eligibility cutoff, and more
    hematologist-oncologists in hospital-owned practices.
  - Outpatient drug charges measure that volume at list prices, so the effect has an
    unambiguous sign.
- **Why not the other candidates.**
  - Drug cost mixes higher volume with a lower unit price.
  - Outpatient revenue dilutes the drug channel with everything else.
  - Margin and uncompensated care are noisy and have no clear jump (table).

`local_jump(s, h)` fits a local-linear regression with a triangular kernel, bandwidth h
(DPP points) and separate slopes, with standard errors clustered by hospital. Y is
measured on report t+2 (t+1 in the last column):

| Outcome (`outcome=`) | Mean | Jump, h = 3 | h = 5 | h = 8 | h = 5, report t+1 |
|---|---:|---:|---:|---:|---:|
| **log outpatient drug charges** (`log_drug_op_chg`) | 17.15 (median $28.9M) | **0.219 (0.088)** | **0.237 (0.074)** | **0.181 (0.066)** | 0.224 (0.072) |
| drug share of outpatient charges, % (`drug_share_op`) | 10.8 | 1.07 (0.49) | 1.29 (0.41) | 1.12 (0.37) | 1.26 (0.38) |
| log drug cost (`log_drug_cost`) | 16.49 | 0.180 (0.077) | 0.177 (0.065) | 0.103 (0.058) | 0.177 (0.064) |
| log gross outpatient revenue (`log_op_revenue`) | 19.74 | 0.159 (0.068) | 0.142 (0.057) | 0.091 (0.050) | 0.134 (0.055) |
| operating margin, % (`op_margin`) | -4.8 | 0.99 (0.90) | 1.21 (0.73) | 1.05 (0.64) | 0.08 (0.70) |
| uncompensated care, % of opex (`uc_share`) | 4.8 | 0.54 (0.22) | 0.28 (0.18) | 0.08 (0.17) | 0.09 (0.18) |
| drug cost-to-charge ratio (`drug_ccr`) | 0.29 | -0.008 (0.009) | -0.004 (0.008) | -0.006 (0.007) | 0.002 (0.008) |

Robustness of the main jump:

| Check | Jump (se) | n |
|---|---:|---:|
| h = 5, main | 0.237 (0.074) | 8,597 |
| DSH group only | 0.231 (0.088) | |
| RRC/SCH group only | 0.296 (0.134) | |
| FY 2011-2016 | 0.226 (0.105) | |
| FY 2017-2022 | 0.227 (0.105) | |
| Donut, \|Q\| >= 0.5 | 0.224 (0.103) | 7,671 |
| Placebo cutoff at Q = -10 (below-side data only, h = 4) | 0.04 (0.11) | |
| Placebo at -6 | 0.06 (0.08) | |
| Placebo at +6 (above-side data only) | 0.02 (0.09) | |
| Placebo at +10 | 0.17 (0.11) | |

- **Covariate balance** (h = 5): log beds 0.072 (0.044), resident-to-bed ratio 0.001
  (0.006), CMI 0.020 (0.014), large urban -0.030 (0.022), RRC/SCH 0.029 (0.023).
  - Lagged Y jumps by 0.232 (0.067). Eligibility is persistent (D = D on report t-1 in 92%
    of rows), so lagged Y is itself treated.
  - That is why `lagged_y` is off by default. With it on, R^2 of Y on X rises from 0.56 to
    0.82.
- **Bunching.** In 0.5-point windows above : below, the pooled sample has 483 : 443
  (1.09), the DSH group 324 : 312 (1.04) and the RRC/SCH group 159 : 131 (1.21).

## Readiness (2026-10-29)

`adapter.readiness(s)`:

| Loader | n | Hospitals | p | Eligible | R^2 of each score on X | R^2 of Y on X |
|---|---:|---:|---:|---:|---|---:|
| `load()` | 25,831 | 2,564 | 33 | 15,745 | Q: 0.271 (SSI 0.255, Medicaid 0.224) | 0.561 |
| `load(lagged_y=True)` | 25,831 | 2,564 | 35 | 15,745 | Q: 0.276 | 0.822 |
| `load_two_score()` (DSH group) | 18,640 | 2,043 | 32 | 10,042 | SSI 0.274, Medicaid 0.249 | 0.528 |
| `load_rrc_sch()` | 7,191 | 988 | 32 | 5,703 | SSI 0.243, Medicaid 0.172 | 0.667 |

Score distribution in `load()`:
- DPP quantiles 5/25/50/75/95%: 12.8 / 21.7 / 28.6 / 37.6 / 62.3; Q quantiles: -14.2 /
  -4.6 / 3.0 / 11.8 / 35.9.
- SSI % 10/50/90: 2.9 / 6.7 / 15.3. Medicaid % 10/50/90: 11.4 / 21.5 / 37.2. corr = 0.51.
- Rows within 2 points of the cutoff: 2,423 (DSH) and 1,121 (RRC/SCH); within 5 points:
  5,887 and 2,710.
- About 2,150 rows per year.

## Loaders and extras

- `load(outcome="log_drug_op_chg", lead=2, years=(2011, 2022), lagged_y=False)`. Q = DPP
  minus the group cutoff; `treatment_rule` = 1{Q >= 0}; `extras["cutoff"] = 0`.
- `load_two_score(group="dsh", ...)`. Q = (SSI %, Medicaid %);
  `treatment_rule` = 1{Q1 + Q2 >= c}; `extras["weights"] = (1, 1)`; `extras["cutoff"]` =
  27.325 (DSH) or 22.765 (`group="rrc_sch"`, also `load_rrc_sch()`);
  `extras["cutoff_exact"]` = 27.3152 or 22.7697 (without the 0.01 grid).
- **Common extras.**
  - `unit_id` (CCN, for clustering), `year` (FY of the cost report).
  - `group`, `group_cutoffs`, `adj_cutoffs` (11.75, 8), `cutoff_dpp`.
  - The scores: `dpp`, `ssi_pct`, `mcd_pct`.
  - `adj_line33` (cost report), `adj_formula`, `eligible_published` (line-33 eligibility).
  - `control_type`, `rrc`, `sch`, `beds`.
  - Lags: `dpp_lag1`, `D_lag1`, `dpp_impact` (the Impact File FY t DPP, CMS's older-data
    projection), `y_lag1` and `y_lag1_names` (all outcomes on report t-1).
  - First stage: `remedy_cy`, `remedy_year`.
  - `fyb`, `fye`, `lead`, `outcome`.
- **Checks.** `rule_check()`, `rule_exceptions()`, `group_check()`, `local_jump(s, h, y=None)`
  and `readiness(s)`.

## Sample

- **Units.** One report per hospital and HCRIS file year t (the federal FY in which the cost
  reporting period begins). When a hospital files more than one report in a year (change of
  ownership, short periods), the longest is kept, then the latest processed.
- **Timing.** Report t covers roughly October t-1 to September t+1, depending on the
  hospital's fiscal year.
  - It must be filed within five months of period end. Eligibility based on it then runs
    until the next report is filed, so roughly calendar t+1 to t+2.
  - Y on report t+2 overlaps the end of that window. The window is mostly a continuation,
    since eligibility persists.
  - X comes from the IPPS Impact File for FY t, published in August of t-1 from older
    cost-report and claims data. No component of the report-t DSH computation is in X.

| Step | Reports |
|---|---:|
| short-term acute CCNs (xx0001-xx0879), FY 2011-2022 | 40,406 |
| not Maryland (not IPPS) or Puerto Rico (no SSI) | 39,236 |
| in the IPPS Impact File for FY t | 38,257 |
| nonprofit (1-2) or state/local government (8-13) | 28,342 |
| not a Pickle hospital (line 33 = 35%, or Impact File) | 28,243 |
| both fractions on line 30 and 31 | 26,510 |
| Y on report t+2 (508 have no report t+2; 171 report no outpatient drug charges) | 25,831 |

- **Who is excluded.**
  - Excluded by control type: 9,586 for-profit and 329 federal reports.
  - Not in the sample: critical access hospitals (no DSH test), children's and free-standing
    cancer hospitals (not in the Impact File; children's hospitals may use S-3 or an
    audit).
- **Mergers and CCN changes** are not tracked.

## Getting the data

    python -m experiments.datasets.medicare_dsh.download      # Impact Files (another folder)
    python -m experiments.datasets.hospital_340b.download     # 1.94 GB
    python -m experiments.datasets.hospital_340b.download --check

- **Access.** Plain HTTPS; no login, form or agreement (downloaded 2026-10-29).
- **Caches.** `read_hcris(y)` reads only the needed worksheet items from each zip, without
  extracting, into `data/processed/hcris_<y>.csv.gz` (about 10 s a year).
  - `build()` caches the panel at `data/processed/hcris_340b_panel.csv.gz` (version 1).
  - Delete the caches after changing the adapter.

| File | Source | Bytes | SHA-256 (first 16) |
|---|---|---:|---|
| `hcris/HOSP10FY2011.ZIP` ... `HOSP10FY2024.ZIP` (14 files) | https://downloads.cms.gov/FILES/HCRIS/HOSP10FY<y>.ZIP (CMS "Cost Reports by Fiscal Year"; files dated 2026-10-05) | 1,936,086,999 | per file in `download.py` |
| `cms_340b/addendum_aaa.zip` | https://www.cms.gov/files/zip/nfrm-opps-remedy-340b-acquired-drug-payment-addendum-aaa.zip (CMS-1793-F final rule page) | 832,455 | e074511310b55f1d |
| `cms_340b/addendum_bbb.zip` | https://www.cms.gov/files/zip/nfrm-opps-remedy-340b-acquired-drug-payment-addendum-bbb.zip | 27,321 | 6a37132d560d0f7d |
| `docs/hrsa_qualification_information.htm` | https://340bregistration.hrsa.gov/help/CoveredEntity/Registration/CE_Registration/Hospital/QualificationInformation.htm | 167,016 | 46413530b5cdbace |

**Not used or blocked.**
- **data.nber.org** (HCRIS and Impact File mirror). Its robots.txt is `User-agent: *` /
  `Disallow: /` (checked 2026-10-29), so the HCRIS files were taken from CMS instead.
  - `medicare_dsh`'s Impact Files, which this folder reads, were downloaded from that
    mirror by another session.
- **HRSA OPAIS covered-entity list** (340bopais.hrsa.gov, "Covered Entity Daily Export").
  - It is generated by a button in a Blazor (SignalR) web app; there is no file URL.
  - HRSA's FAQ says vendors fetch it with UI automation (e.g. Selenium), which the download rules do not
    allow.
  - The Internet Archive was offline when checked. The remedy file stands in for the first
    stage.
- **www.hrsa.gov** answers scripts with 403 (Akamai), including robots.txt.
- **CMS's separate SSI ratio files** were not needed: the SSI fraction is line 30 of each
  cost report, which is the value used for eligibility.

## Caveats

- **Eligibility, not participation.** D is the DSH test on report t.
  - Enrollment also depends on the unobserved ownership test, the GPO prohibition,
    registration timing, and status carried over from earlier reports.
  - The design is fuzzy: about 39% just below the cutoff billed 340B drugs two years
    later, against 73% just above.
- **Y is gross charges** at chargemaster prices. The drug cost-to-charge ratio does not jump,
  so the drug-charge jump reflects volume or mix, not pricing. Charges are not revenue.
- **Medicare payment cut.** From 2018 to September 2022 Medicare cut OPPS payment for
  340B-acquired drugs (ASP - 22.5%), which narrows the 340B margin on Medicare patients.
  The jump is the same before and after 2017.
- **HCRIS versions.** HCRIS holds the latest version of each report (as filed, settled or
  reopened). HRSA used the as-filed value at registration time.
  - CMS updates the zips, so a later download can change a few scores and the SHA-256.
- **Lagged outcomes are treated** (see "Outcome choice"). Use `lagged_y=True` only for
  prediction, not as a pre-treatment control.

## Suggested policy boxes

- **`load()`: cutoff shift.** c in [-6, +6] DPP points around the group cutoff,
  D = 1{Q >= c}.
  - For the DSH group this is an adjustment-percentage cutoff of about 6.9 to 16.7.
  - About 13,000 rows lie within 8 points.
- **`load_two_score()` (DSH group): reweighting the fractions.**
  D = 1{w Q1 + (2 - w) Q2 >= c}, with w in [0.5, 1.5] (SSI weight relative to Medicaid)
  and c in [22, 33].
  - Medicaid dominates DPP (median 21.5 against 6.7 for SSI), so tilting w moves
    eligibility mostly among high-SSI, low-Medicaid hospitals.
  - The deployed rule is w = 1, c = 27.325.
- **Group cutoff gap.** Hold the DSH cutoff at 27.325 and vary the RRC/SCH cutoff in
  [18, 28]; the deployed value is 22.765.

## Citation

- Desai S, McWilliams JM. Consequences of the 340B Drug Pricing Program. *N Engl J Med*
  2018. doi:10.1056/NEJMsa1706475. It is a regression discontinuity at the 11.75% DSH
  threshold among general acute care hospitals.
  - Eligibility was associated with 2.3 more hematologist-oncologists in hospital-owned
    practices.
  - Hospital parenteral drug claims were higher by 90% in hematology-oncology and by 177%
    in ophthalmology.
  - There was no clear evidence of expanded care or lower mortality for low-income
    patients.
  - Verified from the abstract only.
- 42 U.S.C. 256b(a)(4); 42 CFR 412.106(d); CMS-2552-10 cost report form; CMS-1793-F (OPPS
  remedy for 340B-acquired drug payment policy for CY 2018-2022, Addendum AAA).
