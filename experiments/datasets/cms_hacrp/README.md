# CMS Hospital-Acquired Condition Reduction Program: a 1% Medicare payment cut for the worst-scoring quartile of hospitals

| Field | Value |
|---|---|
| **Q** | `load()`: (Total HAC Score - c_t) / sd_t, where c_t is the cutoff CMS deployed in program year t and sd_t is the sd of that year's non-Maryland scores. `load_two_score()`: ((D1 - c_t)/sd_t, (D2 - c_t)/sd_t), with D1 = Domain 1 (PSI 90) and D2 = Domain 2 (mean of the infection measures). `extras["q_raw"]` = score - c_t |
| **Threshold** | 0. On the raw score, c_t is the 75th percentile of the year's non-Maryland Total HAC Scores (42 CFR 412.172(e)(2)), read off the published flags: 7.0125 (FY 2015), 6.8125, 6.5725, 0.3714, 0.3430, 0.3308, 0.3385, 0.2998 (FY 2022), 0.3753, 0.3663, 0.3796 (FY 2026). There is no cutoff in FY 2023 (no scores) |
| **Treatment** | `1{Q >= 0}`, penalized: a 1% cut in all Medicare FFS inpatient payments for the fiscal year. The treated side is the worse one (higher score). `load_two_score()`: `1{w1 Q1 + w2 Q2 >= 0}` |
| **Rule class** | single score with a rank-based cutoff (top quartile). In FY 2015-2019 the score is a fixed-weight sum of two domain scores: (0.35, 0.65), then (0.25, 0.75), then (0.15, 0.85) from FY 2017. From FY 2020 it is an equal-weight mean of up to six measure z-scores |
| **X** | log beds, log average daily census, log Medicare discharges, occupancy, resident-to-bed ratio, teaching and major-teaching indicators, DSH patient percentage, Medicare day share, case-mix index, urban / rural, SCH / MDH / RRC (from that FY's IPPS Impact File); ownership and emergency services (Hospital Compare general information); the previous program year's standardized score, penalty, PSI 90 and infection scores; state and FY indicators. 82 columns in `load()` (8 continuous characteristics, 3 lagged scores, the rest indicators). No same-year score components |
| **Y** | `load()`: the standardized Total HAC Score, (score - c)/sd, in the first program year whose measurement windows start after the FY t result was known: FY t+3, or FY 2024 for FY 2020. `load_psi90()`: that year's PSI 90 score, standardized within the year. `load_next_year*()`: the same outcomes in FY t+1, whose windows mostly come before the result |
| **n** | `load()`, `load_psi90()`: 14,657 and 14,377 hospital-years (FY 2018-2022, 3,098 hospitals, 3,649 treated). `load_all_years()`: 23,935 (FY 2015-2022). `load_next_year()`: 18,145. `load_two_score()`: 5,711 (FY 2018-2019) |
| **Source** | HACRP hospital files FY 2015-2026 and general information (Hospital Compare archives, NBER mirror data.nber.org/compare/hospital); CMS FY 2016 HAC Reduction Program Information and IPPS Impact Files FY 2015-2025 (cms.gov); IPPS final rules in the Federal Register; 42 CFR 412.170-412.172 (eCFR) |

## The rule

Section 1886(p) of the Social Security Act and 42 CFR 412.170-412.172 set the rule. CMS
scores every subsection (d) hospital. The hospitals in the top quartile of the Total HAC
Score are paid 99% of their Medicare inpatient amount for every discharge in the fiscal
year (412.172(b)). The regulation says that "CMS will identify the 25 percent of
hospitals with the highest total scores" (412.172(e)(2), in both eCFR versions in
`data/raw/rule/`). A higher score is worse.

| Program year (FY) | Scale | Domain 1 | Domain 2 (CDC NHSN infections) | Total HAC Score | Source |
|---|---|---|---|---|---|
| 2015 | points 1-10 by decile | PSI 90 | CLABSI, CAUTI | 0.35 D1 + 0.65 D2 | 78 FR 50722-50725 |
| 2016 | points | PSI 90 | + SSI (colon, hysterectomy) | 0.25 D1 + 0.75 D2 | 79 FR 50102-50104 |
| 2017 | points | PSI 90 | + MRSA, CDI | 0.15 D1 + 0.85 D2 | 80 FR 49575-49576 |
| 2018, 2019 | Winsorized z-scores (5th/95th percentiles) | PSI 90 | the five | 0.15 D1 + 0.85 D2 | 81 FR 57022-57025; 82 FR 38568; 83 FR 41758 |
| 2020-2022, 2024-2026 | Winsorized z-scores | no domains | | mean of the measures the hospital has | 83 FR 41486-41489 |
| 2023 | | all six measures suppressed (COVID-19) | | none, no payment cuts | 87 FR 48782, 49478 |

D2 is the mean of the hospital's infection scores. A hospital with only one domain gets
that domain's score as its total (78 FR 50725). A hospital with neither domain gets no
score and no cut.

**The cutoff.** The cutoff is the 75th percentile of the year's Total HAC Scores among
non-Maryland hospitals with a score. A hospital is penalized if its score is strictly
above it. The sources for this:
- CMS's FY 2016 file states it: "The 75th percentile of the Total HAC Score is 6.7500 ...
  Maryland hospitals' ... Total HAC Scores are not included in determining the Top 25th
  Percentile."
- The impact analyses describe "non-Maryland hospitals with a Total HAC Score above the
  75th percentile" (82 FR 38568, 83 FR 41758).
- Under decile scoring, ties at the threshold "prevented CMS from penalizing exactly 25
  percent of hospitals" (81 FR 57023). In FY 2015, 140 hospitals sit exactly at the 75th
  percentile (7.0), and in FY 2016, 134 sit at 6.75. CMS's file penalizes none of the
  FY 2016 ones. FY 2015 has no flag, so its ties are treated the same way.

**Groups**
- **Maryland.** Maryland hospitals are subsection (d) hospitals, and they are scored and
  published. Under the Maryland All-Payer Model, CMS waived section 1886(p) for them
  (79 FR 50089-50090). Their flag is N/A every year, and they are left out of the
  percentile and of the sample.
- **Not in the program** (78 FR 50709-50710):
  - Puerto Rico hospitals (not subsection (d) hospitals). None appears in any HACRP
    file.
  - Critical access hospitals.
  - Hospitals and units excluded from the IPPS: LTCHs, cancer, children's, IRFs and
    IPFs.

  Every sample hospital that matches the general information file is an "Acute Care
  Hospital". SCHs and Indian Health Service hospitals are in the program.
- **Too little data.** A hospital gets no PSI 90 score without 3 eligible discharges for a
  component. It gets no infection measure with fewer than 1 predicted infection. With
  neither, there is no Total HAC Score and no cut. Every year 28-157 such hospitals carry
  a "No" flag (`rule_check()`). Every unscored hospital in the files is unpenalized except
  one, in FY 2026 (see exceptions).

  No rule for new hospitals was found: they enter once they have data.
- **Not submitting infection data** without a waiver: the measure gets the worst score,
  per measure from FY 2017 on (79 FR 50098; 80 FR 49575).
  - Under decile points that is 10 points.
  - Under z-scores it is the maximum Winsorized z-score (as cited at 83 FR 41481, to 81
    FR 57013).
  - The files mark these hospitals with footnote 18, and in FY 2020 they all carry the
    year's maximum z.
  - The FY 2019 rule adds a validation penalty on the same lines for later years; it is
    not examined here.
- **COVID-19.**
  - Infection data from Q1-Q2 2020 are excluded (85 FR 54830).
  - The FY 2022 windows end in December 2019.
  - FY 2023 has no scores and no cuts.
  - FY 2024 uses shortened windows (table below).

## Rule check (2026-10-28)

`adapter.rule_check()`, non-Maryland hospitals with a Total HAC Score:

| FY | Scored | Penalized (published) | Flag source | Deployed cutoff interval (max unpenalized, min penalized] | 75th percentile of published scores | Rule `score > q75` vs flag (mismatches) |
|---|---:|---:|---|---|---:|---|
| 2015 | 3,284 | 721 (rule; no flag) | none | (7.0, 7.025] | 7.0 | - |
| 2016 | 3,209 | 751 | CMS FY 2016 file | (6.75, 6.875] | 6.75 | 1.000 (0) |
| 2017 | 3,203 | 769 | Compare | (6.57, 6.575] | 6.50 | 0.990 (32) |
| 2018 | 3,172 | 751 | Compare | (0.3712, 0.3716] | 0.3446 | 0.987 (42) |
| 2019 | 3,204 | 800 | Compare | (0.3429, 0.3431] | 0.3428 | 0.9997 (1) |
| 2020 | 3,149 | 786 | Compare | (0.3306, 0.3310] | 0.3305 | 0.9997 (1) |
| 2021 | 3,105 | 774 | Compare | (0.3383, 0.3386] | 0.3366 | 0.9994 (2) |
| 2022 | 3,060 | 764 | Compare | (0.2998, 0.2999] | 0.2994 | 0.9997 (1) |
| 2024 | 2,928 | 728 | Compare | (0.3751, 0.3756] | 0.3729 | 0.9986 (4) |
| 2025 | 2,922 | 724 | Compare | (0.3662, 0.3664] | 0.3636 | 0.9976 (7) |
| 2026 | 2,886 | 718 | Compare | (0.3792, 0.3799] | 0.3787 | 0.9986 (4) |

- **The published flags are an exact threshold on the published score in every year.**
  No unpenalized hospital scores above a penalized one. So D = 1{score > c_t}, with c_t
  the midpoint of the interval, agrees with the published flag for 100% of hospitals
  with a score. In FY 2016 the CMS-stated 75th percentile (6.75) reproduces all 751 flags.
- **Recomputing the score.** The Total HAC Score recomputed from the published domain
  scores (FY 2015-2019) or measure scores (FY 2020-) matches the published score within
  2e-4 for 100% of hospitals, except two in FY 2016 (99.94%). One is 370180, whose CMS
  score has no published components.
- **The cutoff is not exactly the 75th percentile of the published file.**
  - The textbook rule (75th percentile of the published non-Maryland scores) misses 1-7
    hospitals a year from FY 2019. It misses 32 in FY 2017 and 42 in FY 2018, where the
    deployed cutoff sits at the 76.0th and 76.3rd percentile of the published scores.
  - Likely cause: CMS fixes the percentile on the scores at the end of its review and
    correction period, and the hospital population there differs slightly from the one
    later posted on Hospital Compare. No public document gives the FY 2017-2026 cutoffs,
    so the deployed values are read off the flags. `extras["q75_raw"]` and
    `extras["penalized_q75_rule"]` keep the textbook version.
- **FY 2015 has no hospital-level flag in the mirror or on cms.gov.** D is the rule:
  score > 7.0, the 75th percentile, which gives 721 hospitals. The abstract of Sankaran
  et al. (2019) reports 724 penalized hospitals in FY 2015, so 3 hospitals are
  unaccounted for. FY 2015 is only in `load_all_years()` and `load_two_score((2015,))`.

**Exceptions**
1. **FY 2016, CMS file vs Hospital Compare (December 2015).** Seven hospitals with
   Compare scores of 8.25-9.25 are "N/A" (not scored, not penalized) in CMS's February
   2016 file: 110234, 170203, 260229, 440231, 440232, 670087 and 670089. All have Domain
   2 = 10.0. These are probably the hospitals whose penalty CMS waived (81 FR 57022),
   although the rule describes the waived hospitals as having no Domain 2 score.

   One hospital, 370180, is unscored on Compare but has 10.0 and is penalized in the CMS
   file. The adapter uses the CMS file for FY 2016.
2. **FY 2026.** 440218 (Saint Thomas Hospital for Spinal Surgery) is flagged "Yes" with no
   score or measures. It is out of the sample because it has no Q.

## Getting the data

    python -m experiments.datasets.cms_hacrp.download          # idempotent
    python -m experiments.datasets.cms_hacrp.download --check  # sizes, hashes

The 52 files total 101.6 MB and go to `data/raw/` (gitignored). They were downloaded on
2026-10-28 with a plain urllib request and an honest user agent; no source in the list
blocked it. `download.py` records each file's SHA-256.

| Group | Files | Bytes | URL |
|---|---:|---:|---|
| `hac` | 12 HACRP files + 3 NBER codebooks | 7,536,219 | `https://data.nber.org/compare/hospital/{yyyy}/{mm}/...`: 2014/12 `hospital_quarterly_hac_domain_hospital_11_24_2014.csv` (FY 2015); 2015/12, 2016/12, 2018/01, 2020/01 `hospital_quarterly_hac_domain_hospital.csv`; 2019/03 `hospital_quartery_hac_domain_hospital.csv` (sic); 2021/01, 2022/01, 2023/01, 2024/10, 2025/02, 2026/02 `fy_{FY}_hac_reduction_program_hospital.csv`; `desc_*.txt` for 2014/12, 2015/12, 2016/12 |
| `general` | 12 | 19,794,569 | `https://data.nber.org/compare/hospital/{2015/04, 2015/10, 2016/12, 2018/01, 2019/03, 2020/01, 2021/01, 2022/01, 2023/01, 2024/10, 2025/02, 2026/02}/hospital_general_information.csv` |
| `cms_hac` | 1 | 100,484 | https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads/fy2016-hac-reduction-program-information.zip |
| `impact` | 11 | 37,507,445 | https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads/ `fy2015-fr-impact-file.zip`, `fy2016-cms-1632-fr-impact.zip`, `fy2017-cms-1655-fr-impact.zip`, `fy2018-cms-1677-fr-impact.zip`, `fy2019-cms-1694-fr-impact.zip`, `fy2020-fr-impact-file.zip`; https://www.cms.gov/files/zip/ `impact-file-fy-2021-final-rule-and-correction-notice.zip`, `fy-2022-ipps-final-rule-and-ca-impact-file.zip`, `fy2023-ipps-fr-impact-file.zip`, `fy2024-ipps-fr-impact-file.zip`, `fy-2025-ipps-fr-impact-file.zip` (1.7-4.8 MB each) |
| `rule` | 9 Federal Register texts + 4 eCFR sections | 36,680,376 | `https://www.federalregister.gov/documents/full_text/text/{date}/{doc}.txt` for 2013-18956 (78 FR 50496), 2014-18545 (79 FR 49854), 2015-19049 (80 FR 49326), 2016-18476 (81 FR 56762), 2017-16434 (82 FR 37990), 2018-16766 (83 FR 41144), 2020-19150 (85 FR 54820), 2021-16519 (86 FR 44774), 2022-16472 (87 FR 48780); `https://www.ecfr.gov/api/versioner/v1/full/{2017-01-01, 2025-10-01}/title-42.xml?part=412&section={412.170, 412.172}` |

- **Snapshots.** Each FY's HACRP file is taken from the first NBER snapshot that carries
  it, the one with the most hospitals. Scores are identical across a year's snapshots:
  2014/12 vs 2015/04 and 2015/10, 2015/12 vs 2016/05, 2016/12 vs 2017/04, 2018/01 vs
  2018/10, 2019/03 vs 2019/10, 2020/01 vs 2020/10, 2026/02 vs 2026/08. Later snapshots
  only drop hospitals.
  - The NBER codebooks give the source Compare archives: `HOSArchive_20141218.zip`
    (FY 2015) and `HOSArchive_20151210.zip` (FY 2016).
  - FY 2024 has a single snapshot (2024/10).
- **Payment flags.** The FY 2015 and FY 2016 Compare files have no payment-reduction
  column. FY 2016's flag comes from the CMS file. For FY 2015 none was found: not in the
  mirror, and the FY 2015 rule-files page on cms.gov has no HAC file (guessed names
  return 404).
- **Reading.** The Impact Files are read in place from the zips: the tab-separated final
  rule ("FR") member, not the correction notice. The `.sas7bdat`, `.dta` and
  `hospitals_compare_*.zip` duplicates on the mirror are not downloaded.
- **Cache.** The adapter caches the hospital x program-year panel in
  `data/processed/cms_hacrp_panel.csv.gz` (3.1 MB, about 8 s to build). It writes a
  temporary file in the same folder and renames it. Delete the cache after changing
  `adapter.py`.

**Blocked or not used**
- **data.cms.gov** answered 403 to a script on 2026-10-28 (https://data.cms.gov/provider-data/).
  It is the current home of the HACRP and general-information files, so they come from
  the NBER mirror instead.
- **QualityNet** (https://qualitynet.cms.gov/inpatient/hac/methodology). It returns only a
  JavaScript shell to a plain request. It probably holds the FY-specific fact sheets with
  the official 75th-percentile values, which were not obtained.

## Sample and matches

Base: non-Maryland hospitals with a Total HAC Score, in the designation years.
- **Outcome observed.** At t+3, 94.3% of hospital-years have a score: 94.1% unpenalized,
  94.9% penalized. Within |Q| < 0.25 it is 95.9% vs 96.6%. At t+1 it is 98.0%.
- **Impact File match.** 99.5-99.9% of hospital-years match their FY's Impact File by CCN.
  Unmatched rows get medians and `impact_missing`.
- **General information match.** 98.7-100% by CCN.
- **Lags.** FY 2018 lags are FY 2017 decile-era scores, standardized by FY 2017's sd. FY
  2024's lag is FY 2022.

## Readiness (2026-10-28, computed as in `multi_rule_readiness`, not written to its summary)

OLS of each score on [1, X]; sd(eta) is the residual sd:

| Loader | n | Treated | R^2 | sd(Q), sd(eta) | Within 0.25 sd(eta) of the cutoff |
|---|---:|---:|---:|---:|---:|
| `load()` (FY 2018-2022) | 14,657 | 3,649 (0.249) | 0.506 | 0.99, 0.69 | 0.113 |
| `load_psi90()` | 14,377 | 3,578 (0.249) | 0.505 | 0.97, 0.69 | 0.112 |
| `load_all_years()` (FY 2015-2022) | 23,935 | 5,806 (0.243) | 0.415 | 0.99, 0.76 | 0.130 |
| `load_next_year()` | 18,145 | 4,452 (0.245) | 0.417 | 0.99, 0.75 | 0.120 |
| `load_two_score()`: Q1 (PSI 90) | 5,711 | 1,446 (0.253) | 0.229 | 1.50, 1.32 | 0.129 |
| `load_two_score()`: Q2 (infections) | 5,711 | | 0.486 | 1.09, 0.78 | 0.140 |

R^2 of Y on X is 0.12 for `load()` and 0.20 for `load_next_year()`.

**The two-score view.**
- corr(Q1, Q2) = 0.11.
- D2 is the value implied by the published total and D1, so w'Q reproduces the published
  total exactly. It differs from the published D2 by at most 1.1e-4 (rounding).
- 465 scored FY 2018-2019 hospitals have only one domain (405 PSI 90 only, 55 infections
  only) and are left out.
- `load_two_score(years)` takes other years with a single weight vector:
  - `(2017,)`: n = 2,998, points scale.
  - `(2017, 2018, 2019)`: n = 8,709.
  - `(2016,)`: weights 0.25/0.75, n = 2,679.
  - `(2015,)`: weights 0.35/0.65, n = 2,556.

  These are decile-scale years; see the caveats.

## Outcome and timing

The FY t result reaches hospitals around the start of FY t, in confidential reports
with a 30-day review (412.172(f)). The impact analyses note that the scores still
undergo that review after the final rule (82 FR 38568). The result is public on Compare
in December or January, and the cut applies to discharges from October 1 of t-1 to
September 30 of t. The windows of FY t+1
end between June of t-1 and December of t-1, so they come mostly before the result:

| FY | PSI 90 window | Infection window |
|---|---|---|
| 2018 | 2014-07-01 to 2015-09-30 | 2015-2016 |
| 2019 | 2015-10-01 to 2017-06-30 | 2016-2017 |
| 2020 | 2016-07-01 to 2018-06-30 | 2017-2018 |
| 2021 | 2017-07-01 to 2019-06-30 | 2018-2019 |
| 2022 | 2018-07-01 to 2019-12-31 | 2019 |
| 2024 | 2021-01-01 to 2022-06-30 | 2022 |
| 2025 | 2021-07-01 to 2023-06-30 | 2022-2023 |
| 2026 | 2022-07-01 to 2024-06-30 | 2023-2024 |

So `load()` takes Y from the first program year whose PSI 90 window starts on or after
July 1 of t-1 and whose infection window starts on or after January 1 of t. That is
FY t+3 (FY 2021, 2022, 2024, 2024, 2025 for FY 2018-2022). The FY 2015-2017 windows are
in `adapter.WINDOWS`.

Y > 0 means penalized again in the outcome year. That is the case for 36.3% of penalized
hospitals and 20.9% of unpenalized ones. At t+1 the shares are 65.4% and 11.7%.

**Crude local-linear RD on Y** (descriptive; uniform kernel, HC0 se):

| Outcome | Bandwidth 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| `load()` | -0.005 (0.069) | +0.010 (0.050) | -0.026 (0.037) |
| `load_psi90()` | -0.110 (0.090) | -0.058 (0.064) | -0.001 (0.046) |
| `load_next_year()` | +0.084 (0.046) | +0.027 (0.033) | +0.006 (0.025) |

There is no detectable effect of the penalty on later HAC scores.

## Literature (checked 2026-10-28)

- Sankaran, R., D. Sukul, U. Nuliyalu, B. Gulseren, T. A. Engler, E. Arntson, H.
  Zlotnick, J. B. Dimick, and A. M. Ryan (2019). "Changes in hospital safety following
  penalties in the US Hospital Acquired Condition Reduction Program: retrospective cohort
  study." *BMJ*, l4109 (published online 2019-07-03), doi:10.1136/bmj.l4109.
  - It uses an RD on the FY 2015 penalty.
  - Its abstract reports no significant change in hospital-acquired conditions, 30-day
    readmission or mortality after penalization.

The bibliographic details come from Crossref.

## Caveats

- **Discrete scores in FY 2015-2017.**
  - Decile points give 188 distinct scores in FY 2015, 62 in FY 2016 and 592 in FY 2017.
  - There are mass points at the cutoff: 140 and 134 hospitals sit at it in FY 2015 and
    2016.
  - Next to the cutoff, hospitals differ in their mix of PSI and infection points, so
    E[Y | Q] can jump there with no effect at all. Example: the FY t+1 PSI 90 outcome,
    whose windows mostly come before the result, jumps by +0.27 (se 0.07) at bandwidth
    0.5 in FY 2015-2017, against 0.00 (0.06) in FY 2018 on.
  - For this reason `load()` uses FY 2018-2022, and `load_all_years()` flags the decile
    years in `extras["decile_scale"]`.
- **Weak response.** The cut is real money: 1% of a hospital's Medicare inpatient
  payments for a year, repeated each year the hospital stays in the quartile. But
  neither this data (table above) nor Sankaran et al. find an effect on later safety
  scores. The outcomes here are noisy composite measures (R^2 on X of 0.12), so this
  dataset may give flat results like the school-level ones.

  An outcome with a mechanical first stage would be Medicare revenue or margin from cost
  reports (HCRIS; not built here).
- **Dynamic treatment.**
  - A hospital penalized in FY t is often penalized again: 65% at t+1 vs 12% of the
    unpenalized.
  - Successive windows overlap by a year, so Q_t and Q_{t+1} are strongly correlated.
  - The t+3 outcome follows the whole sequence of penalties since t. Inference should
    cluster by hospital (`extras["ccn"]`).
- **Rule changes.** The scale, measures and weights changed in FY 2016, 2017, 2018 and
  2020. Standardizing by the year's sd puts the years on one unit, but the meaning of a
  0.1 sd shift differs across scoring regimes. FY 2023 has no scores. FY 2022-2025 cover
  COVID-19, with shortened windows and Q1-Q2 2020 excluded.
- **The cutoff is a rank.** It moves with everyone's scores, and hospitals cannot see it
  in advance, so manipulation at the cutoff is implausible. But the 25% share is fixed by
  statute, so any alternative cutoff is an alternative penalized share.
- **Maryland** hospitals (about 45 a year) are out of the sample: scored, never cut.
- **Covariates.**
  - The Impact File for FY t is published in August of t-1, from cost reports 2-3 years
    old.
  - The lagged score shares a year of data with Q. It is predetermined (published a
    year earlier) but mechanically correlated with Q.
  - The hospital overall star rating and "safety of care" comparisons in the general
    information file are excluded, since they contain the HAC measures.

## Suggested policy box

- **`load()`, single score.** Cutoff shift δ in **[-0.5, 0.5]** sd of the year's score
  around the deployed cutoff. This penalizes 45% to 12% of hospitals, against 24.9% now.
  - D changes for 4,824 hospital-years (33%).
  - With [-0.25, 0.25] (34% to 18% penalized), D changes for 2,375.
- **`load_two_score()`, weighted sum.** PSI 90 weight w1 in **[0.05, 0.35]** (w2 = 1 -
  w1) and δ in **[-0.25, 0.25]**.
  - The range covers every weight CMS used: 0.35 in FY 2015, 0.25 in FY 2016, 0.15 in FY
    2017-2019. It also covers roughly 1/6, the PSI weight under FY 2020's equal weights
    for a hospital with all six measures.
  - D changes somewhere in the box for 1,857 of 5,711 hospitals (33%).
  - At the deployed δ = 0:
    - w1 = 0.05, 0.25 and 0.35 change D for 318, 311 and 597 hospitals.
    - w1 = 1/6 changes it for only 47.
