# Medicare indirect medical education (IME) adjustment on the resident-to-bed ratio

| Field | Value |
|---|---|
| **Q** | Resident-to-bed ratio r (`resbed`) in the IPPS Impact File for FY t, x100: residents per 100 beds. 0 for non-teaching hospitals |
| **Threshold** | 0 (teaching status); no cutoff above it |
| **Treatment** | Dose A = operating IME adjustment in percent, 100 c [(1 + r)^0.405 - 1] |
| **Rule class** | Concave curve on one score from 0. Four multipliers: c = 1.42 (FY 2005), 1.37 (2006), 1.32 (2007), 1.35 (FY 2008 on) |
| **X** | log beds, occupancy, case-mix index, DPP, Medicare share of days, log Medicare cases (FY t-2 MedPAR), wage index, operating CCR, rural, large urban, RRC, SCH, MDH, FY and census-region indicators (`geo="state"`: state indicators) |
| **Y** | `log_cases` (default): log Medicare cases in FY t+1 (the FY t+3 Impact File). `cmi_lead`: transfer-adjusted case-mix index in the FY t+3 file |
| **n** | 61,997 hospital-years (3,776 hospitals), FY 2005-2024; 20,628 teaching hospital-years (1,471 hospitals) |
| **Source** | CMS IPPS Impact Files, read through `medicare_dsh` (NBER copies FY 2005-2024, cms.gov FY 2025-2027); section 1886(d)(5)(B)(ii) of the Act; 42 CFR 412.105(d)(3); 69 FR 49088; 87 FR 49065 |

**Fit: does not stick.** The rule is exact and two thirds of hospital-years are at the
reference dose. But:
- **The reference is a mass point at r = 0** (57-72% of hospitals a year). For those hospitals
  eta = -T(X) exactly (corr -1.00), so their latent part is a function of X, not a latent score.
- **X predicts r poorly:** R^2 0.36 overall, 0.39 among teaching hospitals. So the dose tracks
  eta. Median eta rises from -7.8 in the lowest dose decile to +44.3 in the highest.
- **Support is narrow.** Coverage is at least 80% only for doses up to 5-10% in score bins below
  25 residents per 100 beds.
- **The supported estimates are noisy** (rotation sd as large as the estimate) and explode
  beyond that region.
- **The second schedule is weak.** The changes in the multiplier (FY 2005-2008) make the
  modifier test barely identified (r2_absorbed 0.994).

## The rule

The operating IME adjustment factor is c x [(1 + r)^0.405 - 1].
- r is the ratio of interns and residents (FTE) to available beds, subject to the resident caps
  of 412.105.
- **Multiplier c.** Section 502(a) of Pub. L. 108-173 (MMA) set c as follows (FY 2005 IPPS final
  rule, 69 FR 49088, text in `data/raw/docs/fr_04-17943_fy2005.txt`):
  - 1.47 for 1 Apr-30 Sep 2004;
  - 1.42 in FY 2005, 1.37 in FY 2006 and 1.32 in FY 2007;
  - "1.35 for discharges occurring during FY 2008 and fiscal years thereafter".
- **Still 1.35 recently.** The FY 2023 rule (87 FR 49065) repeats 1.35 under section
  1886(d)(5)(B)(ii)(XII) and estimates "an increase in IPPS payment of 5.5 percent for every
  approximately 10 percent increase in the hospital's resident-to-bed ratio".
- **The Impact File** lists r (`resbed`, "Used to determine IME factor for operating PPS
  payments") and the factor (`tchop`, "IME adjustment factor for Operating PPS"). The
  multiplier implied by tchop / [(1 + r)^0.405 - 1] is 1.42, 1.37, 1.32 and 1.35 by year.

## Rule check (2026-11-12)

`adapter.rule_check()`: |tchop - c[(1 + r)^0.405 - 1]| < 0.0001, all IPPS hospitals in the
50 states and DC (Maryland and Puerto Rico excluded).

| FY | c | n | Match | Share r = 0 | Median r (teaching) |
|---|---:|---:|---:|---:|---:|
| 2005 | 1.42 | 3,884 | 1.0000 | 0.717 | 0.091 |
| 2006 | 1.37 | 3,726 | 1.0000 | 0.713 | 0.091 |
| 2007 | 1.32 | 3,578 | 1.0000 | 0.701 | 0.092 |
| 2008 | 1.35 | 3,516 | 1.0000 | 0.704 | 0.096 |
| 2009-2016 | 1.35 | 3,520 → 3,350 | 1.0000 | 0.692-0.707 | 0.095-0.105 |
| 2017-2024 | 1.35 | 3,311 → 3,104 | 1.0000 | 0.607-0.682 | 0.105-0.121 |
| 2025-2027 | 1.35 | 3,055 → 2,979 | 1.0000 | 0.565-0.595 | 0.130-0.132 |

- **Agreement.** Every hospital-year matches. In the sample, the formula dose and the published
  dose differ by at most 0.0005 percentage points (rounding).
- **Top-coded ratios.** From FY 2020, one or two hospitals a year have r = 1.9999. It looks like
  a top code; they are kept.

## Screen

| Quantity | Value |
|---|---|
| n, hospitals, p | 61,997, 3,776, 40 |
| Share at the reference (r = 0, A = 0) | 0.667 (teaching status changes in 2.6% of hospital-years) |
| R^2 of Q on X, all hospitals | 0.360 (state indicators: 0.383) |
| R^2 of Q on X, fitted among teaching hospitals | 0.387 (state indicators: 0.424) |
| sd(Q), sd(T), sd(eta) | 16.4, 9.8, 13.1 residents per 100 beds |
| Teaching hospitals: Q quantiles 10/50/90 | 1.0 / 10.3 / 56.2 |
| Teaching hospitals: A quantiles 10/50/90/98 | 0.55 / 5.45 / 26.8 / 38.4 % |

eta quantiles (10/50/90) at the reference and by dose decile:

| Group | n | eta q10 | q50 | q90 |
|---|---:|---:|---:|---:|
| r = 0 | 41,369 | -11.9 | -2.5 | 6.1 |
| A 0.01-0.55 | 2,069 | -16.2 | -7.8 | -0.5 |
| A 2.64-3.89 | 2,067 | -12.6 | -4.2 | 3.5 |
| A 5.45-8.26 | 2,069 | -9.2 | 0.4 | 10.2 |
| A 12.0-17.3 | 2,064 | 0.2 | 12.3 | 23.0 |
| A 26.8-75.7 | 2,063 | 29.2 | 44.3 | 68.1 |

**Does the mass at r = 0 break the design?** Partly.
- **eta is not latent there.** Every non-teaching hospital has eta = -T(X); within that group
  corr(eta, T) = -1.00. Two non-teaching hospitals with the same X have the same eta, however
  far each is from starting a residency program.
- **eta's support depends on X.** Q is censored at 0, so the OLS first stage gives an eta
  whose support depends on X (eta >= -T).
- **The model still runs** because the reference eta range (-11.9 to 6.1 at 10-90%) overlaps the
  eta of low-dose teaching hospitals. Coverage is 86-97% for a <= 10% in the r = 0 and 5-10
  bins.
- **But the transported value means something else.** For a non-teaching hospital it is a
  function of X alone, not of a latent teaching propensity.

## Estimator (2026-11-12)

Runs:
- Command: `app.run("medicare_ime", ...)` with `dose_config.config`.
- Logs: `experiments/runs/dose_schedule/medicare_ime.log`, `medicare_ime_cmi_lead.log`,
  `medicare_ime_teaching.log`.
- Settings: seed 0, clustered by hospital, dose basis l=3 on [0, 38.4], K=7, k=4.

Effect on log Medicare cases in FY t+1 against A = 0, by score bin (residents per 100 beds):
estimate (rotation sd, coverage). Y has mean 7.35 and sd 1.30.

| Score bin | a=2 | a=5 | a=10 | a=15 | a=25 | a=35 |
|---|---|---|---|---|---|---|
| r = 0 | -0.025 (0.017, 95%) | -0.058 (0.054, 95%) | -0.078 (0.061, 86%) | 4.8 (4.5, 63%) | 94 (68, 6%) | 102 (308, 0%) |
| 0.5-5 | -0.034 (0.021, 94%) | -0.065 (0.057, 92%) | 0.006 (0.146, 71%) | 10.2 (7.5, 36%) | 190 (128, 1%) | 193 (542, 0%) |
| 5-10 | -0.025 (0.017, 97%) | -0.058 (0.054, 97%) | -0.085 (0.065, 86%) | 3.4 (3.1, 60%) | 65 (44, 3%) | 55 (298, 0%) |
| 10-25 | -0.052 (0.057, 80%) | -0.088 (0.081, 84%) | -0.094 (0.070, 94%) | 0.91 (0.90, 82%) | 22 (15, 26%) | 44 (133, 0%) |
| 25-50 | -0.96 (2.66, 28%) | -1.15 (3.32, 33%) | -0.16 (0.81, 80%) | 0.17 (0.53, 92%) | 0.14 (0.92, 78%) | 0.88 (8.7, 7%) |
| 50-200 | 3852 (5601, 1%) | 5098 (7404, 1%) | 1917 (2763, 9%) | 195 (264, 22%) | 6.1 (13.8, 67%) | 6.1 (13.7, 70%) |

- **Modifier test.** Coefficient -0.0006 (se 0.0002), r2_absorbed 0.9936 (n_fit 40,962). It
  rejects at conventional levels, but the column is 99.4% explained by the rest of the design.
- **`cmi_lead`** (mean 1.572, sd 0.392). In bins below 25:
  - a = 2: +0.03 to +0.09 (rotation sd 0.04-0.09);
  - a = 5: +0.07 to +0.16 (0.09-0.12);
  - a = 10: +0.19 to +0.28 (0.15-0.17);
  - beyond that it explodes as above. Modifier 0.0003 (se 0.0003), r2_absorbed 0.9936.
- **Teaching hospitals only** (`ime_teaching_only=True`): reference 5.5% (the median dose),
  doses below it fitted, n 20,628. Effects against 5.5% in bins 0.5-25:
  - a = 2: -0.10 to -1.5;
  - a = 10: +0.16 to +0.28 (rotation sd 0.13-0.28);
  - a = 15: +0.38 to +0.41 (0.26-0.30).
  - Higher bins and doses explode. Modifier -0.0001 (se 0.0002), r2_absorbed 0.998.

## Caveats

- **The dose is a price.** IME raises the DRG payment per Medicare case; Medicare Advantage
  cases of teaching hospitals also earn IME. Volume responses are slow, since r is set by
  residency programs and capped FTE counts.
- **r is almost fixed within a hospital.** The panel is mostly between-hospital variation.
  Teaching status changes in 2.6% of hospital-years.
- **r is the ratio CMS models in the Impact File,** not the settled cost-report ratio.
- **The score is a choice of the hospital** (number of residents, beds), within caps.

## Getting the data

- **Impact Files.** No new data download: the panel is built from the Impact Files already in
  `experiments/datasets/medicare_dsh/data/raw` (`python -m experiments.datasets.medicare_dsh.download`).
  - FY 2005-2024 are NBER copies downloaded by that adapter on 2026-10-28; nothing new was
    fetched from data.nber.org.
  - FY 2025-2027 come from cms.gov.
- **Rule text.** `python -m experiments.datasets.medicare_ime.download` fetches it into
  `data/raw/docs/` from https://www.federalregister.gov/documents/full_text/text/2004/08/11/04-17943.txt
  and .../2022/08/10/2022-16472.txt.
- **robots.txt.** www.federalregister.gov allows /documents/full_text (copy in
  `data/raw/robots/`). www.ssa.gov answered "Access Denied" to a scripted request for its
  robots.txt, so the statute itself (https://www.ssa.gov/OP_Home/ssact/title18/1886.htm) was not
  fetched.
- **Cache.** `data/processed/ime_panel.csv.gz` (version 1).

## Not verified

- The statute text of section 1886(d)(5)(B)(ii) and the eCFR text of 412.105(d)(3) were not
  read. The multipliers come from the Federal Register and agree with the Impact Files.
- Why r = 1.9999 appears from FY 2020 was not checked.
