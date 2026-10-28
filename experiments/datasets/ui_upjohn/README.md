# UI weekly benefit schedules: Washington 1988 and Illinois 1984 bonus experiments

| Field | Value |
|---|---|
| **Q** | `s / s_cap(group)`, where `s` = average of the claimant's two highest base-period quarters of covered earnings (dollars; `extras["score_dollars"]`) and `s_cap` = the score at which the weekly benefit reaches its cap |
| **Threshold** | 1.0 in every group (`s_cap` = $5,125 WA before 3 Jul 1988, $5,225 WA after, $4,335.50 IL) |
| **Treatment** | Dose `A` = weekly benefit set by the formula, relative to the group's cap (`a_ref = 1`). The dose varies for `Q < 1` and is flat at the cap above it; `D = 1{Q < 1}` |
| **Rule class** | Kinked linear schedule: rate x score between a floor and a cap, in whole dollars, with group-specific floor and cap (benefit-year half in WA, dependents' allowance in IL). A randomized bonus is a second dose at a fixed score |
| **X** | Age, sex, race, log lag-quarter earnings (+ zero / missing flags), IL and IL-dependents indicators, claim month; WA only: years of education, log earnings in the 4 quarters before the base year, number of those quarters with earnings, two dislocation flags, 1-digit industry (10), 1-digit occupation (10), Job Service office (20). 71 columns |
| **Y** | `load_dose(outcome=...)`: `weeks` (default; weeks with UI paid in the benefit year), `benefits` (dollars paid), `spell` (weeks in the first spell), `earn_post` (mean covered earnings in the three quarters after the claim quarter) |
| **n** | 27,590 (WA 15,489 in the analytic sample, IL 12,101) |
| **Source** | W.E. Upjohn Institute Employment Research Data Center public-use files: Washington Reemployment Bonus Experiment, Illinois Unemployment Incentive Experiments, Pennsylvania Reemployment Bonus Demonstration |

**Fit: good on the rule, moderate on X.**
- **The rule is clean.** The formula reproduces the published WBA for 94.7% of WA and
  99.5% of IL claimants (99.5% for IL's total with the dependents' allowance). The score
  is employer-reported earnings, and there is no bunching at any kink.
- **The reference region is large.** 34.6% of claimants are at the cap (WA 35.0%, IL
  34.1%).
- **X predicts Q partly.** R^2 of Q on X is 0.365 (0.498 for log Q). Claimants with the same
  dose differ in eta: within every dose decile the 10-90% range of eta is about 0.5-0.9
  (in units of the cap kink).
- **Weak points.**
  - IL's X is thin: demographics, the lag quarter and the claim month only.
  - Dollars are nominal and the two states are four years apart.
  - Pennsylvania cannot be used for the dose: its score, high-quarter wages, is not in
    the public file.

## Dose design

`load_dose(states=("WA","IL"), outcome="weeks", ref="cap", dose="relative", asc_only=True)`

| Group | Rule (`s` in dollars) | Floor kink | Cap kink | Floor / cap ($/wk) |
|---|---|---:|---:|---:|
| `WA_88H1` (benefit year starts before 3 Jul 1988) | `clip(floor(s/25), 55, 205)` | 1,400 | 5,125 | 55 / 205 |
| `WA_88H2` (from 3 Jul 1988) | `clip(floor(s/25), 57, 209)` | 1,450 | 5,225 | 57 / 209 |
| `IL_nodep` | `AWW = round(s/13)`; `clip(ceil(0.48 AWW), 51, 161)` | 1,384.5 | 4,335.5 | 51 / 161 |
| `IL_dep` (dependents' allowance > 0) | above + `min(ceil(0.144 AWW), 48)` | none | 4,335.5 | 51+ / 209 |

- **`extras`.**
  - `A`: the formula dose.
  - `a_ref`: 1.0, or with `dose="dollars"` a per-unit array of the group caps.
  - `dose_levels`: None, since the dose moves in whole-dollar steps.
  - `schedule(q, group)` reproduces `A` exactly.
  - `group`, `state`.
  - `Z_names`: age, male, black, hispanic, log_lag_earn, educ, log_prebase_earn. The
    last two are WA-only and zero for IL.
  - `published_dose` (same units as `A`), `published_dose_dollars`, `dose_dollars`,
    `dose_match`, `score_dollars`, `kinks`.
- **Second dose: the bonus.**
  - `bonus_arm`: `WA_C`, `WA_T1`..`WA_T6`, `IL_C`, `IL_JSIE`, `IL_HIE`.
  - `bonus`: dollars offered to the claimant; 0 for controls and for IL HIE.
  - `bonus_mult` (WA), `employer_bonus` (IL HIE, $500), `qual_period`.
  - The bonus is randomized at a fixed score. In WA it is a multiple of the WBA, so it
    also kinks at the cap.
- **Third schedule (WA).** Potential duration = `min(30, base-year wages / (3 WBA))`
  weeks, from MBP = min(30 x WBA, base-year wages / 3).
  - It matches the published MBP for 92% (end-of-benefit-year values).
  - 49% of WA claimants have fewer than 30 weeks. It is in `extras["potential_weeks"]`.
  - IL's potential duration is a uniform 26 weeks.
- **`ref="floor"`.** Q is scaled by the floor kink and the dose varies for `Q >= 1`.
  - Only 5.1% are at the floor (WA 3.0-3.3%, IL 9.8%).
  - `IL_dep` is dropped, because its total keeps rising below the floor.
  - The dose is flat again above the cap.
- **Other extras.** `bw_consistent`, `outcomes` (all four), `potential_weeks`, `id`,
  `claim_date`.

## The rules

**Washington 1988.**
- **Formula.** The WREB final report (O'Leary, Spiegelman and Kline 1993, sec. 2.2.1,
  citing RCW 50.20.120) gives WBA = 1/25 x (Q1 + Q2)/2. Q1 and Q2 are the two highest
  quarters of the base year, the first four of the last five completed quarters before the
  benefit year. The WBA is bounded by a state MIN and MAX.
- **Read from the data.**
  - Truncation to the dollar (floor).
  - MIN/MAX = 55/205 for benefit years starting through 26 Jun 1988, and 57/209 from 3
    Jul 1988. Both pairs are consistent with 15% / 55% of a state average weekly wage of
    about $373 / $380 (inferred, not from a document).
- **Missing quarters.** A missing quarterly wage record (99999999.99) means no covered
  wages. Treated as 0, the base-year sum equals the published base-year wages for 95.4%.
- **Alternative base year.** The report quotes RCW 50.04.020: the last four completed
  quarters are used if the standard base year does not establish a claim. It is not
  modelled.
- **Bonus.** Arms T1/T2/T3 offer 2/4/6 x WBA with a short qualification period (20% of
  entitled duration + 1 week). T4/T5/T6 offer the same multiples with a long one (40%
  + 1). The bonus is based on the WBA at filing.

**Illinois 1984.**
- **Formula.** The final report (Spiegelman and Woodbury 1987, ch. 6 note 12) gives the
  WBA as 48% of the average weekly wage in the two highest base-period quarters, with
  maximum $161 and minimum $51.
- **Read from the data.**
  - AWW is (Q1 + Q2)/26 rounded to the nearest dollar.
  - WBA = 0.48 AWW rounded up to the next dollar.
  - The dependents' allowance is ceil(0.144 AWW), capped at $48, so the total is at
    most $209.
  - The report's own example ("$8,721 in any two quarters" for the maximum) is about
    $50 above the data-implied kink of $8,671. This README follows the data, which
    reproduce 99.5%.
- **Bonus.** JSIE offers the claimant $500 for a job within 11 weeks held 4 months. HIE
  offers $500 to the hiring employer.

**Pennsylvania 1988-89 (not in `load_dose`).**
- **Formula.** The WBA comes from the statutory table on high-quarter wages, plus $5/$3
  dependents' allowances that are not included in `wba`. The maximum is $252 (1988) and
  $266 (1989); the data show 266, and the report's footnote OCRs as $268.
- **No score.** The public records file has only total base-year wages, so the rule
  cannot be reproduced. `pa_frame()` returns the cleaned table, and `rule_check()`
  checks what it can.
- **Bonus.** Arms T1/T2 offer 3 x WBA (6/12-week qualification). T3/T4/T6 offer 6 x
  WBA, and T5 starts at 6 x WBA and declines. T6 has no workshop offer.

## Rule check (2026-10-28)

`adapter.rule_check()`:

| Check | Group | n | Match | Within $1 |
|---|---|---:|---:|---:|
| WA WBA at filing (all valid claims) | WA_88H1 | 7,178 | 0.947 | 0.950 |
| | WA_88H2 | 10,308 | 0.947 | 0.950 |
| WA WBA, analytic sample (ASC = 1) | WA_88H1 / H2 | 6,353 / 9,136 | 0.947 / 0.947 | 0.950 / 0.950 |
| WA bonus = multiple x WBA | treated | 14,031 | 0.995 | |
| IL WBA (base) | IL_nodep / IL_dep | 6,429 / 5,672 | 0.995 / 0.995 | 1.000 / 0.999 |
| IL WBA + dependents' allowance | IL_dep | 5,672 | 0.995 | 0.999 |
| PA WBA <= yearly max | all | 17,513 | 1.000 | |
| PA bonus = multiple x WBA | treated | 11,410 | 1.000 | |
| PA MBA = 26 (or 16) x WBA | all | 17,513 | 1.000 | |

- **WA mismatches (5.3%).**
  - They are spread evenly over offices, months and score levels.
  - In 84% of them the published WBA is below the formula. In 27% the end-of-year
    WBA equals the formula.
  - The likely cause is that the wage file postdates the monetary determination: late
    or corrected wage reports, and possibly the alternative base year.
  - `A` is the formula dose; the published one is in `extras["published_dose"]`.
- **WA share at the cap.** 33% of all WA claimants are at the cap, and 35% of the
  analytic sample.

## Sample and outcomes

- **WA.**
  - All 17,554 claimants enrolled March-November 1988 at 21 Job Service Centers.
  - Kept: claims with a WBA at filing > 0 and a 1988 benefit year (17,486), and by
    default (`asc_only=True`) the report's analytic sample, ASC = 1 (15,489).
  - ASC means the claim was monetarily valid with no nonmonetary issue in the
    qualification period.
- **IL.** All 12,101 claimants (ages 20-54, filed 29 Jul-17 Nov 1984) in the public file.
- **Outcomes.**
  - `weeks` for WA is `totalwks`, weeks with compensation in the 52-week benefit year,
    including the waiting week. For IL it is `wkspdbye`.
  - Means: WA 14.95, IL 19.54.
  - `benefits` scales with the dose mechanically; prefer `weeks` or `spell` for
    behaviour.
  - `earn_post` is missing for 239 IL claimants with no wage-record match, and is
    heavy-tailed.
- **Covariates and the score.** No X column enters the formula.
  - The lag quarter (the quarter just before the claim) and the four quarters before the
    base year are outside the base period.
  - Only a WA claimant on the alternative base year would have the lag quarter in the
    formula.

## Readiness (2026-10-28)

`python -m experiments.datasets.ui_upjohn.readiness` (eta = Q - OLS fit of Q on X):

| Sample | n | p | R^2(Q on X) | R^2(log Q) | sd(Q) | sd(eta) | At reference (cap) |
|---|---:|---:|---:|---:|---:|---:|---:|
| WA + IL | 27,590 | 71 | 0.365 | 0.498 | 0.672 | 0.535 | 0.346 |
| WA | 15,489 | 63 | 0.356 | 0.562 | 0.720 | 0.578 | 0.350 |
| IL | 12,101 | 13 | 0.402 | 0.458 | 0.603 | 0.466 | 0.341 |

By group (pooled X): R^2 / sd(eta) / share at cap:
- WA_88H1: 0.370 / 0.566 / 0.349
- WA_88H2: 0.359 / 0.580 / 0.350
- IL_nodep: 0.392 / 0.408 / 0.296
- IL_dep: 0.405 / 0.520 / 0.391

eta quantiles by dose decile (pooled, relative dose; the reference level is its own row):

| Dose bin | n | A range | eta 10% | eta 50% | eta 90% |
|---|---:|---|---:|---:|---:|
| reference (cap) | 9,539 | 1.000 | -0.183 | 0.167 | 0.822 |
| decile 1 | 1,281 | 0.244-0.316 | -0.642 | -0.216 | 0.275 |
| decile 2 | 2,276 | 0.317-0.391 | -0.572 | -0.257 | 0.163 |
| decile 3 | 1,818 | 0.392-0.463 | -0.508 | -0.224 | 0.172 |
| decile 4 | 1,815 | 0.464-0.528 | -0.486 | -0.210 | 0.143 |
| decile 5 | 1,813 | 0.531-0.595 | -0.487 | -0.201 | 0.111 |
| decile 6 | 1,808 | 0.596-0.660 | -0.445 | -0.190 | 0.114 |
| decile 7 | 1,794 | 0.663-0.732 | -0.445 | -0.174 | 0.103 |
| decile 8 | 1,798 | 0.733-0.810 | -0.412 | -0.148 | 0.145 |
| decile 9 | 1,825 | 0.813-0.898 | -0.383 | -0.111 | 0.157 |
| decile 10 | 1,823 | 0.900-0.995 | -0.371 | -0.070 | 0.188 |

Units with the same dose span roughly 0.5-0.9 in eta at every dose level. The lowest
deciles hold the IL floor ($51 = 0.244-0.317 of the cap).

**Bunching.** Counts of `s` in [0.98k, k) and [k, 1.02k) at each kink k, against the same
ratio at placebo points 0.80-1.20 x k:

| Group | Kink | At ($) | Below | Above | Ratio | Placebo range |
|---|---|---:|---:|---:|---:|---|
| WA_88H1 | floor | 1,400 | 23 | 18 | 0.78 | 0.48-1.22 |
| WA_88H1 | cap | 5,125 | 85 | 69 | 0.81 | 0.83-1.22 |
| WA_88H2 | floor | 1,450 | 23 | 31 | 1.35 | 0.53-1.61 |
| WA_88H2 | cap | 5,225 | 129 | 106 | 0.82 | 0.69-1.06 |
| IL_nodep | floor | 1,384.5 | 34 | 40 | 1.18 | 0.81-1.50 |
| IL_nodep | cap | 4,335.5 | 76 | 83 | 1.09 | 0.77-1.15 |
| IL_dep | cap | 4,335.5 | 61 | 73 | 1.20 | 0.80-1.26 |

- No excess mass at any kink. Every ratio is within or next to its placebo range, and
  the score is employer-reported.
- The windows are small (20-130 claimants), so this rules out only large bunching.

## Getting the data

    python -m experiments.datasets.ui_upjohn.download            # WA, PA, IL (~117 MB)
    python -m experiments.datasets.ui_upjohn.download --only jsa spells   # screened extras
    python -m experiments.datasets.ui_upjohn.download --check

- **Source.** Direct links on the ERDC pages,
  https://www.upjohn.org/data-tools/employment-research-data-center/<page>.
- **Access.** No login, form, agreement or CAPTCHA (checked 2026-10-28).
- **Extraction.** Zips are extracted into `data/raw/<zip stem>/`. The Acrobat Reader
  installers (`install/rp500enu.exe`) that ship in the original CD images are skipped.
- **Size.** 436 MB on disk with the extracted copies.
- **Report text.** The reports and codebooks are scanned images. Their text was read
  by OCR (macOS Vision) outside the repo.

| File | ERDC page | Bytes | Use |
|---|---|---:|---|
| `wa_reemp_bonus.zip` | washington-reemployment-bonus-experiment | 40,264,488 | `data/wreb.dta` (17,554 x 236), codebook, final report |
| `Reemployment_Incentives_..._.pdf` | same | 171,668 | executive summary |
| `PA_ReempBonus.zip` | pennsylvania-reemployment-bonus-demonstration | 38,555,725 | `Recsfile.dta` (17,513 x 189), `Survfile.dta` (5,678 x 641), documentation, final report |
| `prbdreport.pdf`, `prbdexecsumm.pdf` | same | 14,706,355 / 357,054 | final report, summary |
| `il_ui_experiment_data-images-install.zip` | illinois-unemployment-incentive-experiments | 12,277,110 | `data/illinois.dta` (12,101 x 49) |
| `il_ui_experiment_reports.zip`, `iuiieexecsumm.pdf` | same | 10,327,903 / 256,632 | final report, data documentation |
| `job_search_assistance.zip` (`--only jsa`) | assisting-unemployment-insurance-claimants-... | 25,241,165 | screened, not used |
| `ui_spells.zip` (`--only spells`) | analysis-ui-recipients-and-unemployment-spells | 17,647,609 | screened, not used |

## Other UI files screened

- **Job Search Assistance Demonstration (FL 17,982, DC 12,038; 1995-96).**
  - The files have the WBA (`wkbnamt1`, FL max $250, 27% at max), quarterly earnings
    from 1993:3, the benefit-year start, and rich X.
  - The FL WBA does not follow 1/26 of the high quarter from these earnings. WBA/HQ has
    IQR 0.033-0.037 against 0.0385, and only 9% are within 1% of the median ratio, for
    any base-period alignment. The public earnings seem adjusted.
  - Not built.
- **UI Spells (CWBH extract, Missouri and Pennsylvania 1979-80, 3,038).**
  - The file has CWBH high-quarter earnings, the basic and augmented WBA, and
    dependents.
  - PA WBA/HQ is about 0.041 (70% within 1% of the median). MO has two caps ($85, $105)
    and a wide ratio.
  - Small, with -5 / 0 missing codes. Not built.
- **Study of UI Exhaustees (25 states, 1998).** The administrative file has the WBA but
  no earnings. Not downloaded.
- **CWBH, five states (Landais 2015).** The replication package, openICPSR 114581
  (https://www.openicpsr.org/openicpsr/project/114581/version/V1/view), holds the
  microdata: `Prog_AEJ_RKD/Data/rawdata/cwbh_{ID,LA,MO,NM,WA}.dta` and `WAwag.dta`,
  about 2.3 GB. Downloading needs an openICPSR login and acceptance of its terms. No
  copy without a login was found (ERDC, NBER, the author's page).

## Caveats

- **Partial claimants** (filing while employed) are in the WA sample. Their weekly
  payment is below the WBA in some weeks.
- **The bonus experiments change behaviour.** `Y` mixes the dose effect with the bonus
  effect in the treated arms. Use `bonus_arm == "*_C"` for controls only, or include the
  arm.
- **Group labels.** `IL_dep` is defined by a positive published dependents' allowance,
  so the group label comes from the benefit record. It is the claimant's dependent
  status, not a choice at the margin.
- **Other benefit schedules.** WA potential duration also varies with base-year wages
  (third schedule above). Extended benefits / FSC weeks may be in `weeks` (IL up to 48).
