# CWBH unemployment claims (Landais 2015): weekly benefit schedules in five states, 1976-84

| Field | Value |
|---|---|
| **Q** | Highest-quarter earnings in the base period (LA, ID, MO, NM); WA uses the two highest quarters, which the file lacks (proxy: hqw and (bpw - hqw)/3). `load_dose()`: Q = score / the earnings at the cap kink of the claimant's schedule |
| **Threshold** | 1 (the cap kink) in every schedule; a floor kink below |
| **Treatment** | Dose A = weekly benefit amount from the formula, `clip(round(s / frac), min_b, max_b)`, as a share of the cap (`dose="relative"`, a_ref = 1) or in 1984 dollars (`dose="real"`) |
| **Rule class** | Kinked linear schedule; 19 state-period schedules in LA, ID, MO, NM (caps $85-205, $108-218 in 1984 dollars) |
| **X** | age, sex, race, education, dependents, household size, expected recall, industry, weeks worked, log(bpw / hqw) (sets potential duration), state x quarter-of-claim indicators |
| **Y** | `weeks_paid` (weeks of UI paid in the benefit year, all tiers; default), `duration` (first to last certified week, Landais's measure), `weeks_claimed` |
| **n** | 130,935 claims on the formula (LA, ID, MO, NM), 128,315 after trimming 1% tails of log Q; 107,140 people. WA adds 42,247 with an imputed score |
| **Source** | Landais (2015), "Assessing the welfare effects of unemployment benefits using the regression kink design", AEJ: Economic Policy 7(4); openICPSR 114581 (author's login) |

**Fit: the best UI data so far.** The formula reproduces 95.9% (ID), 97.5% (LA), 97.5% (MO) and
98.2% (NM) of reported benefits; 45% of claims are at the cap (the reference dose); R^2 of Q on X
is 0.44. Results: `DOSE_SCHEDULE_DATASETS_20261028.md` and
`simulations/dose_schedule/dose_applications_eval.tex`, Section 1b.

## Getting the data (author)

1. Sign in at https://www.openicpsr.org/openicpsr/project/114581/version/V1/view, download the
   project and accept the terms.
2. Unzip `Prog_AEJ_RKD/Data/` (2.3 GB), `Prog_AEJ_RKD/Programs/` and `LICENSE.txt` into `data/raw/`.

The first `claims()` call reads the week-level records (about 15 s) and caches one row per claim
in `data/processed/claims_<state>.parquet`.

## The rule

Parameters from Landais's state programs (`Programs/prog_<state>.do`), keyed on the date the
benefit year began; listed in `adapter.SCHED`. Rounding was read from the data (`rule_check`):
up to the next dollar everywhere, except Louisiana from 3 April 1983 (nearest dollar) and Idaho
from 1 July 1983 (down). Claims off the formula (2-4%) are dropped (`on_schedule=False` keeps
them). Landais instead drops claims on the wrong side of the kink and moves the others onto the
formula.

Sample filters follow `Programs/cleaning_cwbh_v1.do` and the state programs: monetarily
eligible, regular state UI, new claims, no denial, no separation disqualification (not recorded
in LA), benefit year begun before 1 Jan 1984 and after the first certified week in the state's
file, no gap of more than three weeks between certified weeks.

## Caveats

- **Data use.** The readme asks users to contact Bruce Meyer or Patricia Anderson for a formal
  agreement to use the CWBH files; the openICPSR deposit is CC BY 4.0. Settle this before
  publishing results.
- **Potential duration.** Tier-1 potential duration is min(maxdur, c x bpw / WBA), so a lower
  benefit lengthens it for claimants whose base-period earnings bind. The dose effect is the
  effect of the whole schedule. Extended benefits (EB, FSC) also changed potential duration in
  1980-83; the state x quarter indicators absorb their timing.
- **Washington** is left out of the default sample: its score is measured with error.
  `load_dose(states=STATES)` adds it with Landais's imputation (score = 25 x WBA in the interior).
- `WAwag.dta` (WA wage records, for non-employment duration) is not used yet.
