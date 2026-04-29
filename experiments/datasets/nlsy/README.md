# NLSY — AFQT and labor outcomes (stub)

| Field | Value |
|---|---|
| **Q** | AFQT percentile (composite ASVAB) |
| **Threshold** | natural cutoffs: 10 (Cat-V), 30 (Cat-IV), 50, etc. — pick to match question |
| **X** | parental education, race, region, family income, years of schooling |
| **Y** | log hourly wage / annual earnings at chosen reference age |
| **n** | ~12,700 (NLSY79); ~9,000 (NLSY97) |
| **Source** | https://www.nlsinfo.org (free, no DUA for public extracts) |

## Status

**Manual extract required.** No auto-download available.

## Building the extract

1. Sign in (free) at https://www.nlsinfo.org and choose NLSY79 or NLSY97.
2. Open the NLS Investigator web tool and build a tagset including:
   - AFQT score (any of the rescaled variants — `R0614800` is AFQT-89 for NLSY79)
   - demographics: gender, race, region, family income at baseline
   - schooling: highest grade completed
   - an earnings panel for your reference year (e.g. age-30 wages)
3. Download the CSV extract.
4. Place it at `experiments/datasets/nlsy/data/raw/nlsy.csv`.
5. Edit `adapter.py` to map your column names to Q (AFQT), X (covariates),
   and Y (log earnings). The current adapter is a stub.

## Choice of threshold

There is no policy threshold built into AFQT, but several natural
cutoffs from the labor literature:
- AFQT 30 — the AFQT Cat-IV / Cat-III split, used to limit military
  enlistment in some periods.
- AFQT 50 — the median; useful for symmetric comparisons.
- AFQT 10 — Cat-V floor.

Pick the cutoff that matches the question you're asking and pass it
into the `RDDSample` as `threshold`.
