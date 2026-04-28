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

**Not yet implemented.**

## Data

Build an extract via NLS Investigator (web tool) and export as CSV.
Recommended fields: AFQT score, demographics, schooling, earnings panel.
Place at `data/raw/nlsy.csv`.
