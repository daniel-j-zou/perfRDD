# Lending Club — loan-level (stub)

| Field | Value |
|---|---|
| **Q** | `fico_range_low` (or midpoint of FICO range) |
| **Threshold** | 660 (early), 600 (later eligibility floor) |
| **X** | annual income, employment length, DTI, home ownership, loan purpose, state, loan term |
| **Y** | interest rate (continuous), realized return / loss |
| **n** | ~2.2M loans (2007–2018) |
| **Source** | Lending Club historical loan archive |

## Status

**Not yet implemented.**

## Data

The Lending Club archive is mirrored on Kaggle ("All Lending Club loan
data"). Download `accepted_*.csv` and place at `data/raw/loans.csv`
(rename or update the adapter as needed).

Notes on choosing Q:
- FICO is reported as a 5-point range (`fico_range_low`, `fico_range_high`).
  Use `fico_range_low` as Q if studying the eligibility floor.
- For the LC grade-bucket discontinuity, Q can instead be the internal
  base interest rate or the grade-driven sub-grade index.
