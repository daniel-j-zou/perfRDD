# Lending Club — loan-level (stub)

| Field | Value |
|---|---|
| **Q** | `fico_range_low` (or midpoint of FICO range) |
| **Threshold** | 660 (early), 600 (later eligibility floor) |
| **X** | annual income, employment length, DTI, home ownership, loan purpose, state, loan term |
| **Y** | interest rate (continuous), realized return / loss |
| **n** | ~2.2M loans (2007–2018) |
| **Source** | Lending Club historical loan archive |

## Building (requires Kaggle auth)

```bash
pip install kaggle
# Get an API token from kaggle.com (Account -> Create New API Token),
# then place ~/.kaggle/kaggle.json with chmod 600 perms.
python -m experiments.datasets.lending_club.download
```

The download script pulls `wordsforthewise/lending-club` (~1 GB
compressed) and renames the accepted-loans file to `data/raw/loans.csv`.

## Notes on choosing Q

- FICO is reported as a 5-point range (`fico_range_low`,
  `fico_range_high`). The adapter uses `fico_range_low` as Q.
- For the LC grade-bucket discontinuity, an alternate adapter could use
  the internal base interest rate or the grade-driven sub-grade index
  as Q. We default to the eligibility-floor framing (660 / 600) because
  it matches the cleanest hard cutoff in the LC underwriting policy.

## Y choice

The current adapter uses originated `int_rate` (continuous). Alternatives
worth trying: realized return / loss given default (require additional
processing of `total_pymnt`, `total_rec_prncp`, `recoveries`).
