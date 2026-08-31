# Taxi — NYC TLC 2009 default-tips RDD

| Field | Value |
|---|---|
| **Q** | `Fare_Amt` (taxi fare in dollars) |
| **Threshold** | 15.0 |
| **Treatment** | `1{Q >= 15}` — at $15 the Vendor (VTS) tip-suggestion system flips from fixed amounts ($2/$3/$4) to percentages (20%/25%/30%) |
| **X** | trip distance, passenger count, tolls, surcharge, hour-of-day, day-of-week |
| **Y** | `Tip_Amt` (tip in dollars) |
| **n** | ~1.5M trips per month after Vendor + credit-card filter |
| **Citation** | Haggag & Paci (2014), AEJ:Applied |
| **Source** | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |

## Building the dataset

```bash
# from the repo root
python -m experiments.datasets.taxi.download                  # 2009-01 only (~470 MB raw, 11 MB processed)
python -m experiments.datasets.taxi.download --months 1 2 3   # selected months
python -m experiments.datasets.taxi.download --months $(seq 1 12)   # full year (~5.5 GB raw)
```

The script downloads the public NYC TLC parquet files into `data/raw/`,
filters each month to `vendor_name == 'VTS'` + credit-card payments + valid
fare/tip ranges, and writes a single combined `data/processed/vts_credit.parquet`.

## Hard-trim utility pilot

The shared exact-hard-trim runner currently uses a deterministic 30,000-trip
pilot subsample, `eps=0.1`, and the pilot-derived fixed nuisance support
`[-6, 11]`. Run it and then create the dollar-denominated taxi utility figure:

```bash
python -m experiments.scripts.hard_trim_existing_applications
python -m experiments.scripts.taxi_utility_curve --cost 0.20
```

The second command treats the cost as dollars per trip assigned the percentage
tip-suggestion regime. It plots estimated utility in cents per hard-trimmed trip,
relative to the observed $15 threshold, and compares a regularized full-sample
fit with five-fold cross-fitting. The cost is illustrative rather than measured;
the pilot has point estimates only and is not yet a publication-ready analysis.

## Notes on the filter

- The Haggag & Paci paper uses 2009 data because it's the year all NYC
  yellow cabs were equipped with the TPEP credit-card system.
- Vendor (VTS) is the only one with the $15 threshold flip; Competitor
  (CMT) used a flat 15/20/30% scheme. We drop CMT rows.
- 2009 parquet has no `mta_tax` column (it was zero / not yet recorded);
  the column exists with all-NaN and is excluded from X.
- `Payment_Type` appears as both "Credit" and "CREDIT" in the raw data;
  the filter uppercases before matching.
