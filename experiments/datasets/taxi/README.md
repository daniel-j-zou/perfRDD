# Taxi — "Default Tips" RDD (stub)

| Field | Value |
|---|---|
| **Q** | fare amount (running variable around the default-tip-suggestion threshold) |
| **X** | trip distance, time of day, day of week, payment type |
| **Y** | tip amount (continuous) |
| **Citation** | Haggag & Paci (2014), AEJ:Applied |

## Status

**Not yet implemented.** The adapter is a stub; populate `data/raw/` and
fill in `adapter.load()`.

## Data

NYC TLC taxi trip records are public:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Pick a year/month with credit-card transactions (which carry tip) and place
the parquet/CSV at `data/raw/trips.csv` (or update the adapter).
