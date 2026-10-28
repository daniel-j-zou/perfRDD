# CMS nursing-home star ratings (waiting for the author's download)

| Field | Value |
|---|---|
| **Q** | health-inspection weighted score (deficiency points from the last three inspection cycles; lower is better) |
| **Threshold** | state-specific cut points. Within each state the best 10% get 5 stars, the worst 20% get 1 star, and the middle 70% split evenly into 2-4 stars. Recomputed monthly |
| **Treatment** | the inspection star rating; the overall star adds or subtracts a star for the staffing and quality-measure ratings |
| **Rule class** | group-specific cutoffs (state); several cutoffs on one score; a decision matrix for the overall rating |
| **X** | certified beds, ownership type, chain, in-hospital, urban/rural, state, resident mix |
| **Y** | later deficiencies, average residents per day (occupancy), staffing hours (Payroll-Based Journal) |
| **n** | about 15,000 homes per month |
| **Source** | CMS Provider Data Catalog, public |

## Getting the data (author)

data.cms.gov returns "Access Denied" to scripts (checked 2026-10-27), so these are
browser downloads. Save into `data/raw/`:

1. The current **Provider Information** file:
   https://data.cms.gov/provider-data/dataset/4pq5-n9py (Download, CSV).
2. Monthly archives: https://data.cms.gov/provider-data/archived-data/nursing-homes.
   One zip per month. Start with one month per quarter, 2022 onward. Each zip has
   Provider Information and the health-deficiency files.

Claude can fetch the state cut-point tables (PDFs on cms.gov) without your help.

## Caveats

- The cut points are percentiles, so they move with the state's score distribution.
- Homes appear in every month, so outcomes are serially dependent.
