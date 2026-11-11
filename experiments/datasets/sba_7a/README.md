# SBA 7(a) loans at the $150,000 guarantee notch — FOIA loan-level files (data.sba.gov)

| Field | Value |
|---|---|
| **Q** | gross approval amount in dollars (`GrossApproval`) |
| **Threshold** | $150,000: the SBA guarantees 85% of a standard 7(a) loan of at most $150,000 and 75% of a larger loan (program parameter; 90% on both sides in the stimulus windows, which are excluded) |
| **Treatment** | `1{Q <= 150000}`: the higher guarantee applies at or below the line, so the rule treats below; `adapter.mirrored()` flips the score for scripts that treat above |
| **X** (46 columns in the default sample) | term in months; jobs supported (raw and log1p); approval fiscal year; business type (individual, partnership; base corporation); business age (new business, change of ownership, unanswered; base existing); revolver, collateral and franchise flags; log number of 7(a) loans by the lender in the whole file (lender size); processing method (PLP, CLP, LowDoc, Community Express/Advantage, other initiatives, CAPLines; base 7(a) General); NAICS 2-digit sector indicators (19, base 72 accommodation and food); Census division indicators (9, base Pacific). `include_rate=True` adds the initial interest rate (in the file from FY2009 only). Columns constant in the chosen window are dropped and listed in `extras["dropped_constant_columns"]` |
| **Y** | 1 if charged off (`LoanStatus == "CHGOFF"`), 0 if paid in full (`"P I F"`); loans still outstanding (`EXEMPT`), cancelled (`CANCLD`) or undisbursed (`COMMIT`) are excluded |
| **n** | 348,792 resolved loans in the default window FY2001-2016 (1,976,761 loans in the four files; 727,613 standard-program loans inside the notch windows; 412,406 of them FY2001-2016; 349,332 resolved; 348,792 with complete X) |
| **Treated share** | 42.2% (147,203 at or below $150,000) |
| **Charge-off rate** | 18.05% overall; 22.2% at or below the line, 15.0% above (loan size and default are negatively related; the local rates on both sides of the line are 18.1%, see below) |
| **R-squared of Q on X** | 0.361 (numpy least squares, Q in dollars; 0.669 for log Q). Adding the interest rate on FY2010-2016 raises it from 0.334 to 0.341 on the same rows |
| **Source** | U.S. Small Business Administration, Office of Capital Access, "7(a) & 504 FOIA", https://data.sba.gov/dataset/7a-504-foia, public domain (https://www.usa.gov/publicdomain/label/1.0/), refreshed quarterly; vintage "as of September 30, 2026" |
| **Expected effect** | null on default: Bachas, Kim and Yannelis (2021) find strong bunching just below $150,000 and interest rates and ex-post charge-off rates that trend smoothly through the line; the guarantee moves lending volume, not default |
| **Extras** | `guarantee_share` (realised `SBAGuaranteedApproval / GrossApproval`), `guaranteed_amount`, `charge_off_amount`, `interest_rate` (NaN before FY2009), `approval_date`, `state`, `lender_id`, `processing_method`, `naics_code` |

## Getting the data

```bash
python -m experiments.datasets.sba_7a.download          # dictionary + four 7(a) csv files, ~910 MB
python -m experiments.datasets.sba_7a.download --only 2010_2019
```

The script fetches, into `data/raw/` (git-ignored):

| File | Size | Rows |
|---|---|---|
| `7a_504_foia_data_dictionary.xlsx` | 24 KB | field definitions, two sheets (7(a), 504) |
| `FOIA_7a_FY1991_FY1999_asof_260930.csv` | 146.5 MB | 337,033 |
| `FOIA_7a_FY2000_FY2009_asof_260930.csv` | 318.7 MB | 690,333 |
| `FOIA_7a_FY2010_FY2019_asof_260930.csv` | 255.3 MB | 545,751 |
| `FOIA_7a_FY2020_Present_asof_260930.csv` | 188.5 MB | 403,644 |

from `https://data.sba.gov/sites/default/files/uploaded_resources/<file>` (links read from
the dataset page and from the DKAN metastore API,
`https://data.sba.gov/api/1/metastore/schemas/dataset/items`). It is idempotent: a file
already present with the server's `Content-Length` is skipped; a partial download goes to
`<name>.part` and is renamed when complete. A new quarterly vintage changes the `asof_`
tag in the file names: edit `ASOF` in `download.py` and `RAW_FILES` in `adapter.py`.
`adapter.build()` reads the four files once (about 10 s) and caches
`data/processed/sba_7a_all.parquet` (git-ignored); `--force` or `build(force=True)` refresh.

Manual alternative: open https://data.sba.gov/dataset/7a-504-foia, download the four
"FOIA - 7(a) ..." csv files and the data dictionary, and put them in `data/raw/` under the
names above.

### robots.txt check (2026-11-11)

`https://data.sba.gov/robots.txt` (Drupal default) has no rule against the dataset page or
the file directory. The relevant lines are

```
User-agent: *
# Directories
Disallow: /core/
Disallow: /profiles/
# Files
Disallow: /README.md
Disallow: /sites/README.txt
...
# Paths (clean URLs)
Disallow: /admin/
Disallow: /search/
Disallow: /user/login
...
# Search
Disallow: /dataset/search?
Sitemap: https://data.sba.gov/sitemap.xml
```

`/dataset/7a-504-foia` and `/sites/default/files/uploaded_resources/` are not disallowed
(only `/sites/README.txt` is), so the files were fetched by script.

## The design

- **The notch.** Standard 7(a) loans: 85% guarantee at or below $150,000, 75% above
  (13 CFR 120.210; SBA SOP 50 10). In the data the realised guarantee share is exactly
  0.85 for 90.4% of loans at or below the line (within $100,000-250,000) and exactly 0.75
  for 98.5% above it; the mean share is 0.8405 in ($140k, $150k] and 0.7513 in
  ($150k, $160k], a 9 percentage-point jump. The remaining loans below the line took a
  lower guarantee (lender's choice), so the design is slightly fuzzy on the treated side.
- **Programs where the notch does not apply, dropped** (`EXCLUDED_METHODS`): SBA Express
  (802,609 loans, flat 50% guarantee at any size; 41% of all rows), Export Express, EWCP
  and international trade products (90%), dealer floor plan and one domestic revolving
  line. The kept products: 7(a) General, Preferred Lenders (PLP, 66% of the design
  frame), Certified Lenders, Low Documentation (ended FY2005; its own cap was $150,000),
  Community Express and Community Advantage, Small Loan Advantage, Patriot Express, Rural
  and Gulf Opportunity initiatives, and the CAPLines. Processing-method indicators are in X.
- **Windows where the notch is in force** (`NOTCH_WINDOWS`), read off the realised
  guarantee shares by approval month on the standard programs: 2001-01-01 to 2009-02-28,
  2010-06-01 to 2010-09-30, 2011-01-01 to 2020-12-31 and 2021-10-01 onward. Excluded:
  before 2001 (80% at or below $100,000 in 1996-2000, older schedules before; the $150,000
  / 85% schedule starts with loans approved in January 2001), the ARRA window 2009-03 to
  2010-05 and the Jobs Act window 2010-10 to 2010-12 (90% on both sides), and the
  Economic Aid Act window 2021-01 to 2021-09 (90% on both sides). Boundary months are
  mixed; the windows keep months where at least 85% of loans follow the 85/75 pattern.
- **Default fiscal years 2001-2016** (`load(fiscal_years=(2001, 2016))`). Resolved share
  (paid in full or charged off, as of 2026-09-30) is 86-88% for FY2001-2010, 78-84% for
  FY2011-2016, then 68% (FY2017), 62%, 53% (FY2019), 47%, 27% (FY2022), 9% (FY2024). In
  the young cohorts the resolved loans are the early payoffs and the early defaults, and
  the charge-off rate among them rises (10-12% for FY2017-2019 and FY2022-2023 against
  6-9% for FY2011-2016), so the default window stops at FY2016 (ten years of seasoning).
  Charge-off rates by cohort among resolved loans: 14-18% FY2001-2004, 24-38% FY2005-2008
  (the crisis cohorts), 12% FY2010, 7-9% FY2011-2016.
- **Bunching.** 8.15% of the default sample (28,410 loans) is within $10,000 of the line.
  Counts by $10,000 bin: ($130k, $140k] 6,138; ($140k, $150k] 23,842, of which 19,824 are
  exactly $150,000; ($150k, $160k] 2,563; ($160k, $170k] 3,255; ($170k, $180k] 4,599.
  Density ratio just below to just above: 23,842 / 2,563 = 9.3 (10.1 with the $140,000
  mass point included). Round-number heaping is everywhere ($100,000: 9,257; $200,000:
  5,423; $250,000: 5,306 loans exactly), but the hole above $150,000 is the notch.
  Bachas, Kim and Yannelis use this bunching to estimate a credit-supply elasticity near
  5. For the paper's density conditions this is manipulation of the score at the line; a
  donut (drop [$150,000 - delta, $150,000 + delta]) or an analysis away from the line is
  needed, and the mass point at exactly $150,000 belongs to the treated side.
- **Local charge-off rates** are flat through the line: 18.1% in ($140k, $150k] and 18.1%
  in ($150k, $160k] (16.6% in ($100k, $140k] and 16.7% in ($160k, $200k]), consistent
  with the smooth charge-off profile Bachas, Kim and Yannelis report. The expected effect
  at the line is null; this is a negative example.
- **Cost.** The treatment's cost is the extra 10 percentage points of guarantee on the
  loan balance: `extras["guaranteed_amount"]` and `extras["charge_off_amount"]` give the
  realised guarantee and loss per loan.

## Fields not in the file

- Number of employees: not in the FOIA file (only `JobsSupported`, self-reported jobs
  created plus retained).
- Urban/rural: no field. `ProjectCounty` and the zip codes are present, so a USDA
  rural-urban continuum code or the HUD zip-county crosswalk could be merged (another
  download; not done).
- Lender size: the FDIC certificate (`BankFDICNumber`, missing for 10%) would need the FDIC
  institution file; the adapter uses the lender's count of 7(a) loans in the file instead.
- Interest rate: `InitialInterestRate` is empty before FY2009 (89% filled in FY2009, 100%
  from FY2010); the `include_rate=True` option restricts to those rows.
- Borrower financials, credit score and collateral value: not released.
- `BusinessAge` changed coding in FY2004 and FY2018 (the "New Business or 2 years or less"
  and "Change of Ownership" categories exist only from FY2018); the adapter maps the
  categories to new / change of ownership / unanswered / existing.

## Reference

Bachas, N., Kim, O. S. and Yannelis, C. (2021). Loan guarantees and credit supply.
Journal of Financial Economics 139(3), 872-894. Working-paper pdf:
https://cmepr.gmu.edu/wp-content/uploads/2019/08/Bachas-Yannelis-Loan-Guarantees-and-Credit-Supply-1.pdf
