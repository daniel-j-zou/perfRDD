# FoodAPS-1 (2012-13): the SNAP benefit formula as a dose schedule

| Field | Value |
|---|---|
| **Q** | The score: net income reconstructed from PUF variables (gross household income minus the SNAP deductions; not floored at 0). `load_dose()`: Q = 0.3 x net income / maximum allotment (`dose="relative"`), or net income in dollars (`dose="dollars"`) |
| **Threshold** | Q = 0 (net income 0): the maximum allotment ends; the $16 minimum binds for 1-2 persons near Q = 1 |
| **Treatment** | Dose A = formula benefit, max_allot(size) - ceil(0.3 x net income), at least $16 for 1-2 persons; relative: A / max allotment, a_ref = 1 (the maximum allotment) |
| **Rule class** | Linear phase-out with a cap at the maximum allotment and a floor; one schedule per household size (max $200 for 1 person to $1,202 for 8, +$150 each) |
| **X** | 54 columns: household size, children, elderly, disabled, employed and looking, income-type indicators (earnings, UI, transfers, retirement/disability, investment, other), respondent's age, sex, race, Hispanic, married, US-born, health, employment, highest education, tenure, public/subsidised housing, vehicle, WIC, school lunch and breakfast, region, rural, nonmetro, interview month |
| **Y** | `fah_food` (7-day food-at-home spending on food items, item file; default), `fah_paid` (FAH event totals), `fafh_paid`, `total_food` (FAH food + FAFH) |
| **n** | 1,316 households with administratively confirmed SNAP receipt (SNAPNOWADMIN = 1); 1,190 of them report a last SNAP amount; 1,288 after trimming the score's 1% tails |
| **Source** | USDA ERS, FoodAPS public-use files (CSV, updated 4/28/2021); SNAP parameters from USDA FNS FY 2012 and FY 2013 tables |

**Fit: does not stick.** The formula cannot be reproduced from the public-use file. The
reconstruction lands in the recorded $50 bin of the self-reported last SNAP amount for 15.5% of
participants (17.7% within $10 of the bin, 21.7% within 10%). Mathematica's simulated benefits
(BENEST1-4, computed on the restricted file with State rules) do little better: 21.8-22.5% in the
bin, 25-27% within $10, 29.8-31.2% within 10%. ERS reports the same for the administrative amounts
(codebook, table 9: about 20% of matched households within 10%). The household-size maximum alone
correlates more with the recorded amount (Pearson with the bin midpoint 0.58) than the formula does (0.47). Only 9.2% of
participants are at the reference dose by the formula (BENEST3: 20.2%; recorded bin consistent
with the maximum: 12.4%), and 16.9% of participants get a formula benefit of 0. R^2 of the score
on X is 0.34 (trimmed; 0.24 untrimmed). The estimator's transported effects are unstable: rotation
sd larger than the estimates in most cells, values of hundreds to thousands of dollars against a
mean 7-day spending of $93.

The recorded amount cannot replace the formula: in the PUF it is a $50-bin self-report
(SNAPLASTAMT_R, codes 1-18), and it is not a function of the reconstructed score (Spearman 0.39).
The administrative amounts (ADMINAMT1-4) and the State are in the restricted file only.

## The rule and its source

Parameters for the 48 States and DC (`adapter.py`, constants at the top):

| | FY 2012 (Oct 2011 - Sep 2012) | FY 2013 (Oct 2012 - Sep 2013) |
|---|---|---|
| Maximum allotment (ARRA) | 1: $200, 2: 367, 3: 526, 4: 668, 5: 793, 6: 952, 7: 1,052, 8: 1,202, +150 | same |
| Standard deduction | 1-3: $147, 4: 155, 5: 181, 6+: 208 | 1-3: $149, 4: 160, 5: 187, 6+: 214 |
| Excess shelter cap | $459 | $469 |
| Minimum benefit (1-2 persons) | $16 | $16 |
| Net income standard (100%) | 908, 1,226, 1,545, 1,863, ... +319 | 931, 1,261, 1,591, 1,921, ... +330 (computed) |
| Gross income standard (130%) | 1,180, 1,594, 2,008, 2,422, ... +414 | 1,211, 1,640, 2,069, 2,498, ... +430 (computed) |

Sources (fetched 2026-11-12; www.fns.usda.gov now redirects to www.fna.usda.gov, whose robots.txt
disallows only /sites/README.txt and /node/add/; the tables were also read on the mirror host
fns-prod.azureedge.us; www.usda.gov returns 403 to scripts):

- https://www.fna.usda.gov/snap/allotment/fy-2012-information (maximum allotments, standard deductions, shelter cap)
- https://www.fna.usda.gov/snap/allotment/fy-2013-information
- https://fns-prod.azureedge.us/sites/default/files/snap/FY12-Minimum-SNAP-Allotment.pdf and `FY13-Minimum-SNAP-Allotment.pdf`
- https://fns-prod.azureedge.us/sites/default/files/resource-files/FY12-Income-Eligibility-Standards.pdf
  (the FY 2013 file is not linked and returns 404; the FY 2013 standards above are the 2012 HHS
  guideline / 12, rounded up, which is the PUF's POVGUIDE_HH; not checked against an FNS table)
- Benefit = maximum allotment - 30% of net income rounded up to the dollar (7 CFR 273.10(e)(2)(ii)(A));
  deductions as in the FoodAPS household codebook, section 2.3.8.3.6.

Interview month to fiscal year: April-September -> FY 2012; October-December 2012 and January 2013
-> FY 2013. Interviews run April 2012 to January 2013 (INITINTRVMON), all before the ARRA
increase ended on 31 October 2013.

Reconstruction from PUF variables (`reconstruct`): gross = INCHHAVG_R; earnings = gross x
earners / (earners + unearned income sources), because the PUF has only indicators of each
person's income types (`earn="all"` and `"none"` give 15.9% and 14.6% in the bin); dependent care
= EXPCHILDCARE_R; child support paid = EXPCHILDSUPPORT_R; medical = (health insurance + copays +
doctor + prescriptions) x share of elderly or disabled members, minus $35; shelter = rent/mortgage
+ insurance + property tax + reported electricity, heating fuel and sewer/garbage. Elderly: age group
60+. Disabled: under 60, not working because disabled, with retirement/disability income.
Replacing reported utilities by a flat allowance of $250-400 raises the in-bin share to 17-19%.

`rule_check()` returns the table of agreement rates (formula variants, BENEST1-4, maximum
allotment) against the recorded bin and against BENEST3.

## Estimator run (`dose_config.py`)

`runs/dose_schedule/snap_foodaps.log` (Y = fah_food; PSU clusters): readiness n = 1,288, R^2 0.34,
sd(eta) 0.58, sd(T) 0.41, share at the reference 8.3%. Effects vs the maximum allotment by score bin,
estimate (rotation sd, coverage), in dollars per week:

| score bin | a = 0 | a = 0.25 | a = 0.5 | a = 0.75 |
|---|---|---|---|---|
| -0.85 to 0 | 1,091 (1,473, 45%) | -64 (122, 69%) | -64 (29, 73%) | 6 (6, 73%) |
| 0 to 0.25 | 551 (807, 69%) | -59 (82, 87%) | 32 (102, 91%) | 485 (685, 91%) |
| 0.25 to 0.5 | 195 (312, 79%) | -35 (30, 90%) | -25 (14, 93%) | -6 (10, 86%) |
| 0.5 to 0.75 | -12 (147, 89%) | -109 (57, 93%) | 113 (288, 94%) | 1,261 (1,897, 78%) |
| 0.75 to 1 | 110 (401, 90%) | -116 (109, 88%) | -42 (174, 88%) | 197 (482, 63%) |
| 1 to 1.5 | -49 (346, 98%) | -235 (171, 65%) | 576 (1,105, 65%) | 4,678 (6,881, 40%) |
| 1.5 to 4 | -1,703 (3,353, 92%) | -9,310 (9,110, 15%) | 29,798 (45,822, 15%) | 239,302 (341,832, 6%) |

Modifier test: coef -118.5 (se 57.2), r2_absorbed 0.98. Total food spending
(`snap_foodaps_total_food.log`) and 24 core covariates (`snap_foodaps_core.log`) are as unstable.

## Caveats

- The PUF lacks the State (no State utility allowances, BBCE limits or asset rules), individual
  income amounts, the elderly/disabled medical share (EXPOPMEDICAL60) and the administrative SNAP
  amounts; it applies data swapping to individuals (User's Guide, section 4.2.3).
- SNAP units can be smaller than the household (Mathematica splits 273 households); the unit here
  is the household without guests.
- Benefits are budgeted prospectively and certified for months; the survey's "last month" income
  and expenses need not be the income the office used.
- 31.5% of participants have gross income above 130% of the poverty guideline (13.4% above 200%):
  BBCE States, or income measured at a different time.
- FAH event totals include nonfood items and tax; `fah_food` sums food items from the item file.

## Getting the data

`download.py` (run from `code/`, see its docstring) fetches two static files linked from
https://www.ers.usda.gov/data-products/foodaps-national-household-food-acquisition-and-purchase-survey:
`/media/5594/csv-data-files.zip` (39 MB) into `data/raw/csv/` and `/media/5595/documentation.zip`
(6.2 MB) into `data/raw/documentation/`, and extracts only the CSV / PDF members. No login, form or
terms page. www.ers.usda.gov/robots.txt (checked 2026-11-12): `User-agent: * Allow: /`, disallowing
only /core/, /profiles/, /README.md and /web.config. SHA-256 of both zips is recorded in the script.
`adapter.frame()` caches one row per household in `data/processed/households.parquet`.

Restricted-use files (State, administrative SNAP amounts, income amounts) require an application
through the Standard Application Process portal, https://www.researchdatagov.org/, and a fee-based
secure enclave (ERS data-access page).
