# Bankruptcy means test at the state median income — FJC Integrated Database (IDB)

**Status (2026-11-11): the candidate fails as specified; no adapter.** The IDB carries the
debtor's current monthly income (`cntmnthi`, Form 22A/22C, now 122A-1/122C-1) but not the
household size, so the score "income relative to the state median for the household size"
cannot be built from the public file. The means-test presumption flag and an income class
are not in the file either. Details and a possible lower-bound variant below.

| Field | Value (as planned) | What the IDB has |
|---|---|---|
| **Q** | 12 x CMI / state median income for the household size (DOJ UST tables, by state, size and filing date) | CMI yes (`cntmnthi`); household size **no** |
| **Threshold** | 1.0 (CMI at the state median) | — |
| **Treatment** | `1{Q > 1}`: above-median filers face the full means test (Form 122A-2); presumed abusive for Chapter 7 unless allowed expenses rebut it | not identifiable without household size |
| **X** | total assets, total liabilities (secured, priority and nonpriority unsecured), prior filings, joint filing, district or state, fiscal year, pro se, business vs consumer debt | all present (names to confirm in the codebook) |
| **Y** | filed or converted to Chapter 13 (binary); second loader: dismissed or no discharge | present (original, current and closing chapter; disposition; dismissal and conversion dates) |
| **n** | all petitions since FY2008; about 500,000 to 1,500,000 per fiscal year | not downloaded (see robots) |
| **Source** | FJC IDB, "Bankruptcy Cases Filed, Terminated, and Pending FY 2008-Present", public, no registration | [page](https://www.fjc.gov/research/idb/bankruptcy-cases-filed-terminated-and-pending-fy-2008-present) |

## Codebook findings

The FJC codebook itself could not be read by script: `https://www.fjc.gov/robots.txt`
disallows the IDB paths (quoted below), and the ICPSR mirrors of the same files (NACJD
studies 36468, 36483-36489, "Federal Court Cases: Integrated Data Base Bankruptcy
Petitions, 2008-2015") answer 403 to non-browser clients and are restricted-access data.
The findings below come from published work that uses the IDB file and names its
variables, mainly Pattison and Millimet, "Eligibility Screening and Means Testing in
Consumer Bankruptcy" (SMU, January 2025, Online Appendix Tables A2 and A3), and Pattison and
Hynes, "Asset Exemptions and Consumer Bankruptcies: Evidence from Individual Filings"
(J. Law and Economics 63(3), 2020). **Manual step 2 below is to confirm them against the
codebook.**

Present in the IDB (exact names quoted where the source quotes them):

- `cntmnthi` — "The monthly income of the household from all sources, averaged over the six
  months prior to filing; Current Monthly Income from form 22A for Chapter 7 debtors or form
  22C for Chapter 13 debtors." (Pattison and Millimet, Table A3). This is the means-test CMI,
  Form 122A-1 line 11 / 122C-1 line 14 in the current forms.
- `avgmnthi` — "Estimated monthly household income of debtor at the time of filing, reported
  as Average Monthly Income from Schedule I." Average monthly expenses (Schedule J) are also
  present (their `disp income` = Schedule I income minus Schedule J expenses).
- Total assets, real property value, total liabilities, secured debt, priority unsecured and
  nonpriority unsecured debt, and the nondischargeable components reported on the statistical
  summary (domestic support obligations, taxes and government debts, student loans, pension
  obligations).
- Pro se indicator for the first-listed debtor; joint petition; debtor's zip code; district;
  original, current and closing chapter (the FJC page names `ORGFLCHP`-type fields and, from
  FY2023, `ORGFLSUBCHP`, `CRNTFLSUBCHP`, `CLSUBCHPT` for subchapter V); filing and
  termination dates; disposition (dismissal, conversion, discharge, with a reason for some
  dismissals); prior-filing indicator (`PRBANKFL`, per 1328f.org's methodology page).

Not in the IDB:

- **Household size** (Form 122A-1 line 14). Pattison and Millimet: "We do not observe the
  household size, and instead we assume each household has zero non-filing members"; they
  use the one-person median for single filers and the two-person median for joint filers,
  which "creates one-sided measurement error". Pattison and Hynes make the same assumption.
- **Means-test presumption flag** (122A-2 line 39) and the above/below-median class. The
  same authors recover above-median status only by matching cases to PACER docket text and
  looking for a filed Form 122A-2 or 122C-2 (available for a subset of districts, 2015-2019),
  which they would not do if the flag were in the file.
- The two income fields are the only income items; there is no state-median column.

So the file has more than "the presumption flag or the income class" (it has CMI itself) but
less than the design needs (no household size). Since the threshold is specific to the
household size, Q is unknown for every case, and a score built with the one-earner (or
two-person for joint) median is a lower bound on the true Q with one-sided misclassification:
some filers with Q_lb > 1 are truly below the median. The candidate therefore fails the
criterion set for it and stops here.

## Published numbers (no local computation; nothing was downloaded)

From Pattison and Millimet (consumer Chapter 7 and 13 cases, FY2011-2015, 83 districts,
positive disposable income):

- n = 2,525,077 cases (61.7% of all consumer filers); 52.5% Chapter 7, 47.5% Chapter 13.
  The Chapter 13 share of all consumer filings is about 30%.
- Treated share under their lower-bound score (`AMI`, 12 x `cntmnthi` above the one- or
  two-person state median): 29.5% of Chapter 7 cases and 44.9% of Chapter 13 cases, 36.8%
  overall.
- Mean `cntmnthi` $3,200 (sd 2,100) in Chapter 7, $4,000 (2,700) in Chapter 13; mean
  `avgmnthi` $3,000 and $3,700.
- Discharge: 43% of Chapter 13 and 96% of Chapter 7 cases (2010-2014).
- Bunching check (their Appendix Figure C1, $200 bins of 12 x CMI minus the applicable
  median): 8,109 filings in [-$199, $0] and 6,455 in [$1, $200], "a discontinuity of 1,654
  cases", a 20% drop in the density just above the lower-bound threshold. With the true
  household sizes the drop at the true threshold can only be sharper. This is selection into
  filing (or into reported income) at the line, which an RD design would have to confront.

The R-squared of Q on X, the counts in the 0.90-0.99 and 1.00-1.09 bins of Q, and local n,
treated share and Chapter 13 rate were not computed: Q cannot be formed, and the FJC files
were not fetched.

## robots.txt

`https://www.fjc.gov/robots.txt` (fetched 2026-11-11):

```
User-agent: *
Crawl-delay: 30

# FJC customizations
# 1/25/2017 - request that interactive IDB pages not be crawled
Disallow: /research/idb/
Disallow: /sites/default/files/idb
```

The IDB landing page, the bankruptcy page, the snapshot files and the codebooks are all under
these two prefixes, so nothing on fjc.gov was fetched by script and `download.py` does not
fetch from fjc.gov. (The four-page "IDB Research Guide" at
`/sites/default/files/IDB-Research-Guide.pdf`, a different, upper-case path, was read; it
confirms bankruptcy coverage from FY2008 and that the schedules of assets and liabilities
feed the file, but lists no variables.)

`https://www.justice.gov/robots.txt` (fetched 2026-11-11): `User-agent: *`, disallows
`/core/`, `/profiles/`, `/admin/`, `/search/`, `/user/...`, `/media/oembed`, a few README
files, `Crawl-delay: 10`. `/ust/...` is allowed; `download.py --medians` honours the delay.

`https://www.icpsr.umich.edu/robots.txt`: disallows `/files/` (where the ICPSR codebooks
live) for `User-agent: *`, so the ICPSR codebook PDFs were not fetched either.

## Manual download steps (if the author wants the files anyway)

1. Open https://www.fjc.gov/research/idb/bankruptcy-cases-filed-terminated-and-pending-fy-2008-present
   in a browser. The page says the data cover "all petitions filed under the Bankruptcy Code
   on or after October 1, 2007 and any petitions filed before October 1, 2007 still pending on
   that date", that data accessed before September 2021 must be re-downloaded (AOUSC found
   anomalies in August 2021), and that three subchapter V variables were added with the
   FY2023 / December 31, 2022 datasets.
2. Download the bankruptcy codebook (pdf) and confirm the variable list above, in particular
   that no household-size or means-test-presumption variable exists. Save it in `data/raw/`.
3. If proceeding with a lower-bound score: download the three most recent fiscal-year
   snapshot files (text, pipe-delimited, or SAS; a few hundred MB each) into `data/raw/`.
4. `python -m experiments.datasets.fjc_bankruptcy.download` lists what is present.
5. DOJ UST median-income tables, one per effective date (cases filed on or after that date):
   `python -m experiments.datasets.fjc_bankruptcy.download --medians 20260401 ...` writes
   `data/raw/medians/median_income_<date>.csv` (state, one earner, 2, 3, 4 people, add-on
   per person over 4). The dates are listed on https://www.justice.gov/ust/means-testing;
   recent ones include 20240515, 20260401 and 20260715. Tested on 20260401 (55 rows: 50
   states, DC, Guam, Northern Mariana Islands, Puerto Rico, Virgin Islands; add-on $11,100
   per person over four; a second run skips the file).

## If the author wants a variant anyway

- **Lower-bound score.** Q_lb = 12 x `cntmnthi` / median(state, 1 + joint) with threshold 1.
  Treated status is right for households with no non-filing members and too generous above
  the line otherwise; the first stage (a filed Form 122A-2 / 122C-2, or a dismissal under
  707(b)(2)) is not in the IDB, so the design would be fuzzy with an unobserved first stage.
  The published density drop at Q_lb = 1 (20%) also says filers sort around the line.
- **Docket match.** Pattison and Millimet's forms indicator from PACER docket text gives
  above-median status (not Q) for a subset of districts; it needs the archived docket dumps
  (archive.org, "federal-court-dockets") and a case-number match.

Neither was built; the design brief says to stop when the household size is missing.
