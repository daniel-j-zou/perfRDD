# Freddie Mac Single-Family Loan-Level Dataset (sample files 2008-2026 in hand)

| Field | Value |
|---|---|
| **Q** | (credit score, original loan-to-value) at origination; DTI as a third score |
| **Threshold** | the GSE pricing grid: credit-score bands x LTV bands set the up-front fee. Mortgage insurance is required above 80% LTV |
| **Treatment** | the grid cell (fee level), or mortgage insurance at LTV > 80 |
| **Rule class** | a decision matrix on two scores; several cutoffs on each |
| **X** | loan purpose, property type, occupancy, number of borrowers, first-time buyer, state, channel, origination year |
| **Y** | 90+ day delinquency or default within N years (monthly performance file) |
| **n** | the sample files hold 50,000 loans per origination year |
| **Source** | Freddie Mac, free registration |

## Screen for the dose-schedule design (Claude, 2026-11-02)

`python -m experiments.datasets.freddie_mac.screen` (log in `runs/dose_schedule/freddie_screen.log`).

- **The fee grid reaches borrowers.** For 30-year fixed-rate owner-occupied purchase loans
  originated 2009-22, the rate (minus its first-payment-month x LTV-band mean) falls at 11 of
  the 12 grid cutoffs between credit scores 640 and 740 in the 75-80 and 90-95 LTV bands, by
  about 8 bp on average; placebo cutoffs between them give -6 to +3 bp (mean -1.6). After the
  2023 grid change the pattern is noisier.
- **But the covariates barely predict the score.** R^2 of the credit score on LTV, DTI, loan
  size, borrowers, first-time buyer, MI, CLTV gap, occupancy, purpose, channel, property type,
  state, term, units and month is 0.117 (sd(T) about 15 points). The latent part therefore
  moves by about one 20-point fee band, and only neighbouring fees are supported, as for
  Lending Club.
- **Not run.** The fee is not in the data; the historical grids (Freddie Mac Exhibit 19,
  2008-2026) would have to be rebuilt by hand. The one policy-relevant use is the 2023 grid
  change (two schedules for comparable borrowers), which has about two years of performance.
- Layout note: the sample origination files have 31 fields (no servicer name); see `ORIG` in
  `screen.py`.

## Getting the data (author)

Registration needs an account, which Claude cannot create.

1. Register at https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset.
2. Download the **sample** files (`sample_orig_YYYY.txt`, `sample_svcg_YYYY.txt`) for a
   range of origination years, for example 2012-2019.
3. Save them into `data/raw/`.

The full dataset is tens of GB. Start with the samples.

## Caveats

- Credit scores are reported exactly, but the pricing grid changed over time.
  Record the grid for each year.
- Freddie Mac buys only loans that meet its rules, so the sample is selected on the
  scores near some cutoffs (for example, very low credit scores).
