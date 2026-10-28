# Freddie Mac Single-Family Loan-Level Dataset (waiting for the author's download)

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
