# Fannie Mae Single-Family Loan Performance Data (waiting for the author's download)

The same design as `../freddie_mac/`: Q = (credit score, LTV), with a pricing grid of
loan-level price adjustments by bands of both, and mortgage insurance above 80% LTV.
Y = delinquency or default. Use it as a replication of the Freddie Mac sample.

## Getting the data (author)

The site blocks scripts and needs a free account, which Claude cannot create.

1. Register at https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data
   (Data Dynamics).
2. Download a few quarterly acquisition files, for example 2015-2017.
3. Save them into `data/raw/`.
