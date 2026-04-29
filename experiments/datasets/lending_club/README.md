# Lending Club — public loan-stats archive

| Field | Value |
|---|---|
| **Q** | `dti` (debt-to-income ratio, %) |
| **Threshold** | 30 (LC underwriting trigger; cap was 35–40 in some periods) |
| **Treatment** | `1{Q >= 30}` |
| **X** | loan amount, annual income, delinquencies, open accounts, public records, total accounts, inquiries, plus ordinal codes for term / home_ownership / purpose / verification_status |
| **Y** | originated interest rate, % (`int_rate`) |
| **n** | ~890k after concatenating LoanStats3a..3d (2007–2015) |
| **Source** | https://resources.lendingclub.com (no auth) |

## Why DTI and not FICO?

FICO scores were stripped from the public Lending Club archive long
before LC went private, for compliance reasons. The
`resources.lendingclub.com` files don't include FICO at all. DTI is
the closest "rich score" that:
- Lending Club uses as part of underwriting,
- has a hard policy threshold (LC's published max DTI ranges from
  30 to 40 across vintages),
- is more informative than the demographic X covariates alone.

## Building

```bash
python -m experiments.datasets.lending_club.download           # 3a only (small)
python -m experiments.datasets.lending_club.download --all     # 3a..3d (~150 MB total)
```

The script pulls public ZIPs from `resources.lendingclub.com`,
concatenates them, and writes `data/raw/loans.csv`.

## FICO alternative (Kaggle, requires auth)

For a FICO-based RDD, the third-party Kaggle mirror
`wordsforthewise/lending-club` includes FICO. To use it:

```bash
pip install kaggle
# place ~/.kaggle/kaggle.json (Account -> Create New API Token at kaggle.com)
kaggle datasets download -d wordsforthewise/lending-club --unzip -p data/raw/
mv data/raw/accepted_*.csv data/raw/loans.csv
# then edit adapter.py to set Q_COL='fico_range_low', THRESHOLD=660
```

## Y alternatives

Beyond `int_rate`: realized return = `(total_pymnt - loan_amnt) /
loan_amnt` requires additional processing of payment fields. Loss
given default uses `recoveries`.
