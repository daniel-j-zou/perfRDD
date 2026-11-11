# Very low birth weight (1500 g): NCHS 1995-2002 birth-cohort linked birth/infant-death files

| Field | Value |
|---|---|
| **Q** | birth weight in grams (`DBIRWT`, reported; imputed weights dropped) |
| **Threshold** | 1500 g, the very-low-birth-weight line below which infants receive more intensive care (Almond, Doyle, Kowalski and Williams 2010, QJE) |
| **Treatment** | `1{Q < 1500}` (treat below; the estimator scripts mirror the score) |
| **X** | gestation (weeks, and squared), plurality, sex, mother's age (and squared), Hispanic origin, race, education, marital status, prenatal care month and visits, birth order, tobacco and cigarettes, weight gain, father's age; missing-value flags for education, prenatal care, tobacco, weight gain and father's age |
| **Y** | infant death within one year (matched death record), binary; about 0.7% overall, 5-10% near 1500 g |
| **n** | 4,027,475 US births in 2002, 31.7 million over 1995-2002; the sample keeps reported weights in [500, 5000] g with a known, non-imputed gestation (`years=` selects the cohorts; the default is 2002 alone) |
| **Source** | NCHS Vital Statistics Online, public use, no registration |
| **Extras** | `ventilation` (assisted ventilation under or over 30 minutes), `gestat`, `plural` |

## Why 2002 and not a recent cohort

The cohort files from 2003 on (revised birth certificate) suppress birth weight in grams in
the public-use linked file: in `LinkCO14Guide.pdf` positions 504-508 are a filler and only
the 14-category recode `BWTR14` remains, whose lowest categories are 227-499, 500-749,
750-999, 1000-1249, 1250-1499 g. The 1995-2002 cohort files report grams (`DBIRWT`,
positions 81-84). Almond et al. used 1983-2002. The 1995-2001 files share the 2002 layout
(record length 232 with CRLF in every file; the year at 7-10, birth weight at 81-84, gestation
at 74-75, sex at 79 and age at death at 211-213 were checked on each file on 2026-11-11), so
`build(year)` reads all eight with the same `LAYOUT`.

## Getting the data

```
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/cohortlinkedus/LinkCO<yy>US.zip      (90-108 MB each, yy = 95 ... 02)
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/cohortlinked/LinkCO<yy>Guide.pdf
```

Each zip holds `LinkCO<yy>USDen.dat` (the denominator-plus file: every birth, with the
death items of the matched death record appended; about 4 million records, width 230,
CRLF), `LinkCO<yy>USNum.dat` (the linked deaths, width 535) and a small unlinked-deaths
file. Put them in `data/raw/`; the adapter reads only the denominator-plus files (the 2002
names are upper-case `DEN`/`NUM`). `build(year)` parses one file in about a minute and
caches `data/processed/vlbw_<year>.parquet`. All eight were downloaded with `curl` on
2026-11-11 (2002 first, 1995-2001 later the same day) and extracted with Python's `zipfile`
(`unzip -q` extracted nothing).

`ftp.cdc.gov/robots.txt` is `Disallow: /`. These are public-use files published for
download, and the author instructed the download ("you download", 2026-11-11), so it
was fetched directly with `curl`; no crawling. Record this if the files are refetched.

## Layout (from the user guide; 1-based inclusive positions)

`MATCHS` 1 (1 matched death, 2 surviving, 3 unmatched death), `BIRYR` 7-10, `RESSTATB` 11,
`BRSTATE` 12-13, `DMAGE` 30-31, `ORMOTH` 33 (0 non-Hispanic, 1-5 Hispanic), `MRACE` 36-37,
`MRACE3` 38 (1 white, 2 other, 3 black), `DMEDUC` 39-40 (00-17 years, 99 unknown), `DMAR` 43
(1 married, 2 unmarried, 9 unknown), `MPLBIR` 44-45, `DTOTORD` 47-48, `DLIVORD` 49-50,
`MONPRE` 51-52 (00 none, 01-09 month, 99 unknown), `NPREVIST` 54-55, `DFAGE` 61-62,
`GESTIMP` 73, `GESTAT` 74-75 (17-47, 99 unknown), `CSEX` 79, `BWIF` 80 (1 imputed),
`DBIRWT` 81-84 (0227-8165, 9999 unknown), `DPLURAL` 89, `FMAPS` 90-91, `TOBACCO` 118,
`CIGAR` 119-120, `WTGAIN` 126-127, `VENL30` 160, `VEN30M` 161, `AGED` 211-213 (days),
`RECWT` 223-230.

## First results (2026-11-11)

`python -m experiments.scripts.binary_outcome_routes --settings vlbw --cost c` on the
500-2500 g sample (n = 285,610; 16.8% below 1500 g; survival 96.0%), outcome
**survival** (1 - death), effect covariate gestation, eps = 0.1, policy set 1200-1800 g.
Outputs in the git-ignored `experiments/runs/binary_outcome_routes_vlbw/`; the reading is
in `../BINARY_OUTCOME_ROUTES_20261110.md`.

- Local check at 1500 g (local linear, triangular kernel): death jumps by -1.5 pp (se 1.2)
  at h = 50 g and -0.7 pp (0.7) at h = 100 g below the line; ventilation shows no clear
  jump in this one-year file.
- The logit routes give a marginal survival gain that grows as the threshold falls:
  +0.4 pp at 1800 g, +2.1 pp at 1500 g, +7.4 pp at 1200 g. With c = 0 the optimum is the
  top of the policy set (treat every infant under 1800 g); with a cost c per treated
  infant in survival-probability units the optimum is interior: 1653 g at c = 0.01,
  1508 g at c = 0.02, 1425 g at c = 0.03. So the deployed 1500 g line is optimal for a
  cost of about 2 pp of survival per treated infant. Trim level: 1452 g at eps = 0.05,
  1550 g at eps = 0.15 (c = 0.02).
- The linear probability model gives +23 pp at 1800 g falling to -0.6 pp at 1200 g, the
  wrong way, and a boundary at every cost; the additive-effect route is negative
  throughout. Survival is 96% on average, so a probability effect additive in the
  covariates extrapolates badly here, and the link is what makes the setting work.
- The 500-5000 g sample (`variant="all"`) leaves no supported window with these design
  rules (1.3% treated); the low-birth-weight sample is the one to use.
- Caveats: the 1500 g rule is a guideline (fuzzy); $X\perp\eta$ fails in spread
  (sd of the residual 1.74 across deciles of the linear predictor, larger at low
  gestation) and in mean (range -63 to +112 g); one year of data; no driver of the
  cost c in the file.

## Pooled 1995-2002 (2026-11-11)

`sample_frame(weights, years=YEARS)` pools the eight cohorts: 2,138,145 births of 500-2500 g
(17.0% under 1500 g, survival 95.8%). The routes script runs it as `--settings vlbw --outcomes
pooled`. The 2002 result replicates: under the logit the survival gain at the threshold is
+0.2 pp at 1800 g, +2.5 at 1500 g, +8.7 at 1200 g; the optimum is 1660-1663 g at c = 0.01,
1545 g at 0.02 and 1456-1459 g at 0.03 (both logit routes within 3 g), so the 1500 g line is
optimal for a cost near 0.025. The linear probability model stays at the 1800 g edge at every
cost (+24 pp at 1800 g, falling to zero at 1200 g); the additive route is negative throughout.
Bootstrap (`experiments/scripts/binary_outcome_vlbw_boot.py --variant pooled --costs 0.02`,
20 draws): 1544 g, sd 8 g, [1529, 1555] g.
Full table in `../BINARY_OUTCOME_ROUTES_20261110.md`.

## Design notes

- The rule treats below the threshold. `adapter.mirrored()` returns `Q = -weight` with
  `phi0 = -1499.5`, so that the four-role scripts' convention (treated iff Q >= phi0)
  applies; weights are integers, so this is `weight < 1500`.
- The treatment is a guideline, not a statute: Almond et al. document discontinuous
  jumps in hospital charges, ventilation and length of stay at 1500 g. Ventilation is in
  the file (`ventilation` extra) and serves as the first-stage check.
- Gestation is the main score covariate and also the natural effect covariate (the DX
  block): how much extra care helps should depend on maturity, not only on weight.
- The policy question is where the extra-care line should sit, with a per-infant cost of
  care as c (the paper's c in units of survival probability per treated infant).
