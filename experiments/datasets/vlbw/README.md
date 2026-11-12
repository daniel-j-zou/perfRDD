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

With the policy set widened to 1200-2200 g (`--outcomes wide`) the logit effect crosses zero
at about 1870 g on its own, so c = 0 gives an interior optimum (1871 g); a cost calibrated from
Almond et al. ($4,000 per infant, c about 0.0005 survival points) gives 1855 g. The sign
change above 1870 g is the model's extrapolation across X (no treated infant weighs over
1500 g). Table in the results note.

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

## Early cohorts 1983-1991 (2026-11-12)

Almond, Doyle, Kowalski and Williams (2010) used the 1983-1991 and 1995-2002 cohorts. The
1983-1991 files are added to `build(year)`; `YEARS` stays 1995-2002 (so `sample_frame`, `load`
and `mirrored` default and `years=YEARS` behave as before) and `EARLY_YEARS = 1983..1991` lists
the new ones, e.g. `sample_frame(years=EARLY_YEARS)` or `years=EARLY_YEARS + YEARS`.

**Files** (same host and permission as above; one `curl` per file, a few seconds apart,
sizes equal the server listing):

```
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/cohortlinkedus/LinkCO<yy>.zip      (yy = 83 ... 91; no "US" in the name)
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/cohortlinked/LinkCO<yy>Guide.pdf
```

Zips: 83 48,955,671 B; 84 49,320,426; 85 54,563,447; 86 54,461,943; 87 55,313,327; 88
57,798,895; 89 112,315,101; 90 116,658,959; 91 116,010,478. Each holds `LinkCO<yy>USden.dat`
(denominator) and `LinkCO<yy>USnum.dat` (linked deaths); they stay zipped in `data/raw/`
(about 0.74 GB with the guides and `Guide.txt` from `pdftotext -layout`) and `build` streams the
`den` member in 250,000-record chunks, caching `data/processed/vlbw_<year>.parquet` (about 0.27
GB for the nine). The guides are scanned images with OCR text for 1983-1988, so some digits and
code labels in `Guide.txt` are garbled; the layouts below were checked against the data.

**Layouts** (1-based inclusive; `LAYOUTS[year]` in `adapter.py`).
- 1983-1988 (1978 certificate): record 91 bytes plus CRLF (93). Match status 1; resident
  status 11; state of residence (NCHS) 27-28; sex 38; gestation 39-40; birth weight 43-46;
  plurality 50; five-minute Apgar 53-54; Hispanic origin or ancestry 55-56; mother's detail race
  57; mother's age 58-59; education (years) 62-63; marital status 65; mother's place of birth
  66-67; father's age 71-72; month prenatal care began 78-79; prenatal visits 81-82; total birth
  order 83-84; live birth order 86-87; record weight 91 (1983-84 only).
- 1989-1991 (1989 revision): denominator-plus record 225 bytes plus CRLF (227), the 1995-2002
  layout with these positions: match status 1; resident status 11; state 24-25; mother's age
  30-31; Hispanic origin 33; `MRACE3` 38; education 39-40; marital status 43; mother's birth
  place 44-45; total and live birth order 47-48 and 49-50; month prenatal care began 51-52;
  visits 54-55; father's age 61-62; gestation flag 72; gestation 73-74; sex 78; birth weight
  79-82; plurality 87; five-minute Apgar 90-91; tobacco 118; cigarettes 119-120; weight gain
  126-127; ventilation under 30 min 160 and over 30 min 161; age at death in days 213-215.
  Identical in the three guides (checked against the data).

**Code mappings to the 1995-2002 coding.**
- `death`: match status 1 (matched) or 2 (late-filed matched), 3 survivor. 1995-2002 uses
  code 1 only; in 1983-1991 the count of codes 1 plus 2 equals the guide's numerator record
  count exactly in 1986-1988, 1990 and 1991 (and to 33 or fewer in 1983-1985, 1989).
- Sex 1/2, plurality, mother's age, education years 00-17 and 99, marital status 1/2,
  month of care and visits, birth orders, father's age and weight in grams (9999 unknown) are
  the same codes. Plurality is 1-3 in 1983-1988 (3 is every higher multiple) and 1-5 later.
- `ormoth` (1983-1988): 00 and ancestry codes 06-24 become 0 (non-Hispanic), 01-05 stay,
  88 (not reported) and 99 become 9. About 44% (1983-84) to 30% (1988) of records are 9 because
  many states did not ask; Hispanic share is therefore understated before 1989.
- `mrace3` (1983-1988): from detail race of mother, 1 white -> 1, 2 black -> 3, all other
  codes 0, 3-8 -> 2, 9 unknown -> NaN (about 0.3%). 1988 uses the imputed-race item at 91
  when position 57 is 9. 1989-1991 reads `MRACE3` directly.
- Weights and gestation: no imputation flag for weight before 1995 (`weight_imputed` False);
  `gest_imputed` is False in 1983-1988 (gestation 99 is not stated; values 48-52 occur and are
  dropped by `sample_frame`) and, for 1989-1991, true when the gestation flag at 72 is 1 or 2
  (clinical estimate or imputed), as for 1995-2002.
- `recwt`: the record weight at 91 in 1983-1984 (1 or 2: non-VSCP states are a 50% sample, with
  linked deaths added at weight 1); 1 in 1985-1991 (full files; 1988 position 91 is the imputed
  race, one 1987 record has a stray 6). The births of a cell are the sum of `recwt`; deaths are
  counted once (the guide's tables do this: 1983 under 2,500 g, 247,818 births = sum of
  `recwt` and 23,190 deaths against the guide's 23,163 by residence; 1000-1249 g 10,193 births).
  Rates below use deaths over the sum of `recwt`; local-linear weights are 1 for a death and
  `recwt` for a survivor.
- `state` is the NCHS state-of-residence code (27-28 / 24-25).

**Covariates by year** (share of records with a usable value; X_COLS flags the rest).

| Field | 1983-85 | 1986-88 | 1989 | 1990 | 1991 | 1995 |
|---|---|---|---|---|---|---|
| gestation 17-47 | .952-.956 | .952-.954 | .986 | .989 | .990 | .991 |
| Hispanic origin | .56-.60 | .60-.70 | .95 | .98 | .99 | .99 |
| education | .81 / .81 / .76 | .76 / .76 / .72 | .91 | .93 | .97 | .99 |
| prenatal visits | .91 / .91 / .85 | .85 / .85 / .84 | .97 | .97 | .97 | .97 |
| five-minute Apgar | .81 / .81 / .76 | .76 / .76 / .76 | .76 | .76 | .77 | .77 |
| father's age | .87 | .85-.86 | .84 | .84 | .83 | .85 |
| tobacco, cigarettes | absent | absent | .67 | .72 | .73 | .79 |
| weight gain | absent | absent | .68 | .73 | .74 | .78 |
| assisted ventilation | absent | absent | .83 | .88 | .88 | .95 |
| age at death (days) | absent | absent | .010 | .009 | .009 | .007 |

(Triples are 1983 / 1984 / 1985 and 1986 / 1987 / 1988.) Absent fields are NaN columns:
tobacco, cigarettes, weight gain and ventilation do not exist before 1989, and the 1983-1988
denominator has no death items (no `aged`; the numerator file has only age-at-death recodes, and
there is no key to merge it), so `X_COLS` smoker, weight gain flags are all missing and
`ventilation` is 0 in `load` for those years. Pooled work on 1983-1991 should use only the
covariates present in every year, or the missing-value flags.

**Verification** (`records` = denominator records, matched to each guide's record count; births =
sum of `recwt`; res. = excluding foreign residents; deaths = codes 1+2; IMR per 1,000 births):

| Year | Records | Births | Guide births | Deaths | Numerator file | IMR | Res. births | Guide (res. births / deaths) |
|---|---|---|---|---|---|---|---|---|
| 1983 | 3,341,274 | 3,643,001 | 3,643,001 | 39,737 | 39,704 | 10.91 | 3,639,113 | 3,639,113 / 39,683 (ours 39,716) |
| 1984 | 3,364,090 | 3,673,694 | 3,673,693 | 38,306 | 38,314 | 10.43 | 3,669,268 | 3,669,268 / 38,292 (ours 38,284) |
| 1985 | 3,765,336 | 3,765,336 | 3,765,336 | 39,171 | 39,170 | 10.40 | 3,760,833 | table is a subgroup |
| 1986 | 3,760,997 | 3,760,997 | 3,760,997 | 37,966 | 37,966 | 10.09 | 3,756,849 | table is a subgroup |
| 1987 | 3,813,492 | 3,813,492 | 3,813,492 | 37,349 | 37,349 | 9.79 | 3,809,670 | 3,809,670 / 37,328 |
| 1988 | 3,913,967 | 3,913,967 | 3,913,967 | 37,599 | 37,599 | 9.61 | 3,909,684 | 3,909,684 / 37,576 |
| 1989 | 4,045,881 | 4,045,881 | n/a | 38,606 | 38,605 | 9.54 | 4,041,146 | table is a subgroup |
| 1990 | 4,163,150 | 4,163,150 | n/a | 37,066 | 37,066 | 8.90 | 4,158,445 | 4,150,445 (OCR) / 37,039 |
| 1991 | 4,115,494 | 4,115,494 | n/a | 35,520 | 35,520 | 8.63 | 4,111,059 | 4,111,059 / 35,496 |

| Year | Weight reported | Share under 1500 g (reported) | Births under 1500 g (records) | Death rate under 1500 g | Mean weight (g) | Mean gestation (wk) |
|---|---|---|---|---|---|---|
| 1983 | .9985 | .0119 | 41,006 | .394 | 3344.3 | 39.33 |
| 1984 | .9986 | .0119 | 41,276 | .383 | 3350.4 | 39.35 |
| 1985 | .9987 | .0122 | 45,764 | .381 | 3350.3 | 39.28 |
| 1986 | .9986 | .0122 | 45,644 | .365 | 3348.4 | 39.24 |
| 1987 | .9987 | .0124 | 47,292 | .351 | 3345.7 | 39.21 |
| 1988 | .9989 | .0125 | 48,725 | .343 | 3345.8 | 39.21 |
| 1989 | .9987 | .0129 | 51,940 | .336 | 3341.7 | 39.15 |
| 1990 | .9987 | .0128 | 53,080 | .310 | 3342.6 | 39.12 |
| 1991 | .9988 | .0129 | 53,129 | .298 | 3335.2 | 39.07 |

(1995 for comparison: IMR 7.51, 1.37% under 1500 g, mean weight 3326.) The IMRs equal the
guides' 10.9, 10.4, 9.8, 9.6, 8.9 and 8.6 for the all-races tables of 1983, 1984, 1987, 1988,
1990, 1991. The 1983 and 1985 numerator counts differ from codes 1+2 by 33 and 1, 1984 by 8.

**Almond et al. sample idea** (`sample_frame((1000, 2000), years=EARLY_YEARS)`: reported weight,
known non-imputed gestation, singleton or multiple; deaths over `recwt`):

| Year | Births 1000-2000 g | Records 1400-1499 / 1500-1599 | Death rate 1400-1499 g | 1500-1599 g | Local jump h=50 (se) |
|---|---|---|---|---|---|
| 1983 | 59,699 | 4,315 / 6,143 | .0959 | .0784 | -.033 (.031) |
| 1984 | 59,449 | 4,268 / 6,077 | .0885 | .0758 | -.026 (.032) |
| 1985 | 66,252 | 4,870 / 6,700 | .0858 | .0845 | -.040 (.030) |
| 1986 | 66,566 | 4,870 / 6,742 | .0776 | .0764 | -.024 (.026) |
| 1987 | 67,664 | 5,002 / 6,828 | .0762 | .0704 | -.033 (.022) |
| 1988 | 70,113 | 5,212 / 7,070 | .0814 | .0661 | +.008 (.025) |
| 1989 | 60,998 | 4,618 / 5,949 | .0771 | .0592 | -.059 (.018) |
| 1990 | 62,631 | 4,666 / 6,204 | .0619 | .0574 | -.053 (.016) |
| 1991 | 64,044 | 4,720 / 6,347 | .0640 | .0510 | +.004 (.022) |

Pooled 1983-1991: 577,416 births of 1000-2000 g; death rate 7.87% at 1400-1499 g and 6.91% at
1500-1599 g. Local linear one-year death on `1{weight < 1500}` (x = weight - 1499.5, separate
slopes, robust HC1 standard error, deaths weight 1, survivors `recwt`): jump
**-3.17 pp (se 0.79, t = -4.0)** with the triangular kernel at h = 50 g (n = 45,366 births),
-2.83 pp (0.66) uniform; with year dummies -2.87 pp (0.79) and -2.78 pp (0.66);
h = 100 g: -1.46 pp (0.38) triangular, -1.21 pp (0.35) uniform. The sign is the expected one
(lower death just below 1500 g). The estimate is larger than the 2002 one in the first-results
section (-1.5 pp at h = 50) and than Almond et al.'s (about -1 pp at 1 year); weights heap at
multiples of 100 g and 50 g, 1500 g itself among them, so the 50 g window is sensitive to the
heaping (not examined here). Single-year jumps are noisy.

**Build**:

```
cd code
python -c "from experiments.datasets.vlbw import adapter as A; [A.build(y) for y in A.EARLY_YEARS]"
```

(about 15 s per year; needs only `data/raw/LinkCO<yy>.zip`; `force=True` rebuilds.)

**For the threshold runs (Claude, 2026-11-12).** The routes script takes no record weights, so
its `long`, `longwide` and `early` variants use 1985-1991 only: unweighted, 1983 and 1984
overstate infant death in the 500-2500 g range (8.5% against 7.9% weighted in 1983). Birth
weights are mostly ounce conversions (about 10,000 births each at 1446, 1474, 1503, 1531 g
against 20 to 100 at other gram values), and the round-gram heaps carry higher mortality
(exactly 1500 g: 9.9% against 5.8% at 1490-1499 g in 1985-1991). The `_d100` variants drop
every birth recorded at a multiple of 100 g. Local jumps with and without donuts:
`experiments/scripts/vlbw_heaping_check.py`, output in `experiments/runs/vlbw_heaping_check.txt`.
