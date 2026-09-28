# Chile grade retention — MINEDUC open student-performance files (data only; no adapter yet)

| Field | Value |
|---|---|
| **Q** | `PROM_GRAL`, annual general grade average (1.0-7.0, reported on a 0.1 grid; 0.0 for withdrawn students) |
| **Rule** | repeat the year if 1 failed subject (< 4.0) and average < 4.5, or 2 failed subjects and average < 5.0; attendance < 85% can also retain |
| **Treatment** | `SIT_FIN == "R"` (reprobado); `P` promoted, `Y` withdrawn |
| **X (available)** | prior-year `PROM_GRAL`/`ASISTENCIA`/`SIT_FIN` via `MRUN`, grade (`COD_GRADO`, `COD_ENSE`), gender, birth year-month, age, school (`RBD`), dependency, rural, commune |
| **Not in public file** | subject grades / number of failed subjects, so the design is fuzzy in `PROM_GRAL` |
| **n** | about 3.3 million student records per year, 2002-2025 |
| **Source** | https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/ (public, no registration) |

## Getting the data

One RAR per year, 2002-2025: 37-48 MB each, about 1.06 GB compressed in total, and
about 11 GB of CSV once extracted. Download and extract with the idempotent script
(uses the macOS `bsdtar`, libarchive >= 3.4):

```bash
python -m experiments.datasets.chile_retention.download            # all years
python -m experiments.datasets.chile_retention.download 2017 2018  # a subset
```

The script checks each file against the server's size before extracting. All 24 years
were downloaded on 2026-09-28. Reported record counts match the ministry schema
(`ER Rendimiento por alumno, bases Web.pdf`, inside each archive) for every year it
lists (2002-2020).

Each archive contains the CSV (`;`-separated, decimal comma), the variable schema, and
(in most years) a usage-recommendations PDF and a frequency workbook.

Parsing notes:
- Encodings vary: ASCII, UTF-8 with or without BOM, and Latin-1 (2015). Header case also
  varies. Read with `encoding="latin-1"` and normalize headers by stripping the BOM and
  uppercasing (see `experiments/scripts/chile_retention_profile.py`).
- `MRUN` repeats within a year for students who transferred (about 209k duplicate rows
  in 2018; see `SIT_FIN_R == "T"`).
- Filter `ESTADO_ESTAB == 1` for operating schools (the variable exists from 2015).

## First look (2018; `SIT_FIN` in {P, R} and `PROM_GRAL > 0`; n = 2,998,611)

Share retained by grade average:

| Average | 4.3 | 4.4 | 4.5 | 4.6 | 4.9 | 5.0 | 5.1 |
|---|---|---|---|---|---|---|---|
| Share retained | 0.804 | 0.731 | 0.355 | 0.295 | 0.094 | 0.021 | 0.012 |
| Count | 10,860 | 13,315 | 26,425 | 31,726 | 70,254 | 91,550 | 104,117 |

- There are clear first-stage drops at 4.5 (about 38 points) and 5.0 (about 7 points).
- The count doubles from 4.4 to 4.5, far above the neighboring trend. That is heaping at
  the cutoff, which suggests grade nudging and/or rounding. It must be checked before any
  RD use; it is also relevant to the performative-threshold direction.
- Linkage: 90.6% of 2017 promoted or retained students appear in 2018; the rest include
  graduates and leavers. 82.7% of retained students are in the same grade next year, and
  0% of promoted students are.

## All years (`python experiments/scripts/chile_retention_profile.py OUT.csv`)

Heap = count at the cutoff / geometric mean of the counts 0.1 below and above it. Sample:
`SIT_FIN` in {P, R}, `PROM_GRAL > 0`.

| Year | Records | Retained | Heap at 4.5 | Retained 4.4 → 4.5 | Heap at 5.0 | Retained 4.9 → 5.0 |
|---|---|---|---|---|---|---|
| 2002 | 3,376,045 | 140,769 | 1.41 | 0.55 → 0.24 | 1.10 | 0.050 → 0.014 |
| 2003 | 3,528,762 | 181,531 | 1.43 | 0.63 → 0.27 | 1.11 | 0.055 → 0.014 |
| 2004 | 3,515,838 | 188,363 | 1.43 | 0.64 → 0.26 | 1.10 | 0.053 → 0.012 |
| 2005 | 3,506,407 | 199,293 | 1.45 | 0.64 → 0.26 | 1.10 | 0.054 → 0.012 |
| 2006 | 3,465,584 | 213,661 | 1.41 | 0.66 → 0.28 | 1.11 | 0.056 → 0.013 |
| 2007 | 3,413,110 | 200,135 | 1.41 | 0.66 → 0.27 | 1.10 | 0.054 → 0.012 |
| 2008 | 3,356,256 | 202,512 | 1.39 | 0.69 → 0.29 | 1.09 | 0.061 → 0.014 |
| 2009 | 3,326,575 | 186,466 | 1.39 | 0.68 → 0.29 | 1.09 | 0.060 → 0.012 |
| 2010 | 3,335,825 | 187,724 | 1.40 | 0.71 → 0.30 | 1.08 | 0.065 → 0.012 |
| 2011 | 3,326,746 | 221,422 | 1.40 | 0.72 → 0.31 | 1.08 | 0.072 → 0.019 |
| 2012 | 3,308,477 | 177,758 | 1.37 | 0.71 → 0.30 | 1.07 | 0.062 → 0.013 |
| 2013 | 3,255,518 | 168,382 | 1.38 | 0.72 → 0.31 | 1.08 | 0.073 → 0.016 |
| 2014 | 3,227,534 | 152,919 | 1.35 | 0.72 → 0.33 | 1.07 | 0.073 → 0.016 |
| 2015 | 3,238,586 | 152,359 | 1.33 | 0.73 → 0.35 | 1.09 | 0.080 → 0.018 |
| 2016 | 3,226,943 | 144,895 | 1.32 | 0.75 → 0.36 | 1.08 | 0.086 → 0.019 |
| 2017 | 3,246,824 | 130,510 | 1.30 | 0.74 → 0.36 | 1.08 | 0.092 → 0.022 |
| 2018 | 3,293,750 | 121,004 | 1.29 | 0.73 → 0.36 | 1.07 | 0.094 → 0.021 |
| 2019 | 3,328,915 | 98,269 | 1.28 | 0.69 → 0.33 | 1.09 | 0.084 → 0.019 |
| 2020 | 3,164,534 | 58,664 | 1.35 | 0.03 → 0.01 | 1.16 | 0.003 → 0.001 |
| 2021 | 3,237,043 | 89,557 | 1.51 | 0.20 → 0.07 | 1.07 | 0.016 → 0.004 |
| 2022 | 3,405,130 | 79,746 | 1.25 | 0.59 → 0.31 | 1.10 | 0.084 → 0.019 |
| 2023 | 3,584,330 | 81,660 | 1.25 | 0.62 → 0.34 | 1.11 | 0.083 → 0.018 |
| 2024 | 3,568,930 | 77,281 | 1.23 | 0.64 → 0.35 | 1.12 | 0.086 → 0.018 |
| 2025 | 3,540,980 | 75,620 | 1.21 | 0.64 → 0.35 | 1.11 | 0.085 → 0.017 |

- **Heaping at 4.5 in every year:** 21-51% more students than the neighboring counts
  imply, and about 7-16% at 5.0. It is a stable feature of grading, not a one-year
  artifact.
- **The rule's bite changes over time.** The retention jump at 4.5 is about 0.3-0.4
  through 2019. The 2020 and 2021 school years (COVID) nearly suspended retention: only
  2.5% retained at 4.4 in 2020. From 2022 the jump returns, but retention at 4.4 stays
  lower than in 2010-2019 (0.59-0.64 vs 0.69-0.75). That fits schools having more
  discretion over promotion from about 2019 (the rule change itself is not verified here).
  The pre-2019 years are the cleanest for a fixed rule.
- Withdrawn (`Y`) counts rise from about 190k to about 300k around 2010-2011, possibly a
  change in how withdrawals are recorded. Check this before using withdrawal as an outcome.
- `MRUN` is never missing, and every average is on the 0.1 grid.
