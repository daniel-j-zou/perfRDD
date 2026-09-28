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

One RAR per year, about 41-46 MB compressed and about 0.5 GB as CSV. The macOS
`bsdtar` (libarchive 3.7) extracts them:

```bash
cd experiments/datasets/chile_retention/data/raw
curl -sSfLO https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/Rendimiento-2018.rar
mkdir -p 2018 && bsdtar -xf Rendimiento-2018.rar -C 2018
```

Years 2002-2020 are at `wp-content/uploads/2021/12/Rendimiento-<year>.rar`. Later years
have their own paths, listed on the portal page.

Local copies as of 2026-09-28 (SHA-256):
- `Rendimiento-2017.rar`: `360a818a810bbab37dde56b1027c27b1935f0208b4cb1e506607ece5d5dd8d97`
- `Rendimiento-2018.rar`: `883329cf4e6fbafe119e0df8b3dec0c26070517c54b224b5de26693d2d72dc58`

Each archive contains the CSV (`;`-separated, decimal comma, CRLF), a variable schema
(`ER Rendimiento por alumno, bases Web.pdf`), a usage-recommendations PDF, and a
frequency workbook.

Parsing notes:
- Column names are lowercase in 2017 and uppercase in 2018, which also has a BOM.
  Normalize with `.lstrip('﻿').upper()`.
- `MRUN` repeats within a year for students who transferred (about 209k duplicate rows
  in 2018; see `SIT_FIN_R == "T"`).
- Filter `ESTADO_ESTAB == 1` for operating schools.

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
