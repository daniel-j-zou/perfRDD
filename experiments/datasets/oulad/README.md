# OULAD — Open University Learning Analytics Dataset

| Field | Value |
|---|---|
| **Q** | first TMA (tutor-marked assignment) score in a module-presentation |
| **Threshold** | 40 (UK pass mark) |
| **Treatment** | `1{Q >= 40}` — passed first major assessment |
| **X** | `num_of_prev_attempts`, `studied_credits`, plus ordinal codes for gender / highest_education / imd_band / age_band / disability |
| **Y** | mean score on subsequent TMAs in the same module-presentation |
| **n** | ~21,800 student-module pairs |
| **Source** | https://analyse.kmi.open.ac.uk/open_dataset (UCI mirror used in download script) |
| **Citation** | Kuzilek, Hlosta & Zdrahal (2017), Scientific Data |

## Building

```bash
python -m experiments.datasets.oulad.download
```

Fetches the ~45 MB ZIP from the UCI mirror and extracts the seven CSVs
into `data/raw/`. The adapter joins `studentAssessment + assessments +
studentInfo` on the fly (no separate processed file).

## Notes

- OULAD is primarily a learning-analytics dataset (engagement / VLE
  click logs, etc.), not a textbook RDD setting. The first-TMA / 40
  cutoff is one defensible RDD slice; alternatives include exam-mark
  cutoffs or distinction (70) thresholds.
- Categorical X columns are ordinal-encoded for compatibility with
  numeric-array methods. Per-method one-hot encoding can read the
  original CSVs from `data/raw/`.
- **Date fix (2026-09-24).** `assessments.csv` codes missing dates as `?`, so
  the `date` column loads as text. The adapter used to rank TMAs by that text
  ("117" < "19" < "54"), so for ~76% of enrolments the "first" TMA was not the
  first and the outcome averaged earlier TMAs. Dates are now parsed as numbers.
  Results built on the old adapter (e.g. `manuscript/applications.tex`, first-stage
  R² 0.031) predate the fix.
- `load_rich()` uses predetermined covariates for a stronger first stage:
  one-hot demographics, module-presentation fixed effects, registration date,
  VLE clicks and active-day share before the first TMA's due date, and the mean
  CMA score due before it (with a missing flag). First-stage R² is 0.043 with
  `load()` and 0.178 with `load_rich()`. Prior-achievement covariates exist for
  only 6-12% of enrolments, which caps the attainable R².
