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
