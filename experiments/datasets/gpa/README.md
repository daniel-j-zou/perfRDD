# GPA — academic probation (Lindo, Sanders, Oreopoulos 2010)

| Field | Value |
|---|---|
| **Q** | `dist_from_cut` (first-year GPA distance from probation cutoff) |
| **Threshold** | 0 |
| **Treatment** | `1{Q < 0}` — below the cutoff = on probation |
| **X** | `hsgrade_pct`, `totcredits_year1`, `loc_campus1`, `loc_campus2`, `male`, `bpl_north_america`, `age_at_entry`, `english` |
| **Y** | `nextGPA` |
| **n** | ~44k complete cases |
| **Citation** | Lindo, Sanders, Oreopoulos (2010), AEJ:Applied |

## Data

The processed CSV `Dep_Data/final_processed_data.csv` and the original AEJ
replication package live under `Dep_Data/`. The `.csv` file is git-LFS-tracked
(see the repo-level `.gitattributes`); run `git lfs pull` after cloning.

The original data is governed by the AEJ replication terms; see
`Dep_Data/LICENSE.txt`.

## Layout

- `adapter.py` — exports `load() -> RDDSample`. Sole required entry point.
- `Dep_Data/` — raw + intermediate files (preserved from the original
  replication package layout).
- `general/` — legacy GPA-specific analysis scripts and notebooks. New
  per-dataset analysis should go in `analysis/` or `notes/`.
