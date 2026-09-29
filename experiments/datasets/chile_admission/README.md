# Chile university-application eligibility (PAES 2024)

| Field | Value |
|---|---|
| **Q** | `PROMEDIO_CM_MAX`, best average of the reading and math-1 PAES tests (100-1000) |
| **Threshold** | 458: minimum to apply to universities in the centralized system (top-10% students are exempt, so fuzzy) |
| **Treatment** | `1{Q >= 458}` (eligibility, ITT) |
| **X** | NEM and ranking scores, top-10% flag, public/private school, technical track, non-regular program, female, years since graduation, family-income group (+ missing flag), Santiago region |
| **Y** | university enrollment 2024; any undergraduate enrollment 2025; university enrollment 2025 |
| **n** | 267,341 registrants with a valid score and NEM |
| **Source** | MINEDUC datos abiertos (DEMRE PAES files and Matrícula Educación Superior), public, no registration, linked by `MRUN` |

## Getting the data

Files used (downloaded 2026-09-29 into `data/raw/`, extracted with `bsdtar -xf`):

```
https://datosabiertos.mineduc.cl/wp-content/uploads/2025/10/PAES-2024-Inscritos-Puntajes.rar      (12.9 MB)
https://datosabiertos.mineduc.cl/wp-content/uploads/2025/10/PAES-2024-Socioeconomicos.rar         (3.0 MB)
https://datosabiertos.mineduc.cl/wp-content/uploads/2026/09/Matricula-Ed-Superior-2024.rar        (31.7 MB)
https://datosabiertos.mineduc.cl/wp-content/uploads/2026/09/Matricula-Ed-Superior-2025.rar        (33.6 MB)
https://datosabiertos.mineduc.cl/wp-content/uploads/2025/01/Asignaciones-de-Becas-y-Creditos-2024.rar (3.9 MB; used only to locate cutoffs)
```

Each RAR extracts to a folder of the same name. The adapter caches the merged file in
`data/processed/admission_2024.csv.gz`. Admission files exist on the portal for
2021-2026 (PDT 2021-2022, PAES 2023-2026). The PSU-era files (2004-2020) are held by
DEMRE and are not on the portal.

## Cutoffs seen in the 2024 data

- University enrollment jumps at 458: 10.4% at 457 vs 19.2% at 458.
- The FSCU loan starts at 485.
- The Bicentenario and Juan Gomez Millas scholarships start around 510.
- The state-guaranteed loan (CAE, 485 for universities) is not in the public awards file.
- The bandwidth of 25 in the local checks keeps 485 outside the band.

## Results (2026-09-29)

Commands:
- `local_rd_checks.py experiments.datasets.chile_admission.adapter:load_* OUT --bandwidth 25 --subgroup nem`
- `differing_slopes_screen --datasets chile_admission_*`
- `screen_flatness.py OUT chile_admission_* --near 25`

| Outcome | Local RD ITT at 458 | By NEM tercile (low / mid / high) | alpha-only / DS effect near cutoff | Optimum (both models) |
|---|---|---|---|---|
| University 2024 | +0.077 (0.009) | +0.050 / +0.073 / +0.118 | +0.144 / +0.163 | lowest candidate cutoff (boundary) |
| Any enrollment 2025 | -0.017 (0.012) | -0.040 / -0.017 / +0.017 | +0.095 / +0.071 | lowest candidate cutoff (boundary) |
| University 2025 | +0.052 (0.011) | +0.022 / +0.053 / +0.098 | +0.182 / +0.189 | lowest candidate cutoff (boundary) |

- The cutoff is clean: covariates barely move at 458 (NEM -6.6, SE 3.2; ranking -7.5,
  SE 3.6; nothing else).
- The local effects are heterogeneous in NEM. Persistence (any enrollment in 2025)
  changes sign from -4.0 to +1.7 pp across NEM terciles, a mismatch pattern, though the
  tercile estimates have SEs of about 2 pp.
- The global fits overstate the local effects by 2-4x and get the sign wrong for
  persistence. First-stage R^2 = 0.45. SD(eta) rises from 75 to 107 across deciles of
  T, and mean eta is U-shaped (+23 / -14 / +20).
- Adding cubic NEM/ranking terms and interactions (`*_flex`) flattens mean eta but not
  its SD, and leaves the global effects nearly unchanged (+0.13/+0.15, +0.07/+0.06,
  +0.17/+0.18). So the gap is not covariate nonlinearity. It is consistent with the
  additive outcome model failing for a bounded outcome.
