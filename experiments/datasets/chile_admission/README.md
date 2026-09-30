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

## Continuous outcomes (2026-09-29)

`build_continuous()` has the same sample and X, plus outcomes from the 2024/2025
enrollment files:
- `sel_2024` / `sel_2025`: selectivity of the enrolled program, the leave-one-out mean Q
  of the program's first-year 2024 enrollees (programs with at least 10 such peers).
  Enrolled students only.
- `acred_2024` / `acred_2025`: the institution's accreditation years, 0 if not enrolled.
- `duration_2024`: program duration in semesters, 0 if not enrolled.

Outputs: `outputs/screen_chile_admission_continuous_20260929/`. "At cutoff" is the models'
mean fitted effect for |Q - 458| < 5.

| Y | Local ITT (SE) | By NEM tercile | alpha-only at cutoff | DS at cutoff | DS optimum [share of window] |
|---|---|---|---|---|---|
| sel_2024 | +9.8 (2.0) | +7.4 / +14.2 / +6.9 | -13.4 | -7.5 | 470 [69%] |
| sel_2025 | +4.6 (1.8) | +3.0 / +8.7 / +1.7 | -15.3 | -9.7 | 482 [63%] |
| acred_2024 | +0.155 (0.074) | +0.14 / +0.03 / +0.32 | -0.07 | -0.00 | 455 [71%] |
| acred_2025 | -0.162 (0.072) | -0.24 / -0.17 / -0.02 | +0.21 | +0.12 | treat none (bnd) |
| duration_2024 | +0.52 (0.10) | +0.43 / +0.41 / +0.78 | +0.84 | +0.94 | treat all (bnd) |

- alpha-only is a boundary everywhere and gets the sign of the cutoff effect wrong for
  four of the five outcomes.
- DS gives interior optima close to the deployed 458 for selectivity and 2024
  accreditation. Taken at face value, today's cutoff is about right: U at the optimum
  beats the deployed cutoff by 0.07 selectivity points and 0.0001 accreditation years.
- But DS also misses the sign or size at the cutoff (selectivity -7.5 vs +9.8;
  accreditation 2024 about 0 vs +0.16; 2025 +0.12 vs -0.16). Its NEM gradient near the
  cutoff (-21 / -5 / +5 for selectivity) does not match the local one (+7 / +14 / +7).
  None of these optima is validated.
- Substantively, the local estimates show a short-run gain and a longer-run loss.
  Eligibility moves students into more selective, longer, better-accredited programs in
  2024, but by 2025 the accreditation-weighted enrollment falls (-0.16), concentrated in
  low-NEM students. That is a mismatch/dropout pattern.
