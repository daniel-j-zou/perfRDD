# Chile: the PSU 475 line for the state-guaranteed loan (CAE) and scholarships

| Field | Value |
|---|---|
| **Q** | PSU reading/mathematics average (100-850), the admission-test score of the year |
| **Threshold** | 475 points, the academic requirement for the state-guaranteed loan (CAE, law 20.027 of 2006) and most MINEDUC scholarships; a legislative choice |
| **Treatment** | `1{Q >= 475}`: eligibility for the loan and the scholarships (ITT; take-up from the benefits files if wanted) |
| **X** | NEM (school grades), ranking score, school type and dependency, region, family income bracket, parental education, sex, year of graduation (the `chile_admission/` covariates, from the DEMRE inscription and socioeconomic files) |
| **Y** | enrolled in a university the next year (binary; about 15% just below the line and 35% just above, Solis 2017); enrolled in year 2 (persistence); any higher-education enrollment |
| **n** | about 200-250k test takers per admission process; the line sits near the median, so the window is dense |
| **Effect** | Solis (JPE 2017, "Credit access and college enrollment"): enrollment jumps by nearly 20 points at 475 and persists to years 2 and 3; Card and Solis (NBER w27269) on retakers and persistence |
| **Cost** | the loan subsidy or scholarship per eligible student (a free parameter for the sweep; DIPRES publishes program costs) |
| **Status** | waiting for the author's download (2026-11-11); no adapter yet |

## Why this one

Every criterion of `../BINARY_OUTCOME_DATASETS_20261111.md` is plausible: a threshold someone
chose, a documented large effect of money, a binary outcome with a 15-35% base rate, score
covariates that explain the test (R^2 about 0.45 for PAES in `chile_admission/`), public
unit-level data, and an effect that should fade for students who would enroll anyway. The
`chile_admission/` adapter (PAES 2024, threshold 458) is the template: the new work is the
475 rule, the PSU years and the loan take-up variable.

## Getting the data (author; the hosts refuse scripted reads)

1. DEMRE open data, <https://demre.cl/portales/portal-bases-datos> (usage guide:
   <https://portaldemre.demre.cl/portales/datos-abiertos/documentos/documento-uso-datos-20241128.pdf>).
   For each admission process from 2007 to 2022 (PSU years; 2023 onward is PAES, whose
   scale restates the rule), download the "Inscritos y puntajes" (inscription and scores)
   and "Socioeconómico" files, csv, linked by the anonymous `ID_aux`. Put each year in
   `data/raw/<year>/` keeping the file names. Start with 2012-2016 (five years, about 1.2M
   test takers) if the full set is too much.
2. MINEDUC Datos Abiertos, higher-education enrollment files ("Matrícula en Educación
   Superior") for the year after each admission process, from
   <https://datosabiertos.mineduc.cl/matricula-en-educacion-superior/>; these carry `mrun`.
   The DEMRE open files carry `ID_aux`, not `mrun`; the identified DEMRE files (with `mrun`)
   are requested by email through the same portal. Either request them, or download the
   MINEDUC PSU files (2021-2022 PDT, 2023 onward PAES, which carry `mrun`) from
   <https://datosabiertos.mineduc.cl/pruebas-de-admision-a-la-educacion-superior/> and start
   with those years, linked exactly as `chile_admission/adapter.py` does.
3. Optional: the MINEDUC student-benefit files ("Asignación de becas y créditos") give loan
   and scholarship take-up by `mrun` for a fuzzy design.

Both hosts refused automated fetches on 2026-11-11 (connection refused), and the MINEDUC
host was down on 2026-10-29. Check `robots.txt` before any scripted retry.

## Plan for the adapter

`load()` at the test-taker level with Q = the PSU average, `threshold=475`,
`treatment_rule = Q >= 475`, Y = university enrollment the next year, X as in
`chile_admission/` (reuse `X_COLS`); loaders `load_any()`, `load_year2()`. Policy set about
420-560 points, eps 0.1, effect covariate NEM (and the income bracket). Run with
`experiments/scripts/binary_outcome_routes.py` through `build_sample`.
