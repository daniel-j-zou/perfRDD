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
| **Status** | PAES-era line (485) built from the files already in `../chile_admission/data/raw` and run on 2026-11-11: no local effect at 485, see below; the PSU-era files (475 line, 2007-2022) still wait for the author's download |

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

Both hosts refused automated fetches early on 2026-11-11 (connection refused), and the
MINEDUC host was down on 2026-10-29. Later on 2026-11-11 the MINEDUC pages answered 200
(the PDT 2021-2022 and PAES 2023-2026 score files and the 2007-2026 enrollment files are
listed at the two URLs above), but `datosabiertos.mineduc.cl/robots.txt` returns 403, so
nothing was fetched by script; the DEMRE portal page answers 200 behind a reCAPTCHA. The
PAES 2023-2025 score files and the 2024-2025 enrollment files are already in
`../chile_admission/data/raw` (browser downloads of 2026-09-29).

## The PAES-era line, built (2026-11-11)

On the PAES scale the loan requirement for universities is an average of 485 in reading
and mathematics-1 (CAE; the FSCU loan starts at the same score; the official page is
<https://portal.beneficiosestudiantiles.cl/print/213>). `load_paes(outcome)` reuses the
`chile_admission` build (PAES 2024 registrants, `PROMEDIO_CM_MAX`, enrollment 2024 and
2025 by mrun) with `threshold=485`; the routes script has it as setting `cae`, policy set
465-560 so that the 458 application line stays outside.

Local linear jumps in university enrollment (bandwidth 10 / 20 / 30 points):

| Line | Y | Jump (se) h=10 | h=20 | h=30 |
|---|---|---|---|---|
| 458 (application) | univ 2024 | +8.2 pp (1.4) | +7.8 (1.0) | +6.9 (0.8) |
| 485 (loan) | univ 2024 | +1.5 pp (1.5) | +2.0 (1.1) | +0.7 (0.9) |
| 485 (loan) | univ 2025 | +3.3 pp (1.6) | +2.2 (1.2) | +1.8 (0.9) |

The loan line no longer moves enrollment: since free tuition (gratuidad, 2016) covers the
lower six income deciles with no score requirement, the 485 line binds only for students
above those deciles who need a loan. The global fits of the routes put the effect at the
line at +14 pp (logit) with a marginal effect that falls from +15 pp at 465 to +10 pp at
560, seven times the local jump: the same global-versus-local gap as at 458 in
`chile_admission`. The optimum sits at the lower boundary for any cost below about 0.10
(`../BINARY_OUTCOME_ROUTES_20261110.md`). The setting is not usable for the side track;
the PSU-era 475 line (Solis 2017, before gratuidad) remains the candidate and needs the
DEMRE files.

## Plan for the adapter (PSU years)

`load()` at the test-taker level with Q = the PSU average, `threshold=475`,
`treatment_rule = Q >= 475`, Y = university enrollment the next year, X as in
`chile_admission/` (reuse `X_COLS`); loaders `load_any()`, `load_year2()`. Policy set about
420-560 points, eps 0.1, effect covariate NEM (and the income bracket). Run with
`experiments/scripts/binary_outcome_routes.py` through `build_sample`.
