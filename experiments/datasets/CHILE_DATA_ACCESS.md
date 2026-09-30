# Chile education data: what is open and how to request the rest (Claude, 2026-09-29)

All sources below are student-level and anonymized. The ministry's masked national ID,
`MRUN`, links the MINEDUC files across years and systems.

## Open, no registration (already used in `chile_retention/` and `chile_admission/`)

| Source | Content | Years | ID |
|---|---|---|---|
| MINEDUC Centro de Estudios, [datosabiertos.mineduc.cl](https://datosabiertos.mineduc.cl/) | school performance per student (annual average, attendance, promoted/retained) | 2002-2025 | MRUN |
| same | school enrollment; higher-education enrollment (program, institution, accreditation, tuition, duration); graduates (titulados) | 2007-2026 | MRUN |
| same | PAES/PDT registrants, scores, family-income group, applications, admission-system enrollment | 2021-2026 | MRUN |
| same | scholarship and FSCU loan awards (gratuidad, BBIC, BNM, FSCU, ...) | 2008-2025 | MRUN |
| DEMRE open portal, [portal-transparencia.demre.cl/portal-base-datos](https://portal-transparencia.demre.cl/portal-base-datos) | admission files B (registration), C (scores), D (applications/selection), centralized-system enrollment, program offers | processes 2004/2008-2027 (PSU era included) | `ID_aux`, a DEMRE-only ID. Links the stages within a process, **not** to MRUN |

DEMRE drops records in rare cells (birth year, commune, school, etc. appearing fewer
than 8 times). Its usage guide advises against uses outside the university-admission
context and forbids re-identification.

## By request (someone must send the request; none of this is automatic)

| Data | Where | How | Notes |
|---|---|---|---|
| **PSU-era DEMRE files with MRUN** (2004-2020 scores, linkable to MINEDUC enrollment, retention and graduates) | DEMRE | email solicituddatos@demre.cl with a research project; DEMRE replies with the procedure. Named data (with RUT) need written consent from the individuals | Needed for the PSU-era 475 loan cutoff (Solis 2017 design) with long-run outcomes |
| **SIMCE student-level scores** (grades 2/4/6/8/10) | Agencia de Calidad de la Educación | a public-information request under Ley 20.285 (transparencia pasiva), or OIRS; their form asks for the researcher, institution, files and a short research description | Gives an external test score as Y or X for the retention design |
| **Subject-level grades / number of failed subjects** | MINEDUC (SIGE) | a Ley 20.285 request to MINEDUC, or ask estadisticas@mineduc.cl whether a research extract or agreement is possible | Not confirmed to be releasable. It would make the retention cutoff sharp (the one-failed-subject sample) |
| **CAE (state-guaranteed loan) awards** | Comisión Ingresa | a Ley 20.285 request via [transparencia.ingresa.cl](https://transparencia.ingresa.cl/) | Needed for the loan's first stage at the 485 (PAES) / 475 (PSU) cutoff |
| **Program-level earnings** (mean income 1-5 years after graduation, employability) | SIES / [mifuturo.cl](https://www.mifuturo.cl) | a web search tool. No bulk download confirmed; ask SIES or file a Ley 20.285 request for the table | Would allow a continuous "expected earnings of enrolled program" outcome |

Ley 20.285 requests are filed through the Portal de Transparencia. The agency must
answer within 20 business days, extendable once by 10.
