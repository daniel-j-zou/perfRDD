# Chile SEP concentration subsidy: a step dose on the school's priority share

**Status (Claude, 2026-10-28): built but not yet loadable.** The law, the SIMCE school
results and the school-level performance aggregates are in hand and checked. The
MINEDUC files that give the score (priority students by school) and the SEP agreement
could not be downloaded: `datosabiertos.mineduc.cl` did not answer all afternoon
(13:10-15:20 EDT; see "Access"). `download.py` has every URL. Once it runs, `adapter.py`
builds the panel; then fill in the TODO sections with `readiness.py` and `rule_check()`.

| Field | Value |
|---|---|
| **Q** | the school's share of priority students in score year s, in percent: priority students in regular basic and youth secondary grades (MINEDUC SEP file, year s) over student records in those grades (performance file, year s). Variants `Q_b14` (grades 1-4) and `Q_b18` (grades 1-8) in `extras` |
| **Threshold** | 15 (the dose starts). Tier cutoffs 15, 30, 45, 60 (`CUTS`), `>=` at each |
| **Treatment** | a dose: the Art. 16 concentration rate for pre-K to 4th grade (band A), in USE per enrolled student per month, set by Q and paid in t = s + 1. 2008-2010: 0 / 0.098 / 0.168 / 0.224 / 0.252; 2011 on: 0 / 0.118 / 0.202 / 0.269 / 0.302. `D = 1{Q >= 15}` for the registry |
| **Rule class** | several cutoffs on one score with a step schedule; the schedule differs by period (and by grade band) |
| **X** | year s: dependency (municipal / private voucher / SLEP), rural, log enrolment, share of enrolment in grades 1-4, has secondary, share of girls, lagged SIMCE 4 mean and SIMCE socioeconomic group (with missing flags), promotion rate, attendance and grade average of grades 1-8, region and year dummies |
| **Y** | `load_dose()`: year-t school mean SIMCE 4th grade, (reading + math) / 2. Also `load_dose_math`, `load_dose_reading`, `load_dose_promotion` (grades 1-8 promotion rate in t), `load_dose_attendance` (mean attendance in t) |
| **n** | TODO. SEP schools x payment years 2009-2025 with the outcome; SIMCE 4 exists for t = 2009-2018 and 2022-2025, about 6,500-7,800 schools a year with a reported mean |
| **Source** | Ley 20.248 Art. 16 (LeyChile XML, three versions); MINEDUC Centro de Estudios open data (SEP students, SEP summaries by school, subsidies paid, school directory; performance by student, read from `chile_retention`); Agencia de Calidad de la Educacion public SIMCE bases |

## The law (Ley 20.248, Art. 16)

Ley 20.248 (Subvencion Escolar Preferencial, SEP, 2008) pays schools that sign a SEP
agreement two things on top of the regular voucher:
- a voucher per priority student (Art. 14), and
- a **concentration subsidy** (Art. 16): a monthly amount per student, in USE (unidad de
  subvencion educacional), that rises in steps with the school's share of priority
  students.

Two details differ from the usual summary. The concentration subsidy is paid for
**every enrolled student** of the levels covered ("por todos los alumnos", Art. 16,
third paragraph), not only for priority students. It is multiplied by the students'
average attendance over the three previous months.

**The share** (Art. 16, last paragraph; the same text since 2008): the average
enrolment of priority students in the levels covered by SEP, over the average enrolment
in those levels, both as registered in the school from March to December of the
previous year.

**The table** (USE per enrolled student per month). It comes from the LeyChile XML of
each version (`data/raw/law/`). From 2011 the table is an image inside the XML;
`download.py` extracts it next to the XML.

| Share of priority students | Pre-K (1st transition level) to 4th basic | 5th-6th basic | 7th-8th basic | 1st-4th secondary |
|---|---|---|---|---|
| **Original text, in force 2008-02-01** | | | | |
| < 15% | 0 | 0 | 0 | not covered |
| 15% to < 30% | 0.098 | 0.065 | 0.033 | |
| 30% to < 45% | 0.168 | 0.112 | 0.056 | |
| 45% to < 60% | 0.224 | 0.149 | 0.075 | |
| 60% or more | 0.252 | 0.168 | 0.084 | |
| **Ley 20.501 table, in force from 2011-03-01** | | | | |
| 15% to < 30% | 0.118 | 0.078 | 0.040 | 0.040 |
| 30% to < 45% | 0.202 | 0.134 | 0.067 | 0.067 |
| 45% to < 60% | 0.269 | 0.179 | 0.090 | 0.090 |
| 60% or more | 0.302 | 0.202 | 0.101 | 0.101 |
| **Current table (image "LEY20637"), in force from 2012-10-26** | | | | |
| 15% to < 30% | 0.118 | 0.118 | 0.078 | 0.078 |
| 30% to < 45% | 0.202 | 0.202 | 0.134 | 0.134 |
| 45% to < 60% | 0.269 | 0.269 | 0.179 | 0.179 |
| 60% or more | 0.302 | 0.302 | 0.202 | 0.202 |

- The 2008 band-A values match Feigenberg, Yan and Rivkin (NBER w23178, pp. 8-9):
  9.8, 16.8, 22.4 and 25.2% of a USE.
- **When the 2011 table starts.** LeyChile versions dated 2011-02-27 and 2011-02-28 still
  have the original Art. 16; the 2011-03-01 version has the new table. So it applies
  from the 2011 school year, which starts in March.
- The 2012-10-26 change raises only grades 5-8 and secondary; band A is unchanged.
  `adapter.period()` applies it from the 2013 school year.
- No later version changes Art. 16. The other dates are 2011-08-27, 2011-10-26,
  2012-02-02, 2015-06-08, 2018-08-25 and 2021-01-05; the latest LeyChile text is the
  2021-01-05 one.

## Dose design

`adapter.load_dose() -> RDDSample`:

| Item | Value |
|---|---|
| `Q` | the priority share in year s (percent, capped at 100) |
| `threshold` | 15.0, where the dose starts |
| `extras["A"]` | `schedule(Q, year)`: the band-A rate in USE per student per month |
| `extras["a_ref"]` | 0.0 |
| `extras["dose_levels"]` | [0, 0.098, 0.118, 0.168, 0.202, 0.224, 0.252, 0.269, 0.302] |
| `extras["schedule"]` | `schedule(q, group, band="A")`; `group` is a payment year, a table key ("2008", "2011", "2013") or a sample label |
| `extras["group"]` | "2008-2010" or "2011+" (the band-A table in force in t) |
| `extras["Z_names"]` | rural, private_voucher, log_enrol, lag_simce4, female_reg, lag_gse4 |
| other `extras` | `tier` (0-4), `rbd`, `year` (t), `score_year` (s), `Q_b14`, `Q_b18`, `n_reg`, `cod_depe2`, `cuts`, `tables` |

**Why band A.** SIMCE 4th grade is the main outcome, and its cohort is paid the band-A
rate. A school's total concentration payment also depends on its grade mix (bands B-D
pay less per student). `schedule(..., band="B"/"C"/"D")` gives the other columns.

**Timing.** The share of year s sets the rate paid in t = s + 1 (Art. 16). SIMCE 4 is
given in the second half of year t, so Y follows a full year at the dose. X is measured
in year s. The first score year is 2008, because the first public priority list is
2008's. The rule for the 2008 payments (share of 2007) cannot be built from public
files.

**Sample** (`sample_frame`). Schools with a SEP agreement in t; municipal, private
voucher or SLEP in s; at least one regular student in s; the outcome observed in t.

## How Q approximates the law (to verify with `rule_check`)

- **Numerator.** Students flagged priority in the year-s SEP file, in regular basic
  (`COD_ENSE` 110) and youth secondary grades.
- **Denominator.** Student records in the same grades in the year-s performance file
  (`chile_retention`). That file has every student who was enrolled during the year,
  including those who left; a transferred student counts at both schools. The law
  uses the March-December average, which no public file gives.
- **Pre-K is missing.** Pre-K (both transition levels) is in the law's share but not
  in the performance files. `sep_school()` counts pre-K priority students separately
  (`n_prio_parv`) if the SEP file has them.
- **Levels covered.** SEP was phased in by grade (original text, fifth transitory
  article). The concentration subsidy ran from 2008-01-01 for pre-K to 4th grade, and
  grades 5-8 joined one level a year from the law's second year: 5th grade in 2009
  through 8th grade in 2012. The 2011 text adds secondary; its phase-in was not checked.
  The adapter's default Q uses all regular grades; `Q_b14` and `Q_b18` are the
  alternatives. Pick the one that best reproduces the official payments.

## Rule check

TODO once the files are in. `rule_check()` is to compare `tier(Q)` with the tier implied
by the official record: the concentration line in "Subvenciones a establecimientos"
(MINEDUC, payments by school and month, 2005-2025). Report agreement by year and Q
variant.

## Readiness

TODO: `python -m experiments.datasets.chile_sep.readiness` reports n, the R^2 of Q on X,
sd(eta), the share at dose 0, the 10/50/90% quantiles of eta at each dose level, and
bunching at 15/30/45/60 against placebo points. The code was run on placeholder counts
(random, not saved) to check the mechanics only.

## What is in hand (checked 2026-10-28)

**SIMCE school results** (Agencia de Calidad, public bases, `read_simce`): schools with
a 4th-grade mean (both subjects reported; suppressed results are blank or 0 in the
files and are set to missing):

| Year | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Schools in file | 7,540 | 7,611 | 8,000 | 7,861 | 7,924 | 7,928 | 7,943 | 7,753 | 7,813 | 7,667 | 7,558 | 7,507 | 7,444 | 7,414 | 7,210 | 7,204 | 7,185 | 7,143 |
| With a mean | 7,505 | 7,599 | 5,735 | 7,810 | 5,521 | 7,296 | 7,082 | 7,095 | 6,997 | 6,947 | 6,779 | 6,779 | 6,769 | 6,743 | 6,531 | 6,520 | 6,578 | 6,568 |
| Mean score | 242.3 | 242.7 | 245.4 | 245.4 | 252.1 | 252.8 | 254.9 | 254.9 | 251.9 | 251.6 | 254.6 | 256.2 | 256.9 | 256.9 | 251.9 | 258.8 | 265.0 | 263.2 |

- There is no 4th-grade base for 2019 in the catalog, and SIMCE was not given in
  2020-2021. So Y is missing for t = 2019-2021.
- 8th grade: 2007, 2009, 2011, 2013, 2014, 2015, 2017, 2019, 2025.
- Separators (`|`, tab, `;`) and encodings vary by year; the reader detects them.
- The socioeconomic group (`cod_grupo`, 1-5) is text before 2017 and a code after.

**Performance aggregates** (`rend_school`, from the 2007-2025 student files in
`chile_retention`, read in place). About 9,000-10,000 schools a year. The count rises to
about 10,700 from 2023 because more special-education schools appear; their students
are outside the regular grades. They give enrolment by band, the share of girls,
promotion rate, attendance, grade average, dependency, rural, region and commune.

## Getting the data

    python -m experiments.datasets.chile_sep.download              # everything (idempotent)
    python -m experiments.datasets.chile_sep.download --only simce4 law
    python -m experiments.datasets.chile_sep.download --check      # sizes on disk
    python -m experiments.datasets.chile_sep.download --drop-extracted sep_students

Run from the repo root with `python -I` (as in the task instructions). If a host does
not answer, the script reports it and skips that host's files in seconds; rerun later.

| Group | Source | Years | Status 2026-10-28 |
|---|---|---|---|
| `sep_students` | MINEDUC "Alumnos preferentes, prioritarios y beneficiarios SEP" (`Alumnos-SEP-<year>.rar`) | 2008-2025 | host down; archive sizes 10-39 MB a year (from the Internet Archive index) |
| `sep_summary` | MINEDUC "Resumen de alumnos preferentes, prioritarios y beneficiarios SEP por establecimiento" | 2008-2023 | host down |
| `subsidies` | MINEDUC "Subvenciones a establecimientos" | 2005-2025 | host down |
| `directory` | MINEDUC "Directorio de establecimientos educacionales" | 2004-2025 | host down (not strictly needed: the performance files carry dependency, rural, region and commune) |
| `simce4` | Agencia de Calidad, `rest/archivo/obtener?uuid=...` | 2005-2018, 2022-2025 | 99.4 MB downloaded, 235 MB extracted |
| `simce8` | same | 9 years | 48.0 MB downloaded, 93 MB extracted |
| `catalog` | Agencia's public file catalog, `rest/archivo/getAllByTipo/0` | | 0.1 MB |
| `law` | LeyChile `Consulta/obtxml?opt=7&idNorma=269001&idVersion=<date>` | 2008-02-01, 2011-03-01, 2012-10-26, latest | 1.1 MB |

The MINEDUC URLs are the links on the dataset pages (read from the pages' archived
copies, because the live pages were down). No data were taken from the archive.

**Access.**
- **Agencia and LeyChile.** No login, form, click-through or CAPTCHA. The Agencia
  portal's "Descargar" button calls the same public endpoint as `download.py`. Each
  SIMCE archive includes a "Leer antes - Condiciones de uso bbdd.doc" with the
  Agencia's conditions of use; the author should read it before publishing results.
- **MINEDUC.** On 2026-10-28, `datosabiertos.mineduc.cl` (163.247.44.239) timed out
  from this machine and refused connections from Anthropic's fetcher. The same held for
  `www.mineduc.cl`, `centroestudios.mineduc.cl` and `sige.mineduc.cl`, and for
  `www.supereduc.cl`, all in 163.247.0.0/16. A traceroute reached 163.247.20.162, so
  the servers drop traffic rather than the route failing. `chile_retention`
  downloaded from the same host on 2026-09-28. The cause is an outage or a block on
  foreign traffic.

## Problems to expect

- **Selection into SEP.** The dose exists only with a SEP agreement. Municipal schools
  joined almost universally; private-voucher schools chose, and those with few priority
  students had the least to gain. Conditioning on the agreement in t selects on Q.
  Check the agreement rate by Q among private-voucher schools.
- **Bunching.** Feigenberg, Yan and Rivkin report bunching at the 60% cutoff. The share
  is computed by MINEDUC from its own priority list, but schools shape it through whom
  they enrol. `readiness.py` compares counts just above and below each cutoff with
  placebo points.
- **The score is approximated.** Q is not the law's March-December average, and pre-K
  is missing. Measurement error blurs the tiers, so the dose is fuzzy in Q until
  `rule_check` shows otherwise.
- **The dose per school is not one number.** A school's total concentration payment
  mixes band rates by its grade mix and scales with attendance and the USE value of the
  year. The sample's A is the band-A rate.
- **Persistence.** Shares change slowly, so the year-t tier mostly equals the year-(t-1)
  tier. Y measures the effect of the current tier plus its history; cluster by `rbd`.
- **Strategic test-taking.** Feigenberg et al. find more missing 4th-grade scores after
  SEP. The SIMCE count variables (`simce4_nlect`) allow a check against enrolment in
  grade 4.
- **2016 on.** Preferential students (Ley 20.845) receive half the priority voucher but
  do not count in the Art. 16 share. `n_pref` is kept separately and is not in X.
