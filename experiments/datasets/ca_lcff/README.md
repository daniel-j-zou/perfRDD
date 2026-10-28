# California LCFF concentration grant: a continuous dose above a 55% kink

| Field | Value |
|---|---|
| **Q** | Unduplicated pupil percentage (UPP, 0-1) of the LEA in fiscal year t: low-income, English-learner and foster pupils counted once, summed over the current and two prior Census Days, over enrollment summed the same way (EC 42238.02(b)(5)). Taken from CDE's LCFF Summary Data, latest certification |
| **Threshold** | 0.55: the concentration grant is zero at UPP <= 0.55 and rises linearly above it |
| **Treatment** | Dose `extras["A"]` = c_t x r46_t x max(min(Q, cap) - 0.55, 0) dollars per funded ADA. c_t = 0.50 (2019-20, 2020-21) and 0.65 (from 2021-22, AB 130); r46_t = the grades 4-6 base grant per ADA of year t; cap = +inf for school districts. `load()`'s D = 1{Q > 0.55} |
| **Rule class** | A kink: a continuous dose that starts at a cutoff, with a slope that differs by fiscal year (rate regime x base grant). For charter schools the UPP in the formula is capped at a school district's UPP, so their dose can be flat in Q above the cap |
| **X** | Charter flag, log enrollment (CALPADS Census Day), Hispanic / Black / Asian / Filipino / American Indian / Pacific Islander / multiracial shares (White is the reference), disability share, TK-3 / 4-6 / 7-8 shares of funded ADA, elementary and high-school district flags, virtual charter, county-authorized charter, lagged outcome (+ missing flag, + "lag is 2019" flag), latitude / longitude (+ missing flag), fiscal-year and county indicators (84 columns, 15 continuous) |
| **Y** | `load_dose()`: CAASPP Smarter Balanced percent meeting or exceeding the standard, all tested grades, ELA, in spring of fiscal year t+1. Options: `subject="math"`, `measure="mss_gap"` (tested-weighted mean over grades of the LEA's mean scale score minus the state's), `outcome_lead=0` (spring of t) |
| **n** | 10,645 LEA-years (927 school districts, 1,277 charter schools), fiscal years 2020-21 to 2024-25, outcomes spring 2022-2026; districts only (`lea="district"`) 4,575 |
| **Source** | CDE Principal Apportionment "LCFF Summary Data" 2019-20 to 2025-26; CALPADS unduplicated pupil counts (UPC); CDE cumulative / Census Day enrollment; CAASPP research files (ETS for CDE); CDE public schools directory |

**Fit: good for the dose design, with caveats.**
- **The rule is reproduced exactly for school districts.** The published concentration
  grant equals the formula to the dollar for all 6,575 district-years 2019-20 to 2025-26
  (rule check below). Charters reproduce for 95% of the years where the authorizing
  district is known; the rest need the cap taken from the published grant.
- **The score is not sorted at the kink.** LEA-years just above 0.55 are not more
  frequent than those just below (ratio 0.83 with 1-point bins, 0.97 with 2-point bins;
  placebo points 0.82-1.21 and 0.87-1.17).
- **X predicts Q well, and units with the same dose still differ.** R^2 is 0.82 and
  sd(eta) is 0.105 (10.5 points of UPP). Within every dose decile the 10-90% range of eta
  is about 0.17-0.25.
- **Weak points.** Q is a three-year rolling share and barely moves within an LEA
  (within-LEA sd 0.037). There is only one outcome year under the 0.50 rate (2020-21). And
  41% of charter-years have a binding cap (dose set by another LEA's UPP).

## The rule

Per unit of funded ADA in grade span g, a school district or charter school receives

    base_g,t x [1 + 0.20 x UPP + c_t x max(UPP - 0.55, 0)]

The source is the "LCFF Summary Data Descriptions" tab of each workbook (CDE, June 2023
and later). It cites EC 42238.02(e) and (f).
- **The base grant.** The adjusted base grant includes the 10.4% K-3 and 2.6% 9-12
  add-ons. The necessary-small-school allowance is not in the supplemental or
  concentration base.
- **The charter cap.** "The charter school concentration grant calculation will be capped
  at the greater of the UPP of their authorizing school district or any school district in
  which the charter school reported having a physical location."
- **County offices of education** have a different formula (35% above 50%; juvenile
  court) and are excluded.

**Rates.** The adjusted base grant per ADA by grade span (TK/K-3, 4-6, 7-8, 9-12) is
`adapter.BASE_RATES`. The rates were recovered from the data.
- **Which LEAs.** The check uses the 12,553 LEA-years (2020-21 to 2025-26) whose
  supplemental grant equals 0.20 x base x UPP, that is, those with no small-school
  allowance.
- **Result.** For these, the published base grant equals the sum over grade spans of
  round(rate x ADA) within $2 for all of them, and within $1 for 99.96%.

| Fiscal year | c_t | TK/K-3 | 4-6 | 7-8 | 9-12 | Slope of A: c_t x r46_t ($ per ADA per unit of UPP) |
|---|---:|---:|---:|---:|---:|---:|
| 2019-20 | 0.50 | 8,503 | 7,818 | 8,050 | 9,572 | 3,909 |
| 2020-21 | 0.50 | 8,503 | 7,818 | 8,050 | 9,572 | 3,909 |
| 2021-22 | 0.65 | 8,935 | 8,215 | 8,458 | 10,057 | 5,340 |
| 2022-23 | 0.65 | 10,119 | 9,304 | 9,580 | 11,391 | 6,048 |
| 2023-24 | 0.65 | 10,951 | 10,069 | 10,367 | 12,327 | 6,545 |
| 2024-25 | 0.65 | 11,068 | 10,177 | 10,478 | 12,460 | 6,615 |
| 2025-26 | 0.65 | 11,323 | 10,411 | 10,719 | 12,746 | 6,767 |

- **COLAs.** The year-on-year changes match the statutory COLAs: 0% in 2020-21, 5.07% in
  2021-22, 6.56% plus the 6.28% base augmentation in 2022-23, then 8.22%, 1.07% and 2.30%.
- **The 2021-22 change.** The rate rose from 50% to 65% in 2021-22 (AB 130, Chapter 44,
  Statutes of 2021). This is CDE's wording as quoted by a web search; the CDE page itself
  is blocked (see "Blocked").
- **Phase-in.** Not relevant here. The 2018-19 Budget Act closed the LCFF funding gap, so
  2019-20, the first year in the data, is under full implementation. The 2013-14 to
  2017-18 gap-funding years are not covered (see "Not covered").

**How CDE rounds.** The grant is the sum over grade spans of
round(c_t x r_g,t x max(UPP - 0.55, 0) x ADA_g), with UPP to four decimals. Rounding the
total instead fails for 3% of districts by $1.

## Rule check (2026-10-28)

`adapter.rule_check(tol)` compares the published concentration grant ("Total LCFF
Concentration Grant", latest certification) with the formula. It covers every school
district and charter school with positive ADA and base grant. tol = $0.50 for districts;
the charter column uses $1.

| FY | Cert | Districts | Exact | Charters | Exact (authorizer cap, $1) | Cap: authorizer / unknown / from published |
|---|---|---:|---:|---:|---:|---|
| 2019-20 | AN R1 | 943 | 1.000* | 1,285 | 0.903* | 1,019 / 141 / 125 |
| 2020-21 | AN R1 | 942 | 1.000 | 1,276 | 0.901 | 1,003 / 146 / 127 |
| 2021-22 | AN R1 | 940 | 1.000 | 1,272 | 0.899 | 992 / 151 / 129 |
| 2022-23 | AN R1 | 939 | 1.000 | 1,267 | 0.893 | 982 / 150 / 135 |
| 2023-24 | AN R1 | 938 | 1.000 | 1,262 | 0.888 | 976 / 146 / 140 |
| 2024-25 | AN R1 | 937 | 1.000 | 1,257 | 0.886 | 964 / 150 / 143 |
| 2025-26 | P-2 | 936 | 1.000 | 1,244 | 0.887 | 949 / 154 / 141 |

\* 2019-20 has no grade-span ADA. For that year the adjusted base is the one implied by
the published supplemental grant, supp / (0.20 x UPP), and the tolerance is max($2, 0.01%).
The 2019-20 check is therefore "given the published supplemental grant".

**Charter cap.**
- **Which UPP caps a charter.** The cap is the authorizing district's **P-2** UPP, even in
  the annual certifications. Using the annual UPP instead fails often: Oakland charters in
  2020-21 are capped at 0.7597 (Oakland Unified at P-2), not 0.7592 (at AN R1).
- **When the authorizer is a school district.** This holds for 7,244 charter-years. The
  formula matches 95.0% of them, and 87.1% of the 2,797 where the cap binds. The misses
  have a site in another district with a higher UPP; CDE's site list is not public in these
  files.
- **County- or state-authorized charters.** For these 1,619 charter-years the cap is
  unknown, and the uncapped formula matches 64.1%.
- **Taking the cap from the published grant.** Where the formula misses, the adapter
  takes the cap implied by the published grant: 0.55 + grant / (c_t x adjusted base).
  `extras["cap_source"]` is "authorizer", "unknown" (no cap needed) or "published". The
  cap was taken from the published grant for 604 of the 6,070 charter-years in the sample.
  After this, `extras["A_lea"]` (formula per ADA at the LEA's own grade mix) is within
  $0.20 per ADA of the published grant per ADA for every LEA-year.

**UPP against CALPADS.** The UPP was recomputed from the three single-year CALPADS UPC
files, as sum(UPC) / sum(enrollment) (`upp_calpads`).
- It is within 1 point of the published UPP for 94-96% of districts and 82-96% of charters.
- It equals the published UPP to four decimals for only 16-29% of districts. The LCFF
  UPP adds adjustments, such as pupils in county programs and, from 2025-26, the TK
  English-learner proxy.
- Q is the published UPP throughout.

## Dose design

`load_dose(subject="ela", measure="pct_met", lea="all", years=(2021, ..., 2025), outcome_lead=1)`
returns an `RDDSample` with:

| Key | Content |
|---|---|
| `Q`, `threshold` | published UPP; 0.55 |
| `extras["A"]` | dose by the formula, $ per ADA at the year's grades 4-6 rate: c_t x r46_t x max(min(Q, cap) - 0.55, 0). 0 for 39.9% of the sample; 10/50/90% of positive doses: $253 / $1,222 / $2,087; max $2,977 |
| `extras["a_ref"]`, `["dose_levels"]` | 0.0; None (continuous) |
| `extras["schedule"]` | `adapter.schedule(q, group, cap=None, unit="dollars")`; reproduces `A` exactly. `unit="share"` gives c_t x max(min(q, cap) - 0.55, 0), the grant as a fraction of the base grant |
| `extras["group"]` | fiscal year (2021 = 2020-21); `extras["regime"]` = c_t (0.50 / 0.65) |
| `extras["cap"]`, `["cap_source"]`, `["cap_binds"]` | charter cap (inf for districts), where it comes from, whether min(Q, cap) < Q |
| `extras["Z_names"]` | `charter`, `log_enroll`, `share_hispanic`, `share_swd`, `y_lag`: suggested effect modifiers that vary among LEAs with the same UPP |
| `extras["lea_id"]` | 14-digit CDS code; cluster on it (LEAs repeat up to five times) |
| `extras["A_share"]`, `["A_lea"]`, `["published_dose"]` | dose as a share of the base grant; formula grant per ADA at the LEA's own grade mix; published grant / funded ADA |
| `extras["conc_published"]`, `["conc_formula"]`, `["supp_per_ada"]` | LEA totals; supplemental grant per ADA (linear in Q, part of the baseline) |
| `extras["upp_calpads"]`, `["upc_share_cur"]` | UPP recomputed from CALPADS; the current year's single-year unduplicated share |
| `extras["basic_aid"]` | district whose local revenue covers its LCFF entitlement (2020-21 on; 181 sample rows with A > 0) |
| `extras["Y_same"]`, `["tested"]`, `["df"]` | outcome in spring of t; students tested; the full panel rows |

- **Two schedules.** Under the 0.50 rate a point of UPP above 0.55 is worth $39 per ADA
  (2020-21). Under 0.65 it is worth $53-68, depending on the year's base. In base-grant
  units (`unit="share"`), the schedule is exactly two lines with slopes 0.50 and 0.65.
- **Dollars versus share of base.**
  - `A` uses one base rate per year (grades 4-6), so `schedule(q, fy)` is exact.
  - The LEA's own dollar dose (`A_lea`) also depends on its grade mix. K-3 and 9-12 ADA
    carry higher rates: at the 2024-25 rates, A_lea / A runs from 1.00 (all ADA in
    grades 4-6) to 1.22 (all ADA in grades 9-12).
  - X includes the grade-span ADA shares for this reason.
- **Charters.** `lea="district"` gives a pure s(Q, fiscal year) schedule. With charters
  (`lea="all"`, the default, or `lea="charter"`), 40.6% of charter rows have a binding
  cap, which is mostly LAUSD's UPP. For those rows the dose does not rise with the
  charter's own UPP.
- **Fiscal year 2019-20** (c = 0.50) is in the panel, but `load_dose` leaves it out by
  default. Its t+1 outcome is the spring 2021 test, which was optional: 52% of LEA-years
  have it, and those LEAs tested 22% of enrolled students statewide.
  - `years=(2020, ..., 2025)` adds 1,168 rows.
  - The same reasoning stops the lag from using 2021: the lag is the most recent CAASPP
    year among 2019 and 2022-2026 before t.

## Readiness (2026-10-28)

`adapter.readiness(load_dose())`. eta = Q - OLS fit of Q on (1, X); dose deciles are
over A > 0.

| Quantity | All (default) | Districts | Charters |
|---|---:|---:|---:|
| n, p | 10,645, 84 | 4,575, 80 | 6,070, 75 |
| R^2 of Q on X | 0.824 | 0.852 | 0.831 |
| sd(Q), sd(T), sd(eta) | 0.250, 0.227, 0.105 | 0.244, 0.225, 0.094 | 0.254, 0.231, 0.104 |
| Share at dose 0 | 0.399 | 0.401 | 0.398 |

- **R^2 without parts of X.** Without the lagged outcome it is 0.783; without the race and
  disability shares, 0.603.
- **By year.** R^2 is 0.82-0.84 and sd(eta) 0.099-0.106 in each fiscal year 2021-2025.
- **Math sample.** It has n = 10,640 and R^2 = 0.822.

eta quantiles by dose (all LEAs):

| Dose ($/ADA) | n | eta 10% | 50% | 90% |
|---|---:|---:|---:|---:|
| 0 | 4,251 | -0.168 | -0.045 | 0.066 |
| 1-253 | 641 | -0.123 | 0.002 | 0.126 |
| 253-527 | 641 | -0.125 | 0.008 | 0.123 |
| 527-799 | 639 | -0.101 | 0.021 | 0.151 |
| 799-1,048 | 640 | -0.097 | 0.025 | 0.151 |
| 1,048-1,222 | 639 | -0.079 | 0.039 | 0.145 |
| 1,222-1,489 | 639 | -0.071 | 0.036 | 0.171 |
| 1,489-1,692 | 642 | -0.061 | 0.038 | 0.151 |
| 1,692-1,892 | 642 | -0.050 | 0.029 | 0.139 |
| 1,892-2,087 | 639 | -0.038 | 0.041 | 0.132 |
| 2,087-2,977 | 640 | -0.041 | 0.034 | 0.143 |

- **Within dose deciles.** LEAs with the same dose differ in eta: the 10-90% range is
  0.17-0.25 in every decile, against sd(eta) = 0.105.
- **Across deciles.** The median of eta rises from -0.045 at dose 0 to about +0.03. The
  spread narrows at the top (fewer LEAs far above their fitted UPP).

**Bunching at 0.55.** These are counts of LEA-years in the full panel (15,438, fiscal
years 2019-20 to 2025-26) just above vs just below a point.

| Window | Kink 0.55: below / above / ratio | Placebo ratios at 0.30, 0.35, ..., 0.80 (min / median / max) |
|---|---|---|
| 1 point | 155 / 128 / 0.83 | 0.82 / 0.99 / 1.21 |
| 2 points | 316 / 306 / 0.97 | 0.87 / 1.00 / 1.17 |

Districts: 0.88 and 1.01; charters: 0.78 and 0.93. There is no excess mass above the
kink. The three-year rolling average makes precise sorting hard.

**Persistence.** 97-98% of LEAs are on the same side of 0.55 as in the previous year.
The sd of the year-on-year change in UPP is 0.027-0.031. The within-LEA sd of UPP is
0.037, against a between-LEA sd of 0.248.

## Sample and outcome

- **Unit.** An LEA x fiscal year: school districts (school code 0000000) and charter
  schools (charter number on the LCFF Summary). County offices of education are dropped.
- **Certification.** Each fiscal year uses its latest certification: Annual
  Recertification (AN R1) for 2019-20 to 2024-25, P-2 for 2025-26.
- **Sample filters.** LEA-years with zero ADA or base grant are dropped. The 2019-20 and
  2020-21 P-1 sheets give UPP in percent, and the adapter rescales them.
- **Outcome.**
  - The CAASPP research file "all students" row (student group 1, Smarter Balanced)
    supplies Y: for districts the district row, for charters the school row of the
    charter's CDS code.
  - The district row covers the district's own schools and its locally funded charters,
    not the direct-funded charters. LAUSD, ELA 2023: district row 213,677 tested against
    214,734 in its non-charter and locally funded charter school rows. This matches the
    LCFF split.
  - The 2026 file renames the levels ("Proficient and Above"). The statewide all-grade ELA
    share moves from 48.81 (2025) to 49.27 (2026), so the scale is treated as continuous.
  - Y: mean 46.0, sd 17.8, corr(Y, Q) = -0.66. 95-97% of LEA-years in 2020-21 to 2024-25
    have a t+1 outcome.
- **Covariates.**
  - **Enrollment and shares.** Log enrollment is the CALPADS Census Day K-12 count (TK-12
    from 2025-26) from the UPC file. Race and disability shares come from CDE enrollment
    files: district rows for non-charter schools, school rows for charters.
  - **Timing.** The enrollment files are cumulative enrollment for 2019-20 to 2022-23 and
    Census Day for 2023-24 on (`enroll_measure`). CDE changed the published series, and
    shares are comparable across the break.
  - **2019-20 grade-span shares** are carried from 2020-21: ADA was held at 2019-20 P-2
    levels in 2020-21.
- **Left out of X because they are the score.** FRPM, English-learner, foster and
  homeless shares, direct certification and CEP status. `upp_calpads` and
  `upc_share_cur` are in `extras` only.
- **Cache.** The adapter caches `data/processed/ca_lcff_panel.csv.gz`: 15,438 LEA-years,
  about 50 s to build. It is written atomically. Delete it, or call
  `panel(rebuild=True)`, after changing `adapter.py`.

## Getting the data

    # from the repository root; never run Python inside data/
    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_module('experiments.datasets.ca_lcff.download', run_name='__main__')"
    # add --check (via sys.argv) for sizes and hashes only

- **Access.** All files were downloaded on 2026-10-28 with Python's default `urllib`
  user agent: no login, form, cookie or challenge.
- **Size.** 31 files, 252.4 MB (395 MB on disk with the CAASPP zips extracted, into
  `caaspp/<year>/`, CSV/TXT members only).
- **Hashes.** `download.py` records each file's SHA-256 and reports a change.
- **Directory file.** The administrator name, e-mail and phone columns are dropped on
  download.
- **The `.xlsx` reader.** The workbooks are read by `xlsx.py`, a standard-library reader
  of cached cell values (this Python environment has no openpyxl/xlrd).

| Files (data/raw/) | URL | Bytes | Use |
|---|---|---:|---|
| `pa/lcffsummary{1920..2526}.xlsx` (7) | https://www.cde.ca.gov/fg/aa/pa/documents/lcffsummary1920.xlsx etc. | 12,373,269 | Q, ADA, base, supplemental, concentration grants, local revenue |
| `upc/cupc{1718..2526}-k12.xlsx` (9) | https://www.cde.ca.gov/ds/ad/documents/cupc1718-k12.xlsx etc. | 17,446,361 | enrollment, single-year UPC, UPP recomputation |
| `enroll/cenroll{1920,2021,2122,2223}.txt` | https://www3.cde.ca.gov/demo-downloads/ce/cenroll1920.txt etc. | 80,416,726 | race and disability shares (cumulative enrollment) |
| `enroll/cdenroll{2324-v2,2425,2526}.txt` | https://www3.cde.ca.gov/demo-downloads/census/cdenroll2425.txt etc. | 95,995,900 | same (Census Day) |
| `caaspp/sb_ca{2019,2021..2026}_1_csv_v*.zip` (7) | https://caaspp-elpac.ets.org/caaspp/researchfiles/sb_ca2019_1_csv_v4.zip etc. | 38,287,987 | outcomes and lags |
| `directory/pubschls.txt` | https://sd.cde.ca.gov/schooldirectory/report?rid=dl1&tp=txt | 7,902,844 | district type, charter funding type, virtual, location |

The CAASPP file versions are v4 (2019), v2 (2021) and v1 (2022-2026). The file names came
from the Internet Archive's public URL index and from direct requests for the version
numbers.

**Blocked.**
- Every `.asp` page on www.cde.ca.gov answers a script with `302` to a Radware
  bot-manager challenge, and nothing here works around it. This includes the LCFF
  Summary landing page https://www.cde.ca.gov/fg/aa/pa/lcffsumdata.asp, the rates page
  https://www.cde.ca.gov/fg/aa/pa/ratesandinfo.asp and the calculation pages such as
  https://www.cde.ca.gov/fg/aa/pa/calc21p1.asp.
- The static `/documents/` files and www3 / sd / ETS downloads are not behind the
  challenge.
- So the base rates come from the data, and the rule text from the workbooks' "Data
  Descriptions" tab.

**Not covered.**
- **Before 2019-20.** `lcffsummary1819.xlsx` and `lcffsummary1718.xlsx` return 404. Of the
  older "LCFF snapshot" workbooks only `lcffsnapshot13an.xls` (2012-13 annual, legacy
  `.xls`) still answers. Those years are the gap-funded phase-in anyway.
- **Urbanicity.** The NCES locale code is not in the CDE files. Latitude/longitude and
  county stand in for it.

## Caveats

- **The dose is revenue, not spending.** LEAs must spend supplemental and concentration
  funds on unduplicated pupils ("increased or improved services"), but the money is
  fungible.
- **Basic-aid districts** get no extra revenue from a higher entitlement
  (`extras["basic_aid"]`; not identifiable for 2019-20).
- **Timing.**
  - Y (spring t+1) also reflects the t+1 dose, which is almost the same: the score is a
    rolling average, and 97-98% of LEAs stay on the same side of the kink.
  - The single 0.50-rate outcome year (2020-21, Y in spring 2022) falls in the first
    0.65 year.
  - 2019-20 and 2020-21 are COVID years: ADA was held harmless and 2021 testing was
    optional.
- **Lagged outcomes.** For fiscal years 2020-21 and 2021-22 the lag is spring 2019
  (`lag_is_2019`).
- **Charters.** Charter UPP and outcomes come from small schools; 41% of charter-years
  have a binding cap; the cap of 604 sample rows comes from the published grant.
- **The UPP is self-reported.** It is built from LEA-collected meal applications and
  direct certification. There is no bunching at 0.55 (see "Readiness").
