# US bank capital categories (PCA): "well capitalized" is a both rule on three or four capital ratios

| Field | Value |
|---|---|
| **Q** | `load()`: the binding margin min_j (ratio_j - well-capitalized cutoff_j), in percentage points, over total risk-based, tier 1 risk-based, CET1 (2015 on) and tier 1 leverage ratios. `load_two_score()`: (total risk-based ratio %, leverage ratio %). `load_adequate()`: the binding margin to the adequately-capitalized cutoffs. Q is capped at 10 pp below and 30 pp above each cutoff |
| **Threshold** | well: total >= 10, tier 1 >= 6 (8 from 2015), CET1 >= 6.5 (from 2015), leverage >= 5 (12 CFR 325.103 to 2014, 12 CFR 324.403 from 2015). Adequately: 8 / 4 (6) / 4.5 / 4. On Q: 0 |
| **Treatment** | `load()`: `1{Q >= 0}`, well capitalized by ratios. `load_two_score()`: `1{Q1 >= 10 and Q2 >= 5}`, the same thing. `load_adequate()`: `1{Q >= 0}`, adequately capitalized or better. The treated side is always "at or above every cutoff" |
| **Rule class** | well capitalized is a both rule (a cutoff on each ratio, "and"); not well capitalized, and undercapitalized, are either rules ("or"). Cutoffs change in 2015 |
| **X** | log total assets, loans, deposits, real-estate loans, C&I loans, securities and brokered deposits / assets, noncurrent loans / loans (all at the previous quarter-end); log(1 + age); charter class, holding-company status, specialization group, state, mutual, subchapter S, metro, community-bank flag and year indicators (114 columns in `load()`, 8 continuous). No capital ratio |
| **Y** | `load()`, `load_two_score()`, `load_adequate()`, `load_quarterly()`: log(gross loans t+4 quarters) - log(gross loans t), clipped to [-1, 1]. `*_fail`: 1{the bank fails within the next 8 quarters} |
| **n** | `load()`: 248,741 bank-years (Q4 1992-2024, 15,626 banks), 243,985 well capitalized. `load_fail()`: 259,430 (Q4 1992-2023), 253,568 treated, 1,274 failures. `load_quarterly()`: 974,124 bank-quarters, 957,081 treated |
| **Source** | FDIC BankFind Suite API (api.fdic.gov/banks): financials (call reports), failures, history; 12 CFR 325.103, 324.401-324.403, 324.12, 324.303, 337.6 (govinfo.gov, ecfr.gov); FDIC Quarterly Banking Profile graph books 2006-2019 (fdic.gov) |

## The rule

Section 38 of the FDI Act (prompt corrective action, PCA) sorts every insured bank into
five capital categories. The two regulations in force over the sample (checked in
`data/raw/rule/`):

| Period | Well capitalized | Adequately capitalized | Source |
|---|---|---|---|
| 1992-12-19 to 2014-12-31 | total RBC >= 10 **and** tier 1 RBC >= 6 **and** leverage >= 5, and no written agreement, order or directive to keep a specific capital level | total >= 8, tier 1 >= 4, leverage >= 4 (>= 3 for CAMELS 1 banks not expecting significant growth) | 12 CFR 325.103(b) (FDIC, 2014 CFR edition); identical cutoffs in OCC 6.4(b) (2014), Fed 208.43(b) (2014), OTS 565.4(b) (2011) |
| from 2015-01-01 | total >= 10 **and** tier 1 >= 8 **and** CET1 >= 6.5 **and** leverage >= 5, and no written agreement, order or directive | total >= 8, tier 1 >= 6, CET1 >= 4.5, leverage >= 4 (advanced approaches and Category III banks: also supplementary leverage >= 3) | 12 CFR 324.403(b) (2015 CFR edition; eCFR 2026-06-30); 324.401(c) says subpart H applies "beginning on, and thereafter, January 1, 2015" and part 325 until then |

- **Below adequate.** Undercapitalized: any ratio below its adequate cutoff. Significantly
  undercapitalized: total < 6, tier 1 < 3 (4 from 2015), CET1 < 3, or leverage < 3.
  Critically undercapitalized: tangible equity / total assets <= 2.
- **CBLR (2020 on).** A qualifying community bank (assets under $10 billion, limited
  off-balance-sheet and trading exposure) may elect the community bank leverage ratio
  framework. It then reports no risk-based ratios and "shall be considered to have met"
  the well-capitalized ratio requirements if its leverage ratio exceeds 9%
  (324.12(a)(1), 324.403(b)(1)(ii)). The cutoff was >= 8% from 2020-04-23 to 2020-12-31
  and > 8.5% in 2021 (324.303). An elector that falls below has a two-quarter grace period
  if its leverage stays above 8% (7% in 2020, 7.5% in 2021). A 2026 rule (91 FR 22973,
  effective 2026-07-01) lowers the cutoff to 8%, after the last quarter in the data.
  **CBLR electors are dropped from every loader** (see below).
- **Timing.** A bank is deemed to know its category, and the category takes effect, as of
  the date its Call Report is due, or an exam report or FDIC notice if later (324.402(b)).
  Call Reports are due about 30 days after the quarter-end.
- **Consequences** (checked for brokered deposits only, 12 CFR 337.6 as of 2019-01-01):
  - A well-capitalized bank may accept brokered deposits without restriction.
  - An adequately capitalized bank needs an FDIC waiver, and then may not pay more than
    75 bp above market or national rates.
  - An undercapitalized bank may not accept brokered deposits.

  337.6(a)(3) uses the PCA categories of section 38. Below adequate, section 38 adds
  mandatory restrictions: capital restoration plans, asset-growth limits, and limits on
  dividends and management fees (section 38 of the FDI Act; not checked here). Deposit
  insurance assessments also depend on the capital group; this was not checked either.

## Ratios in the data

The FDIC computes the PCA ratios from call-report components. The definitions are in
`data/raw/docs/all_financial_reports.xlsx` ("Reference-Variables&Definitions") and
`risview_properties.yaml`.

| Ratio | Adapter | API field | Formula |
|---|---|---|---|
| total risk-based | `tot` | RBCRWAJ | 100 RBC / RWAJT |
| tier 1 risk-based | `t1` | IDT1RWAJR (RBC1RWAJ rounded to 0.01) | 100 RBCT1J / RWAJ |
| CET1 (2014 on) | `cet1` | IDT1CER (RBCT1CER rounded) | 100 RBCT1C / RWAJ |
| tier 1 leverage | `lev` | RBC1AAJ | 100 RBCT1 / AVASSETJ |

- The adapter recomputes each ratio from its components at full precision. The API sets
  the risk-based ratios to 0 for CBLR electors, noninsured banks and insured branches of
  foreign banks.
- Thrifts that filed the OTS Thrift Financial Report (before 2012) have the same fields.
  For them, leverage is core capital / adjusted tangible assets, under the OTS rule with
  the same cutoffs.
- RBCT1 and RBCT1J (tier 1 for leverage and for the risk-based ratio) differ in 0.2% of
  bank-quarters. RWAJ and RWAJT differ in 0.004%.

## Rule check (2026-10-28)

No bank-level PCA category is published: 324.401(e) even bars banks from advertising it. So
the rule on the ratios is the definition, and `adapter.rule_check()` makes two checks.

**1. The recomputed ratios equal the API's ratio fields.**

| Pair | Bank-quarters | Agreement |
|---|---:|---:|
| `tot` vs RBCRWAJ | 1,097,288 | 1.000000 |
| `t1` vs IDT1RWAJR | 1,097,288 | 1.000000 |
| `t1` (rounded) vs RBC1RWAJ | 1,097,286 | 0.999985 (16 rounding ties) |
| `lev` vs RBC1AAJ | 1,142,411 | 1.000000 |
| `cet1` vs IDT1CER | 192,157 | 1.000000 |
| `cet1` (rounded) vs RBCT1CER | 192,157 | 0.999990 |

**2. The rule's category counts against the only published PCA categories.** These are the
FDIC Quarterly Banking Profile's "Capital Category Distribution" tables, which are
aggregates. They cover DIF-member institutions excluding US branches of foreign banks, and
the categories come from the ratios only (the QBP prints the cutoff table). The rule uses
the same universe (BKCLASS N, NM, SM, SB, SI, SL). Critically undercapitalized uses
leverage <= 2 as a proxy for tangible equity <= 2. CBLR electors count as well
capitalized above their grace floor.

| Date | QBP: well / adequate / under / signif. / critical | Rule | Not well: QBP, rule |
|---|---|---|---:|
| 2006-12-31 | 8,621 / 57 / 3 / 0 / 0 | 8,616 / 57 / 6 / 0 / 1 | 60, 64 |
| 2007-12-31 | 8,444 / 79 / 6 / 2 / 2 | 8,421 / 98 / 8 / 5 / 2 | 89, 113 |
| 2008-12-31 | 8,098 / 147 / 31 / 13 / 16 | 8,009 / 191 / 66 / 23 / 16 | 207, 296 |
| 2009-12-31 | 7,652 / 172 / 91 / 50 / 47 | 7,582 / 193 / 111 / 66 / 60 | 360, 430 |
| 2010-12-31 | 7,338 / 140 / 80 / 66 / 33 | 7,292 / 157 / 83 / 78 / 48 | 319, 366 |
| 2011-12-31 | 7,082 / 101 / 78 / 72 / 24 | 7,056 / 116 / 79 / 74 / 32 | 275, 301 |
| 2012-12-31 | 6,882 / 91 / 55 / 41 / 14 | 6,870 / 94 / 61 / 40 / 18 | 201, 213 |
| 2013-12-31 | 6,667 / 72 / 39 / 26 / 8 | 6,663 / 72 / 36 / 29 / 12 | 145, 149 |
| 2014-12-31 | 6,415 / 52 / 24 / 14 / 4 | 6,415 / 49 / 26 / 15 / 4 | 94, 94 |
| 2015-12-31 | 6,112 / 35 / 16 / 17 / 2 | 6,110 / 36 / 16 / 18 / 2 | 70, 72 |
| 2016-12-31 | 5,865 / 22 / 9 / 12 / 5 | 5,863 / 24 / 9 / 11 / 6 | 48, 50 |
| 2017-12-31 | 5,637 / 22 / 5 / 6 / 0 | 5,636 / 23 / 4 / 7 / 0 | 33, 34 |
| 2018-12-31 | 5,384 / 11 / 5 / 5 / 1 | 5,384 / 10 / 6 / 5 / 1 | 22, 22 |
| 2019-12-31 | 5,163 / 6 / 3 / 3 / 2 | 5,160 / 9 / 3 / 3 / 2 | 14, 17 |
| 2020-06-30 | 5,057 / 2 / 3 / 2 / 2 | 5,056 / 3 / 3 / 2 / 2 | 9, 10 |

- **Coverage.** The total number of institutions matches the QBP exactly in 12 of the 15
  quarters and is off by one in 2006, 2007 and 2010. No institution in these quarters
  lacks a ratio.
- **Agreement.** In 2012-2020 the rule's well-capitalized count is within 0-12 of the
  QBP's, and the 2015 change of cutoffs and ratio definitions is reproduced.
- **Exception: 2007-2011.** The rule finds 24-89 more banks below well capitalized than the
  QBP did, all in the crisis years. The likely cause is data vintage, but this is not
  verified: the QBP was computed from the Call Reports as first filed, while the API
  holds the latest amended filings. Amendments in those years often added loan-loss
  provisions after exams, which cuts capital. Bank-level QBP categories do not exist, so
  the rows cannot be matched.
- **Not observable.** The fifth well-capitalized condition (no written agreement, order or
  directive requiring a specific capital level), and reclassification for unsafe or
  unsound condition (324.403(d)), are neither in the data nor in the QBP counts. So D is
  "well capitalized by ratios". Banks under capital orders are concentrated among
  distressed banks in 1992-1993 and 2008-2012. Some of them have D = 1 and were in fact
  adequately capitalized. The bank-level FDIC enforcement decisions are public, but they
  are not linked to CERT in any bulk file and were not used.

## Getting the data

    python -m experiments.datasets.fdic_capital.download          # idempotent, about 35 minutes
    python -m experiments.datasets.fdic_capital.download --check  # sizes and hashes only

Every source answered a plain request with an honest user agent on 2026-10-28; nothing
was blocked or rate-limited. The API needs no key. Files go to `data/raw/` (gitignored),
151 MB on disk.

| Source | Files | Bytes on disk | Use |
|---|---:|---:|---|
| `https://api.fdic.gov/banks/financials?filters=REPDTE:<q>&fields=...&sort_by=CERT&limit=2500&offset=<k>&format=csv`, 139 quarters 1991-12-31 to 2026-06-30 (index `risview_20260819185831`) | 527 pages + `counts.json` | 138,568,879 (361,917,409 uncompressed; 1,155,148 rows) | ratios, components, balance sheet, structure |
| `https://api.fdic.gov/banks/failures?filters=FAILYR:[1991 TO 2026]` (index `failures_1787667198788`) | 1 | 57,806 (201,058; 1,137 rows) | failures, Y |
| `https://api.fdic.gov/banks/history?filters=ORG_ROLE_CDE:FI AND CHANGECODE:[200 TO 299] AND EFFDATE:[<y>-01-01 TO <y>-12-31]`, 1991-2026 (index `history_20261023090007`) | 36 | 512,231 (2,438,491; 16,472 rows) | acquisitions (flag for Y) |
| `https://api.fdic.gov/banks/docs/` `risview_properties.yaml`, `institution_properties.yaml`, `failure_properties.yaml`, `history_properties.yaml`, `swagger.yaml`, `All%20Financial%20Reports.xlsx` | 6 | 1,576,319 | field definitions |
| `https://www.govinfo.gov/content/pkg/CFR-<year>-title12-vol<v>/xml/...`: 325.103, 324.401 (2014), 324.403 (2015), 6.4, 208.43 (2014), 565.4 (2011) | 6 | 54,226 | the rule to 2014; its 2015 version |
| `https://www.ecfr.gov/api/versioner/v1/full/<date>/title-12.xml?part=...&section=...`: 324.403, 324.12, 324.402 (2026-06-30), 324.303 (2021-06-30), 324.12 (2026-10-23), 337.6 (2019-01-01) | 6 | 46,656 | current rule, CBLR, timing, brokered deposits |
| `https://www.fdic.gov/analysis/quarterly-banking-profile/graph-book/<yyyy>dec/qbpgr.pdf`, 2006-2019 | 14 | 10,117,203 | published category counts (rule check) |
| `https://www.fdic.gov/quarterly-banking-profile/fdic-qbp-graph-book-490` (June 2020 table) | 1 | 65,848 | published category counts, CBLR era |

- **Paging.** The API is paged politely: 2,500 rows per request, sorted by CERT, with a
  1-second pause.
  - Each page is stored as the CSV the API returned, gzipped.
  - `api_manifest.csv` (tracked, next to `download.py`) lists every page: URL,
    uncompressed bytes, rows and SHA-256.
  - `download.py --check` re-hashes the pages against it.
  - Each quarter's row count is checked against one aggregation request
    (`financials/counts.json`).
- **Hashes.** The SHA-256 of every static file (docs, rule, QBP) is in `download.py`.
- **Not used.** The QBP graph books exist at the same URL for Q4 1997-2023. 1997-2005
  cover BIF members only, so they are not comparable. In 2020-2023 the table is an image.
  So only 2006-2019 are fetched. The FDIC's enforcement-decision search is not used.
- **Cache.** The adapter caches `data/processed/fdic_capital_panel.parquet` (250 MB,
  1,155,148 bank-quarters, 1-2 minutes to build). It writes a temporary file and
  renames it. Delete the cache after changing `adapter.py`. `load_quarterly()` needs about
  5 GB of memory.

## Sample

| Step | `load()` (Q4 1992-2024) | `load_fail()` (Q4 1992-2023) |
|---|---:|---:|
| Q4 bank reports from 1992-12-31 | 274,672 | 270,112 |
| insured banks and thrifts (drops noninsured NC and insured foreign branches OI) | 271,643 | 267,156 |
| not a CBLR elector | 262,990 | 260,190 |
| all rule ratios present | 262,984 | 260,184 |
| a call report at the previous quarter-end (for X) | 262,228 | 259,430 |
| Y observed: a report 4 quarters later with loans > 0 / the 8-quarter window inside the data | **248,741** | **259,430** |

- **One quarter per year.** The main loaders use Q4. Capital ratios are very persistent:
  adding the lagged score to X raises R^2 from 0.42 to 0.94. So the four quarters of a
  year add little independent variation, and the Y windows of consecutive quarters
  overlap. `load_quarterly()` keeps all quarters (974,124 bank-quarters, 1992Q4-2025Q2,
  with quarter indicators).
- **Pre-PCA data.** Data before 1992-12-31 serve only as lags.
- **CBLR electors** are 35-39% of banks at each year-end from 2020, 45,092 bank-quarters
  (2,584 banks) in all. They are dropped because they have no risk-based ratios. So from
  2020 the sample halves (about 2,900 bank-years a year) and holds the larger or more
  leveraged banks that did not elect.
  - In their own framework, the electors' leverage sits close to the CBLR cutoff: 3,576
    elector-quarters are within 0.5 pp above it, and 1,018 are at or below it (in the
    grace period).
  - That is a separate single-score rule with a time-varying cutoff. It is not offered as
    a loader, because "deemed well capitalized" also depends on the grace period.

## Which ratio binds

Shares of bank-years in `load()` whose binding margin (the minimum) is each ratio:

| | total RBC | tier 1 RBC | CET1 | leverage |
|---|---:|---:|---:|---:|
| all bank-years | 0.372 | 0.002 | 0.001 | 0.626 |
| near the cutoff, abs(Q) < 0.5 | 0.877 | 0.002 | 0.000 | 0.120 |
| not well capitalized (D = 0) | 0.910 | 0.001 | 0.002 | 0.088 |
| 2015 on, all | 0.423 | 0.005 | 0.003 | 0.569 |

- **Which ratios fail.** Of the 4,756 not-well bank-years:
  - total RBC alone fails in 2,924;
  - leverage alone in 302;
  - total and leverage in 679;
  - total, tier 1 and leverage in 707;
  - CET1 alone in 3, and tier 1 never alone.
- **So in practice it is a two-score rule.** Tier 1 and CET1 almost never decide the
  category. Total RBC >= 10 implies tier 1 >= 6 unless tier 2 capital is very large. From
  2015, tier 1 >= 8 and CET1 >= 6.5 are usually met when total >= 10, since tier 2 is
  mostly the allowance (about 1.25 pp of RWA).
- **`load_two_score()`** therefore uses (total RBC, leverage) in all years. It drops the 3
  bank-years that meet both cutoffs but fail on CET1, so that D equals well capitalized
  in every row. Leverage is the binding ratio in 62.9% of its rows; within 0.5 pp of the
  boundary, total RBC binds in 87.9%.

## Bunching (density check)

`adapter.density_check()` gives counts just below and above each cutoff in the `load()`
sample. For each ratio, it uses only the bank-years whose other ratios meet their cutoffs,
so that ratio alone decides D.

| Cutoff | [c-0.1, c) vs [c, c+0.1) | [c-0.25, c) vs [c, c+0.25) | [c-0.5, c) vs [c, c+0.5) |
|---|---:|---:|---:|
| total RBC = 10 | 327 vs 1,148 | 775 vs 3,069 | 1,380 vs 6,908 |
| leverage = 5 | 51 vs 155 | 118 vs 423 | 201 vs 1,041 |
| tier 1 RBC | 0 vs 2 | 0 vs 7 | 0 vs 37 |
| Q = 0 (binding margin) | 379 vs 1,297 | 905 vs 3,447 | 1,641 vs 7,735 |
| adequately: total RBC = 8 | 30 vs 88 | 78 vs 223 | 141 vs 444 |
| adequately: leverage = 4 | 16 vs 42 | 29 vs 127 | 54 vs 282 |

- **There is strong bunching just above every binding cutoff.** In 0.1 pp bins of total
  RBC, the count is about 300 per bin just below 10% and 1,148 in [10.0, 10.1), a jump of
  3.5 times; it keeps rising smoothly after that. Leverage jumps from 51 to 155 at 5%.
- **The quarterly panel looks the same**: 1,368 vs 4,321 at total RBC = 10 (±0.1 pp), and
  169 vs 441 at leverage = 5.
- **Banks manage capital to the cutoffs.** The density of Q is not continuous at 0, so
  local RD assumptions fail for the score itself. Treat D as a sorted assignment. For
  rules near the deployed one, the bunched banks would likely have moved with the cutoff.

## Readiness (2026-10-28, computed as in `multi_rule_readiness`, not written to its summary)

OLS of each score on [1, X]:

| Loader | n | Treated | R^2 | sd(T), sd(eta) | sd(eta) max/min over T deciles | Near cutoff (0.25 sd eta) | abs(Q) < 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `load()` | 248,741 | 243,985 (0.981) | 0.423 | 2.77, 3.23 | 3.27 | 0.065 | 0.038 |
| `load_fail()` | 259,430 | 253,568 (0.977) | 0.482 | 3.19, 3.30 | 3.19 | 0.068 | 0.039 |
| `load_quarterly()` | 974,124 | 957,081 (0.983) | 0.425 | -, 3.25 | - | 0.063 | 0.036 |
| `load_adequate()` | 248,741 | 247,675 (0.996) | 0.409 | 2.60, 3.13 | 3.52 | 0.006 | 0.004 |
| `load_adequate_fail()` | 259,430 | 257,630 (0.993) | 0.469 | 3.00, 3.19 | 3.42 | 0.007 | 0.004 |
| `load_two_score()`: total RBC | 248,738 | 243,985 (0.981) | 0.541 | 5.41, 4.99 | 3.19 | - | - |
| `load_two_score()`: leverage | | | 0.410 | 2.59, 3.11 | 3.76 | - | - |
| `load_two_score_fail()` | 259,427 | 253,568 | 0.547, 0.474 | | | | |

- **R^2 by regime.** For `load()`, R^2 is 0.429 to 2014 (207,462 bank-years) and 0.456
  from 2015 (41,279). Without the year indicators it is 0.388.
- **The two-score view.**
  - corr(T1, T2) = 0.752 and corr(eta1, eta2) = 0.808.
  - With each score scaled by its sd(eta), the weakest direction of T has sd 0.46.
  - The two ratios move together, but X still separates them.
- **The near-boundary share is small.** 3.8% of bank-years are within 0.5 pp of
  well-capitalized and 8.3% within 1 pp. For the adequately-capitalized boundary it is
  0.4% and 0.8%. The not-well side is small too: 4,756 bank-years. 783 of them are in
  1992 and 945 in 2008-2011; only 5-16 a year from 2018.

## Outcomes

- **Loan growth** (`y_dloan`):
  - log(LNLSGR at t+4) - log(LNLSGR at t), gross loans and leases, clipped to [-1, 1].
    This clips 0.9% of rows; the raw sd is 0.244 and the clipped sd 0.173.
  - `extras["y_dloan_raw"]` keeps the unclipped value.
  - It needs a report at t+4, so it conditions on survival: 19.4% of not-well and 4.8% of
    well-capitalized Q4 bank-years have no report four quarters later (mergers, failures).
  - `extras["acq_next4"]` flags banks that acquire another institution in the next four
    quarters (FDIC history change codes 211-225), which inflates loan growth: 3.2% of
    rows.
- **Failure** (`y_fail8`):
  - 1 if the FDIC failures list has a FAILURE for the bank's CERT dated in the 8 quarters
    after t. It is defined only when that window ends by 2026-06-30.
  - It does not condition on survival.
  - In `load_fail()` it is 1 for 1,274 bank-years, from 664 of the 668 failures of
    1993-2025.
  - `extras["y_assist8"]` holds open-bank assistance (26 bank-years).

| Bank-years | D = 0 | D = 1 |
|---|---:|---:|
| loan growth, all | -0.007 | 0.085 |
| loan growth, abs(Q) < 0.5 (n = 1,641; 7,735) | 0.061 | 0.099 |
| loan growth, abs(Q) < 0.25 (905; 3,447) | 0.063 | 0.097 |
| placebo: past loan growth t-4 to t, abs(Q) < 0.5 | 0.136 | 0.169 |
| failure in 8 quarters, all | 0.135 | 0.002 |
| failure, abs(Q) < 0.5 (1,783; 8,261) | 0.022 | 0.014 |
| failure, abs(Q) < 1 (2,883; 19,293) | 0.030 | 0.011 |
| median Q four quarters later, abs(Q) < 0.5 | 0.46 | 0.64 |
| change in brokered deposits / assets, abs(Q) < 0.5 | -0.0015 | 0.0031 |

- **Loan growth near the cutoff.** Banks just below well capitalized grow loans 3.8 pp
  less over the next year. But they already grew 3.3 pp less over the previous year
  (placebo), so the raw gap is mostly a pre-existing difference.
- **Failure near the cutoff.** The failure rate just below is 1.5-2.8 times the rate just
  above.
- **Capital is rebuilt.** Banks just below the cutoff are back above it a year later
  (median Q = +0.46). Brokered deposits, the direct consequence, fall slightly just below
  and rise just above.
- **Other variables in `extras`.** `y_dbrokered`, `q_well_lead4`, `q_well_lag1`,
  `placebo_dloan`, `exit_4q`, `fail_date`, the four ratios uncapped, `q_well` and
  `q_adequate` uncapped, `binding_ratio`, `category` (five PCA categories), `regulator`,
  `cert`, `date`, `group` (regime, pre2015 / basel3) and `group_cutoffs`.

## Caveats

- **Sorting at the cutoffs.** The bunching above is large: the density jumps 3-3.5 times at
  total RBC = 10% and leverage = 5%. The deployed rule shapes the score, so for any other
  rule the scores would differ (banks target the cutoff that applies).
- **Unobserved part of the rule.** Written agreements, capital orders and reclassification
  (see the rule check). D is well capitalized by ratios. Before 2015, the 3% leverage
  floor for CAMELS-1 banks in "adequately capitalized" is also unobserved; it affects only
  the `adequate` loaders.
- **Supplementary leverage ratio.** From 2018 (advanced approaches, and later Category III),
  324.403 also uses the supplementary leverage ratio for the adequately-capitalized
  category. It is not in the API and is ignored. This concerns a few dozen of the
  largest banks. The well-capitalized supplementary leverage standards for subsidiaries of
  the largest holding companies under OCC and Fed rules were not checked.
- **Rule change in 2015.** The tier 1 cutoff (6 to 8) changed, CET1 was added, and the
  definitions of capital and risk-weighted assets changed. `extras["group"]` marks the
  regime. The binding margin Q is comparable across regimes only as a distance to the
  regime's own cutoffs.
- **Data vintage.** The API holds amended Call Reports. The categories that banks and
  supervisors acted on came from the original filings. They differ in 2007-2011 (see the
  rule check).
- **Timing.** The category takes effect when the Call Report is due, about a month after
  the quarter-end. Y windows start at the quarter-end.
- **Selection.** The loan-growth loaders condition on survival, as above. CBLR electors
  (2020 on) are dropped. Mergers inflate loan growth (`acq_next4`).
- **Treated share.** 98% of bank-years are well capitalized. Most of the near-boundary and
  below-boundary mass is in 1992-1993 and 2008-2012.
- **Covariates.** All balance-sheet X are at the previous quarter-end, and none is a
  capital ratio. The lagged score `q_well_lag1` is in `extras`; it is not in X, because
  with it R^2 = 0.94. The specialization group and the community-bank flag are FDIC
  classifications of the current quarter's balance sheet (asset concentrations, size).
  They are not functions of capital.
- **Clustering.** 248,741 bank-years come from 15,626 banks, so cluster by
  `extras["cert"]`.

## Suggested policy boxes

- **`load()` (binding margin).** Shift the cutoff in **[-1, +1] pp**, which moves every
  ratio's cutoff together.
  - D changes for 20,728 bank-years (8.3%).
  - Narrowed to [-0.5, 0.5]: 9,376 (3.8%).
- **`load_two_score()`.** Total RBC cutoff in **[9, 11]** and leverage cutoff in
  **[4, 6]**, around (10, 5).
  - D changes somewhere in the box for 20,686 bank-years (8.3%): 18,441 on the total-RBC
    strip and 4,296 on the leverage strip.
  - [9.5, 10.5] x [4.5, 5.5] leaves 9,362.
  - Both cutoffs at their adequately-capitalized values (8, 4) lie outside the box. A wider
    box toward them, [8, 12] x [4, 6], gives 42,659.
- **`load_adequate()`.** Shift in [-1, +1] pp. Only 0.8% of rows (about 1,900) are in that
  band, so this boundary is thin.
- **For all boxes.** The density above each deployed cutoff is inflated by sorting, so
  counts on the strict side of a box overstate the banks that would actually be near a
  different cutoff.
