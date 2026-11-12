# MIMIC-IV ICU titration protocols as dose schedules (screen on the open demo)

Screened 2026-11-12 (Claude) on the open MIMIC-IV Clinical Database Demo v2.2 (100 patients).
Not related to the older severity-score stub in `../mimic/`.

| Field | Potassium replacement | Heparin infusion nomogram | Insulin sliding scale |
|---|---|---|---|
| **Q** (the score) | serum potassium, mEq/L (labevents 50971) | aPTT, s (labevents 51275), drawn while a heparin infusion runs | finger-stick glucose, mg/dL (chartevents 225664, 226537), ICU only, while a sliding-scale order is active and no insulin infusion runs |
| **Threshold** | no fixed cut in the data; dosing becomes common below 4.0 (lab reference range 3.3-5.1) | no change at 55-90 s; increase below, decrease 90-120, hold above 120 | no insulin at <= 150; dose rises in steps of 40-50 mg/dL above |
| **Treatment** (dose A) | KCl given (mEq) from the result until the next draw or 6 h; reference 0 | change in the infusion rate (units/kg/h) from the result to the next draw or 4 h; reference 0 (no change) | SC regular or lispro insulin (units) from 15 min before to 2 h after the stick; reference 0 |
| **Rule class** | step schedule (a protocol order, "Potassium Chloride Replacement (Critical Care and Oncology)", exists in the eMAR; its text is not stored); in the data a tendency, not a rule | step nomogram (4 categories); possibly two target ranges | step schedule, linear in bands (k units per 40-50 mg/dL above c), scale written per patient (not stored) |
| **X** | age, sex, weight, creatinine (72 h), previous potassium and log hours since it, KCl in the previous 24 h, maintenance KCl in the previous 24 h, hour of day, unit type | age, sex, weight, creatinine, previous aPTT and hours since it, current rate (units/kg/h), previous rate change, hour, unit type | age, sex, weight, creatinine, previous glucose and hours since it, SC insulin in the previous 24 h, long-acting insulin in the previous 24 h, hour, unit type |
| **Y** | next serum potassium (within 24 h) | next aPTT (within 24 h) | next finger-stick glucose (within 12 h) |
| **n** (demo) | 981 events, 54 patients (2,418 draws before restrictions) | 209 events, 21 patients | 808 events, 47 patients; eMAR version 328 events, 20 patients |
| **Source** | MIMIC-IV Clinical Database Demo v2.2, https://physionet.org/content/mimic-iv-demo/2.2/ (ODbL 1.0, no login); full data MIMIC-IV v3.1, https://physionet.org/content/mimiciv/3.1/ (credentialed) | same | same |

## Fit

| Protocol | Best step rule match | Share at reference | R^2 (Q on X) | Outcome gap, median (IQR) | Projected n, full MIMIC-IV | Verdict |
|---|---|---|---:|---|---|---|
| Potassium | 86% (the best monotone rule is "no KCl above 2.9", i.e. the share at 0); among draws below 3.5 only 50% (44 of 88) get any KCl within 6 h | 86% | 0.48 | 12.1 h (7.5-22.3) | ~640,000 | **does not fit**: dosing is discretionary |
| Heparin | 81% for the 4-category rule (increase / no change / decrease / hold); 72% for the rate change rounded to 1 unit/kg/h | 61% | 0.23 | 7.4 h (6.3-9.5) | ~137,000 | **promising** |
| Insulin, ICU (inputevents) | 66% pooled monotone; 75% with a linear scale fitted per patient | 65% | 0.58 | 6.0 h (4.0-6.0) | ~528,000 | **promising, with caveats** (scale not stored; zero doses inferred) |
| Insulin, eMAR records | 66% pooled monotone; 76% per-patient scale; 84% when the stick is a finger stick within 15 min of the dose (126 events) | 50% | 0.70 | 5.0 h (3.6-6.0) | ~214,000 | **promising** (zeros recorded; 96-100% dosed at >= 200 mg/dL) |

"Best step rule": score cut into fine bins (0.1 mEq/L, 5 s, 10 mg/dL), dose rounded (10 mEq, 1 unit/kg/h,
1 unit); the rule is the dose per bin that most events received, constrained to fall (potassium, heparin)
or rise (insulin) with the score (`fit_step_rule`). The unconstrained modal rule fits 86%, 74%, 68%, 68%. For
comparison, the UI benefit formula matched 96-98% of claims.

**Estimator.** Only potassium reached about 1,000 events (981). `dose_config.py` runs it (dose levels 0, 20,
40, 60 mEq; patient clusters); log in `experiments/runs/dose_schedule/mimic_titration.log`. It runs end to end
but the output is not interpretable: effects from -62 to +35 mEq/L on the next potassium, rotation sds as large,
coverage 0% for 60 mEq in every bin; modifier test -0.0029 (se 0.0081), r2_absorbed 0.83. With 54 patients
this is a pipeline check only. Heparin (209) and insulin (808, 328) were not run.

## Rules found

**Potassium** (981 events; share given any KCl within 6 h, by band):

| band | 0 | 10 | 20 | 30 | 40 | 60 | 80+ | n | dosed |
|---|---|---|---|---|---|---|---|---|---|
| < 3.0 | 2 | 0 | 1 | 0 | 0 | 1 | 3 | 7 | 0.71 |
| 3.0-3.3 | 14 | 0 | 3 | 1 | 4 | 7 | 3 | 32 | 0.56 |
| 3.3-3.5 | 28 | 1 | 3 | 0 | 10 | 3 | 4 | 49 | 0.43 |
| 3.5-3.7 | 48 | 1 | 4 | 0 | 14 | 8 | 0 | 75 | 0.36 |
| 3.7-3.8 | 38 | 0 | 7 | 0 | 9 | 0 | 0 | 54 | 0.30 |
| 3.8-3.9 | 41 | 1 | 12 | 0 | 11 | 0 | 0 | 65 | 0.37 |
| 3.9-4.0 | 51 | 0 | 4 | 0 | 1 | 0 | 0 | 56 | 0.09 |
| 4.0-4.2 | 150 | 0 | 3 | 0 | 0 | 0 | 0 | 153 | 0.02 |
| 4.2-4.5 | 182 | 4 | 4 | 0 | 1 | 0 | 0 | 191 | 0.05 |
| >= 4.5 | 289 | 4 | 6 | 0 | 0 | 0 | 0 | 299 | 0.03 |

When KCl is given, the dose rises as potassium falls (20-40 mEq at 3.7-3.9, 40-60 at 3.0-3.5, 80+ below 3.0),
and almost none is given at >= 4.0. But below 3.5 half the draws (44 of 88) get nothing within 6 h, and 59% of
those get nothing within 24 h of the result either. The same holds in the ICU stays whose oral doses are recorded
(50% dosed below 3.5) and in the cardiac ICU (8 of 21 at 3.0-3.5). The lab's reference range is 3.3-5.1 mEq/L
(3.5-5.4 for 9% of values), so most draws at 3.3-3.9 are normal and replacement there is the clinician's choice. The eMAR order "Potassium Chloride Replacement (Critical Care and Oncology)"
(a protocol order with ordered doses 0/20/40/60/80 mEq) exists but covers few patients in the demo (124 records).

**Heparin** (209 events; rate change category by aPTT band):

| band (s) | increase | no change | decrease | hold | n | changed |
|---|---|---|---|---|---|---|
| < 40 | 8 | 6 | 0 | 0 | 14 | 0.57 |
| 40-50 | 21 | 1 | 1 | 1 | 24 | 0.96 |
| 50-55 | 11 | 7 | 0 | 1 | 19 | 0.63 |
| 55-60 | 6 | 13 | 0 | 0 | 19 | 0.32 |
| 60-70 | 0 | 44 | 0 | 0 | 44 | 0.00 |
| 70-80 | 0 | 33 | 0 | 1 | 34 | 0.03 |
| 80-90 | 0 | 14 | 7 | 0 | 21 | 0.33 |
| 90-100 | 0 | 2 | 4 | 0 | 6 | 0.67 |
| 100-120 | 0 | 2 | 6 | 0 | 8 | 0.75 |
| >= 120 | 0 | 4 | 1 | 15 | 20 | 0.80 |

Best monotone rule (categories): increase below 55, no change 55-90, decrease 90-120, hold above 120 (81%).
Median sizes: increase +3.6 units/kg/h below 40 and +1.9 at 40-55; decrease -1.8; hold, then restart about
-3.9. A bolus was given in 18% of events. The 50-60 and 80-90 bands are split within and between patients,
consistent with two nomograms with different target ranges (to be checked in the full data; the indication is
not stored in a structured field). Also excluded: 22 events in which the infusion was stopped for good inside
the window.

**Insulin, eMAR sliding-scale records** (328 events):

| band (mg/dL) | 0 | 2 | 4 | 6 | 8 | 10 | 12+ | odd | n | dosed |
|---|---|---|---|---|---|---|---|---|---|---|
| < 150 | 137 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 142 | 0.04 |
| 150-175 | 18 | 14 | 6 | 1 | 0 | 0 | 0 | 4 | 43 | 0.58 |
| 175-200 | 7 | 15 | 7 | 1 | 0 | 0 | 0 | 1 | 31 | 0.77 |
| 200-250 | 2 | 8 | 21 | 14 | 7 | 2 | 0 | 1 | 55 | 0.96 |
| 250-300 | 0 | 1 | 1 | 11 | 10 | 8 | 2 | 1 | 34 | 1.00 |
| 300-350 | 0 | 0 | 0 | 0 | 6 | 6 | 5 | 2 | 19 | 1.00 |
| >= 350 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 4 | 0.75 |

The standard scale "151-200: 2, 201-250: 4, 251-300: 6, ... units" (`sliding_scale(g, 150, 50, 2)`) matches
62.5%. Each patient's order has its own scale, which MIMIC-IV does not store; fitting
`k * ceil((g - c) / w)` per patient (c 100-200, w 30-50, k 1-6) matches 76%, with k = 1 for 8 patients, 2 for
11, 4 for 1. These scales are several schedules for comparable patients (requirement 5), but they must be
inferred. The ICU inputevents version (808 events) shows the same pattern with more zeros at 150-300
(29-60% undosed), partly because some doses are charted only in the eMAR.

## How the events are built (`adapter.py`)

- `events(protocol, source)` builds and caches the event table; `sample()` applies the analysis restrictions;
  `rule_check(protocol)` returns counts, the dose-by-score table, the step rules and match shares, share at the
  reference, R^2, outcome timing, and (heparin) category rule, (insulin) scale fit; `load_dose(protocol)`
  returns an `RDDSample` with extras `A`, `a_ref`, `dose_levels`, `schedule` (the fitted monotone step rule,
  a function of Q), `group` (icu / ward), `Z_names`, `patient` (cluster id), `A_rule`, `on_rule`, `gap_h`.
- Result time: the lab's `storetime` (median 1.3 h after the draw for potassium). Doses count from the result,
  not the draw. Potassium events with KCl given between the draw and the result are dropped (another value,
  e.g. a blood gas, drove that dose).
- Potassium doses: IV KCl from inputevents 225166 (additives to bags of >= 500 mL are maintenance fluid, kept
  as a covariate); oral KCl in the ICU and all ward KCl from the eMAR (sum of the given product doses). IV KCl
  in the eMAR during an ICU stay is dropped (it duplicates inputevents). Events within 12 h of CRRT are
  dropped. Potassium phosphate is not counted.
- Heparin rates: ICU from inputevents 225152 drip rows; wards from the eMAR "Heparin IV Infusion" records
  (each record sets the rate until the next, at most 12 h). An event needs the infusion running at the draw
  and the result; rate change = rate at the end of the window minus rate before the result, divided by the
  latest weight. A hold is a zero rate inside the window followed by a restart.
- Insulin: the sliding-scale order must be active (pharmacy `sliding_scale = "Y"`), no insulin drip within 1 h;
  ward events are impossible because MIMIC-IV has no ward finger-stick glucose.
- **Complete dose capture.** inputevents covers every ICU stay; the eMAR covers only admissions after it was
  adopted (181 of 275 demo admissions). `complete` requires eMAR records in the admission (ward events) or in
  the ICU stay itself (ICU events, for oral KCl). 1,506 of 2,418 potassium draws pass.
- **The score depends on earlier doses** (a low value is treated, then rechecked; the next outcome is the next
  score). X holds the previous value, log hours since it and the previous 24 h dose (potassium, insulin) or
  the current rate and previous change (heparin), so eta is the part of today's value not predicted from the
  last value and dose. Events of one patient form a sequence; the estimator should use patient clusters (as
  `dose_config.py` does). A per-event model treats each decision separately; it does not estimate the effect
  of a whole titration strategy.

## Projected n in the full MIMIC-IV

All 100 demo patients have an ICU stay (the demo took the first 100 random subject_ids whose anchor year group
is 2011-2013 or 2014-2016). Analysis events per demo patient, with 95% bootstrap intervals over patients:
potassium 9.81 (6.35-13.80), heparin 2.09 (1.08-3.39), insulin 8.08 (5.39-11.12), insulin eMAR 3.28.
MIMIC-IV v3.0 (counts on the v3.1 page) has 65,366 ICU patients (94,458 ICU stays) and 546,028 hospitalizations of
223,452 people in the hosp module (364,627 people in all; 2008-2022).
Scaled by ICU patients: potassium ~641,000 (415,000-902,000), heparin ~137,000 (71,000-222,000), insulin
~528,000 (352,000-727,000), insulin eMAR ~214,000 (109,000-341,000). Unknowns: the eMAR share differs by year
(the demo patients' anchor years are 2011-2016; earlier years likely have less eMAR, 2017-2022 more), and ward events of patients
never in an ICU are not counted (they would add potassium and heparin events, not insulin).

## Caveats

- The written protocols (BIDMC potassium replacement, heparin nomograms, sliding scales) are not in MIMIC-IV;
  the rules above are read from the data. For a known rule, the hospital's protocol documents would be needed.
- Care is sequential: the same patient contributes many events, and treatment at one event changes the score
  at the next.
- The heparin nomogram depends on the indication (target range), and doses are weight-based; weight is the
  latest charted (ICU chart, inputevents, OMR), with the first later weight as fallback.
- Insulin SC doses in inputevents do not say whether a dose came from the sliding scale or a fixed prandial
  order; the eMAR does (`administration_type`), but only from its adoption.
- Demo license: ODbL 1.0. Aggregate tables only here; no patient rows are copied.

## Getting the data

**Demo (done, open):** `python -m experiments.datasets.mimic_titration.download` (from `code/`) fetches the
34 files listed in SHA256SUMS.txt of https://physionet.org/files/mimic-iv-demo/2.2/ with plain curl into `data/raw/demo_2.2/` and checks
each SHA-256 (16.2 MB). No login, form or terms page; physionet.org/robots.txt allows all paths (checked
2026-11-12). Cite: Johnson, Bulgarelli, Pollard, Horng, Celi, Mark (2023), MIMIC-IV Clinical Database Demo
(version 2.2), PhysioNet, https://doi.org/10.13026/dp1f-ex47.

**Full MIMIC-IV (author steps).** Only credentialed users who complete the training and sign the data use
agreement can access the files (https://physionet.org/content/mimiciv/3.1/, v3.1 of 11 Oct 2024; license and
DUA: "PhysioNet Credentialed Health Data License 1.5.0" / "... Data Use Agreement 1.5.0").

1. Create a PhysioNet account: https://physionet.org/register/ (use an institutional e-mail address; PhysioNet
   news of 18 Jun 2026: https://physionet.org/news/2026/).
2. CITI course "Data or Specimens Only Research" (the training named at
   https://physionet.org/content/mimiciv/view-required-training/3.1/). Steps at
   https://physionet.org/about/citi-course/: register at https://about.citiprogram.org/, "Add affiliation"
   "Massachusetts Institute of Technology Affiliates" (not independent learner), answer the questionnaire,
   complete the "Data or Specimens Only Research" and "Conflicts of Interest" modules, then under Records ->
   "View-Print-Share" download the **training report** (not the certificate).
3. Upload the report at https://physionet.org/settings/training/ and apply for credentialed access at
   https://physionet.org/settings/credentialing/ (needs login). Students and postdocs give their supervisor as
   reference (https://mimic.mit.edu/docs/gettingstarted/). PhysioNet warns of "significant delays" in review
   (news of 18 Jun 2026).
4. Sign the DUA: https://physionet.org/sign-dua/mimiciv/3.1/.
5. Get the tables, either way:
   - **Download** (the site recommends the cloud; several GB, the disk has about 10 GB free): only these files,
     into `data/raw/mimiciv_3.1/hosp/` and `.../icu/`:
     hosp: `admissions, patients, transfers, labevents, emar, emar_detail, pharmacy, omr`;
     icu: `icustays, chartevents, inputevents` (each `.csv.gz`), e.g.
     `wget -N -c --user <physionet-user> --ask-password -P data/raw/mimiciv_3.1/hosp https://physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz`.
     The first `events(..., source="full")` call reads them in 2M-row chunks and caches small filtered parquet
     files in `data/processed/mimiciv_3.1/`; the raw files can then be moved off the disk
     (`experiments/datasets/relocate_data.py`).
   - **BigQuery** (https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery.html): add a Google account to the
     PhysioNet profile, click "Request access using Google BigQuery" in the Files section of the MIMIC-IV page
     (https://mimic.mit.edu/docs/gettingstarted/cloud/request.html; granted at once to credentialed users who
     signed the DUA); queries need a Google Cloud project with billing. Project `physionet-data`, datasets
     `mimiciv_v3_1_hosp`, `mimiciv_v3_1_icu` (the MIMIC-IV page; `mimiciv_hosp`, `mimiciv_icu` hold v3.1 since
     25 Nov 2024). Export these filtered tables
     as single `.csv.gz` files with the same names and columns (the adapter reads what it needs):

```sql
-- hosp/labevents.csv.gz
SELECT subject_id, hadm_id, itemid, charttime, storetime, valuenum FROM `physionet-data.mimiciv_v3_1_hosp.labevents`
WHERE itemid IN (50971, 51275, 50931, 50809, 50912);
-- icu/chartevents.csv.gz
SELECT subject_id, hadm_id, stay_id, itemid, charttime, valuenum FROM `physionet-data.mimiciv_v3_1_icu.chartevents`
WHERE itemid IN (225664, 226537, 226512, 224639, 226531);
-- icu/inputevents.csv.gz
SELECT subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom, orderid,
       ordercategoryname, patientweight, totalamount, totalamountuom, statusdescription
FROM `physionet-data.mimiciv_v3_1_icu.inputevents` WHERE itemid IN (225166, 227536, 225152, 223258, 223262, 223260, 223259);
-- hosp/emar.csv.gz
SELECT subject_id, hadm_id, emar_id, charttime, medication, event_txt FROM `physionet-data.mimiciv_v3_1_hosp.emar`
WHERE REGEXP_CONTAINS(LOWER(medication), r'^\s*(potassium chloride|heparin|insulin)');
-- hosp/emar_detail.csv.gz
SELECT d.subject_id, d.emar_id, d.parent_field_ordinal, d.administration_type, d.dose_due, d.dose_due_unit,
       d.dose_given, d.dose_given_unit, d.product_description, d.infusion_rate, d.prior_infusion_rate,
       d.infusion_rate_adjustment, d.infusion_rate_unit
FROM `physionet-data.mimiciv_v3_1_hosp.emar_detail` d JOIN `physionet-data.mimiciv_v3_1_hosp.emar` e USING (emar_id)
WHERE REGEXP_CONTAINS(LOWER(e.medication), r'^\s*(potassium chloride|heparin|insulin)');
-- hosp/emar_hours.csv.gz (eMAR coverage, needed because hosp/emar above is filtered)
SELECT DISTINCT subject_id, hadm_id, DATETIME_TRUNC(charttime, HOUR) AS charttime FROM `physionet-data.mimiciv_v3_1_hosp.emar`;
-- hosp/pharmacy.csv.gz
SELECT subject_id, hadm_id, starttime, stoptime, medication, sliding_scale, proc_type
FROM `physionet-data.mimiciv_v3_1_hosp.pharmacy` WHERE LOWER(medication) LIKE '%insulin%';
-- hosp/omr.csv.gz
SELECT subject_id, chartdate, result_name, result_value FROM `physionet-data.mimiciv_v3_1_hosp.omr`
WHERE result_name LIKE 'Weight%';
-- unfiltered: hosp/admissions, hosp/patients, hosp/transfers, icu/icustays
```

Then, from `code/`: `adapter.rule_check("heparin", source="full")`, and the estimator as above with
`argparse.Namespace(seed=0, x="linear", outcome=None, protocol="heparin", source="full")`.

## Files

`download.py` (demo), `adapter.py`, `dose_config.py` (estimator loader, registered at run time), `data/raw/demo_2.2/`
(git-ignored), `data/processed/demo_2.2/` (filtered tables and event caches, git-ignored).
