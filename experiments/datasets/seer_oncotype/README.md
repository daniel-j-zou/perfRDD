# SEER breast cancer with the Oncotype DX Recurrence Score (chemotherapy line at 25)

| Field | Value |
|---|---|
| **Q** | 21-gene Recurrence Score (0-100), reported for hormone-receptor-positive, HER2-negative early breast cancer |
| **Threshold** | 25: guidelines after TAILORx (Sparano et al., NEJM 2018) advise chemotherapy plus endocrine therapy above 25 and endocrine therapy alone at 11-25 for women over 50; before 2018 the lines were 18 and 31 |
| **Treatment** | chemotherapy (fuzzy: its use rises with the score rather than jumping; `1{Q > 25}` as the ITT rule, chemotherapy receipt from the treatment fields) |
| **X** | grade, tumor size, PR status, ER level, age, histology, nodal status, year, registry; county or tract attributes in the extended versions |
| **Y** | death within 5 years (all-cause, about 5-10% in tested cohorts); breast-cancer-specific death (2-3%, rare); distant recurrence is not in SEER |
| **n** | 293,668 tumors with a score, diagnoses 2004-2019, follow-up through 2020 (22-registry file) |
| **Effect** | TAILORx: no chemotherapy benefit at RS 11-25 overall, benefit above 25, some benefit at 16-25 for women 50 or younger; the benefit rises with the score, the cleanest case of an effect that crosses a cost |
| **Cost** | chemotherapy's monetary and toxicity cost per patient |
| **Status** | waiting for the author's access application (2026-11-11); no adapter yet |

## Access (author; needs an institutional account, US-based researchers only)

1. Obtain SEER Research Plus access: <https://seer.cancer.gov/data/access.html> (eRA Commons
   or HHS account, institutional email; "processed within 2 business days").
2. Request the specialized database "Invasive Breast Cancer with Oncotype DX Recurrence
   Score": <https://seer.cancer.gov/data/specialized/available-databases/breast-oncotype-dx-rs-request>
   (a study plan and a signed data use agreement; the treatment fields need the separate
   treatment-data limitations agreement).
3. Export from SEER*Stat as a case listing with: Recurrence Score, age at diagnosis, grade,
   tumor size, ER and PR status, HER2, histology, nodal status, chemotherapy (first course),
   year of diagnosis, registry, survival months, vital status and cause of death. Save the
   export under `data/raw/` (restricted data: never commit; keep the agreement's terms).

## Plan for the adapter

Sample: women with a score, node-negative, HR-positive, HER2-negative, diagnosed 2010-2015
(five years of follow-up by 2020). Q = Recurrence Score, `threshold=25.5`,
`treatment_rule = Q > 25`, Y = survival to 5 years (binary), X as above with missing flags.
Policy set about 11-40, eps 0.1, effect covariates age and grade. First check: the
first-stage jump in chemotherapy receipt at 25 (and at 18 and 31 for 2004-2017), since the
rule is a guideline. Run with `experiments/scripts/binary_outcome_routes.py` through
`build_sample`. Published SEER-RS survival work: Breast Cancer Research 2019,
<https://link.springer.com/content/pdf/10.1186/s13058-019-1190-4.pdf>.
