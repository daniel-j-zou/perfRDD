# SEER Oncotype DX: access steps and a study-plan draft (2026-11-11)

Checked on seer.cancer.gov on 2026-11-11 (access page, specialized-database list, requirements
page, the Oncotype request page). The two PDFs (DAC standard operating procedure, data use
agreement) were not readable by script; read them before signing. Exact database title:
**Invasive Breast Cancer with Oncotype DX Recurrence Score**. Request page:
<https://seer.cancer.gov/data/specialized/available-databases/breast-oncotype-dx-rs-request/>.

## Step A: SEER Research Plus (prerequisite; about two business days)

1. eRA Commons account tied to the institutional email, linked to a Login.gov account
   (free-mail addresses are rejected). After linking, wait two days for SEER to see it.
   An HHS account works instead.
2. Submit the Research Plus request at <https://seerdataaccess.cancer.gov/seer-data-access>
   (database type "SEER Research Plus"). It asks for a supervisor or mentor at the same
   institution; the signing-official field is optional here but is used again in Step B.
3. Acknowledge the SEER Research Data DUA and the Treatment Data Limitations (first-course
   treatment only; the chemotherapy field is first-course).
4. SEER states processing "within 2 business days". Install SEER*Stat (Windows only; a VM
   or a Windows machine is needed); sign in with Login.gov using the same email.

## Step B: the specialized database (Data Access Committee review; time not stated)

5. Register on the specialized-database site (Log In link at
   <https://seer.cancer.gov/data/specialized/>).
6. Requestor eligibility: a permanent employee at professor or senior-researcher level, with
   an institutional email, accountable for the DUA. Graduate students and post-docs cannot be
   the requestor but can be named collaborators. US-based researchers at US institutions only.
7. Institutional Signing Official: someone with authority to bind the institution (research
   office). They sign the attestation
   <https://seer.cancer.gov/data/SEER-Specialized-DBs-DUA-SOAttest.pdf>.
8. Upload with the request: contact information; the SEER*Stat username; the research use
   statement (objectives, study design, analysis plan, alignment with the data use
   limitations); the signed specialized-database DUA
   <https://seer.cancer.gov/data/SEER-Specialized-DBs-DUA-DAR.pdf>; the signing-official
   attestation.
9. Approval runs at most 12 months; renewal needs a progress report. At close-out confirm
   that all copies are destroyed. Cite the database name, release date and submission in
   any publication, plus Petkov et al., JNCI Monographs 2024;65:168-179 for the linkage.

## What the database holds (from the request page)

Invasive breast cancers diagnosed 2004-2019 linked to Exact Sciences test results; follow-up
through 2020; about 1.63 million cases in the linkage, 364,292 with a result. Fields:
`OncotypeDX Breast RS (2004-2019)` (0-100), reason no score linked, year and month of the
test report, months from diagnosis to test. Two versions: county-level attributes
(17 registries: 187,608 linked tumors; 22 registries limited-field: 343,392; 22 registries
without IL and MA: 293,668) and census-tract attributes (no geography; 185,351 / 290,558).
The 22-registry files lack the regional-nodes fields; the 17-registry file has them.
Everything else is the Research Plus record (age, grade, size, ER/PR/HER2, histology,
first-course treatment including chemotherapy, survival months, vital status, cause of death).

Request the **17-registry county-level version** (nodes fields present) unless the extra
cases matter; the adapter (`adapter.py`) reads either.

## Research use statement: draft

**Objectives.** To estimate how the probability of five-year survival after a diagnosis of
hormone-receptor-positive, HER2-negative, node-negative invasive breast cancer depends on the
Oncotype DX Recurrence Score and on receipt of adjuvant chemotherapy, and to evaluate, on the
survival-probability scale, where a chemotherapy recommendation threshold on the score would
best be placed. Guidelines after TAILORx (Sparano et al., NEJM 2018) advise chemotherapy
above a score of 25 and endocrine therapy alone at 11-25 for women over 50; the study asks
whether the observational record is consistent with that line and how far an alternative line
would change expected survival.

**Study design.** Retrospective cohort from the SEER specialized database "Invasive Breast
Cancer with Oncotype DX Recurrence Score" (17 registries, county-level attributes). Inclusion:
women aged 18-90 diagnosed 2010-2015 (five years of follow-up by 2020) with a linked score,
ER-positive, HER2-negative, regional nodes negative, first primary. Exposure: chemotherapy as
first-course treatment, and the guideline rule 1{score above 25} as an intention-to-treat
assignment. Outcome: death within 60 months of diagnosis (all cause, and breast-cancer
specific as a secondary outcome). Covariates: age, grade, tumor size, PR status, histology,
year of diagnosis, registry, and county attributes. The score's dependence on the covariates
and the jump in chemotherapy receipt at 25 (and at 18 and 31 for earlier guidelines) are
checked first.

**Analysis plan.** (i) Descriptive: chemotherapy receipt and five-year mortality by score
bins. (ii) A regression-discontinuity estimate of the effect of the guideline rule on
chemotherapy receipt and on five-year mortality at 25, with local linear fits and standard
bandwidth selectors. (iii) A semiparametric outcome model (a logistic model with a flexible
function of the score and covariate interactions, fitted separately with sample splitting)
used to compute, for each candidate threshold between 11 and 40, the expected five-year
survival of the population under the rule "chemotherapy above the threshold", and the
threshold that maximizes it with and without a per-patient cost of chemotherapy. (iv)
Bootstrap confidence intervals for the optimal threshold; sensitivity to the age split at 50,
to the follow-up window and to the inclusion criteria. No individual-level data leave the
approved SEER*Stat environment; only aggregated estimates and fitted curves are reported.

**Alignment with data use limitations.** Research use only; no attempt to identify
individuals or facilities; treatment fields used as first-course treatment with the stated
limitation acknowledged; results reported in aggregate; the database cited as required.
