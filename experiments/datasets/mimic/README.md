# MIMIC-IV — ICU severity score RDD (stub, DUA-restricted)

| Field | Value |
|---|---|
| **Q** | day-1 severity score (SOFA / SAPS-II / APACHE-IV — pick one) |
| **Threshold** | study-specific (e.g. SOFA >= 10 for high-mortality cohort) |
| **X** | age, sex, admission diagnosis, comorbidities, vitals on admission |
| **Y** | ICU length of stay (continuous) or ventilator-free days |
| **Source** | https://physionet.org/content/mimiciv/ |

## Access (one-time, ~1 day)

1. Create a PhysioNet account at https://physionet.org/register/.
2. Complete CITI "Data or Specimens Only Research" training (free, ~3–4 hours).
3. Submit the MIMIC-IV DUA on the dataset page; it auto-approves once your
   CITI cert is on file.
4. Download via the PhysioNet web interface or, more practically, via
   Google BigQuery (`physionet-data.mimiciv_*`).
5. Build a single-row-per-ICU-stay CSV with at minimum:
   - severity score on day 1
   - X covariates above
   - LOS or VFD outcome
   Save as `data/raw/mimic.csv` and update the adapter.

## eICU alternative

If PhysioNet credentialing is in place but you want a larger sample,
eICU-CRD (~200K stays, also at PhysioNet) follows the same access path.
