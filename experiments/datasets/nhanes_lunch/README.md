# NHANES school-lunch income eligibility (2005-2016)

| Field | Value |
|---|---|
| **Q** | `INDFMPIR`, family income-to-poverty ratio (0-5, top-coded) |
| **Threshold** | 1.85: free or reduced-price eligibility (free below 1.30); Schanzenbach (2009) design |
| **Treatment** | `1{Q <= 1.85}` (eligibility, ITT) |
| **X** | age, age^2, female, race/ethnicity, household size, household reference person's education / age / married status, cycle dummies |
| **Y** | free/reduced-price lunch receipt; school lunches per week; BMI |
| **n** | 14,307 children aged 5-18 attending school, cycles D-I |
| **Source** | CDC NCHS public files, no registration |

## Getting the data

```bash
cd experiments/datasets/nhanes_lunch/data/raw
for pair in 2005:D 2007:E 2009:F 2011:G 2013:H 2015:I 2017:J; do y=${pair%:*}; s=${pair#*:}
  for f in DEMO DBQ BMX; do curl -sSfLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/$y/DataFiles/${f}_${s}.xpt"; done
done
```

That is 21 files, about 67 MB. Cycle J is downloaded but unused, because its
household-education and age variables were recoded.

## Results (2026-09-29)

Local RD at 1.85 (bandwidth 0.5):
- Receipt: +0.028 (SE 0.037). There is no detectable first stage in survey income.
- Lunches per week: +0.30 (SE 0.15).
- BMI: -0.50 (SE 0.43).
- Only about 2,400-2,900 children fall within the band.

Global fits:
- Receipt: overstated (+0.14 / +0.16 vs +0.028).
- Lunches per week: matches (+0.29 / +0.28 vs +0.30), but the optimum is a boundary
  (subsidize everyone).
- BMI: alpha-only has an "interior" optimum at 1.30 that beats the boundaries by only
  0.006 BMI units, which is noise. DS is a boundary.
- SD(eta) falls from 1.3 to 0.64 across deciles of T; the ratio is bounded at 0 and
  top-coded at 5.

Not a usable application: the sample is too small and survey income does not track
eligibility.
