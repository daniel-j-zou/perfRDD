# CalEnviroScreen 4.0 and SB 535 disadvantaged communities (waiting for the author's download)

| Field | Value |
|---|---|
| **Q** | CalEnviroScreen score = Pollution Burden score x Population Characteristics score (each 0-10) |
| **Threshold** | top 25% of tract scores statewide (CalEPA's 2022 SB 535 designation, with a few other inclusion categories) |
| **Treatment** | designated disadvantaged community: priority for California Climate Investments funding |
| **Rule class** | a composite with chosen weights; as two scores, a curved (product) boundary |
| **X** | tract characteristics not used in the score (from the decennial census) |
| **Y** | Climate Investments dollars per tract; later environmental or health indicators |
| **n** | about 8,000 tracts |
| **Source** | OEHHA, public |

## Getting the data (author)

oehha.ca.gov serves scripts a bot check (checked 2026-10-27), so this is a browser
download. From https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40, save the
results spreadsheet (Excel) and its data dictionary into `data/raw/`.

## Caveats

- The score is built from public indicators, so given those indicators it has no
  noise. Use covariates outside the score.
- The sample is small (about 8,000 tracts).
