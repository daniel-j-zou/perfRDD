# HUD Qualified Census Tracts for the housing tax credit (waiting for the author's download)

| Field | Value |
|---|---|
| **Q** | (share of households with income below 60% of area median gross income, tract poverty rate) |
| **Threshold** | eligible if share >= 50% **or** poverty >= 25%. Designated tracts in a metro area may hold at most 20% of its population |
| **Treatment** | QCT designation: a 30% boost to the LIHTC eligible basis |
| **Rule class** | either cutoff on two scores; the population cap makes it fuzzy at the margin |
| **X** | tract characteristics from the decennial census |
| **Y** | LIHTC units placed in service after designation (HUD LIHTC database) |
| **n** | about 74,000 tracts per year |
| **Source** | HUD USER, public |

## Getting the data (author)

huduser.gov serves scripts a bot challenge (HTTP 202, checked 2026-10-27), so this is a
browser download. From https://www.huduser.gov/portal/datasets/qct.html, save the
**QCT designation data** spreadsheets (the data used to designate each year's QCTs) for
as many years as are listed into `data/raw/`.

Without your help, Claude can fetch:
- the designation lists for 2025 and 2026, from HUD's ArcGIS services;
- the LIHTC properties, also from HUD's ArcGIS services;
- tract poverty rates, from the ACS.

The sister design, the low-income community test for New Markets / Opportunity Zones,
is in `../us_tracts_lic/`.
