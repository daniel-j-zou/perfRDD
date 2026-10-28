# Kenya tracking (Duflo, Dupas & Kremer 2011) (waiting for the author's download)

| Field | Value |
|---|---|
| **Q** | baseline test score |
| **Threshold** | in each tracking school, the school's median baseline score: students below it go to the lower section |
| **Treatment** | assignment to the upper section |
| **Rule class** | group-specific cutoffs (one per school) |
| **X** | baseline student characteristics (gender, age), school |
| **Y** | endline test scores |
| **Source** | Harvard Dataverse doi:10.7910/DVN/LWFH9U (CC0, with a guestbook); AEA package doi:10.3886/E112446V1 |
| **Citation** | Duflo, Dupas & Kremer (2011), AER 101(5): 1739-74 |

## Getting the data (author)

The Dataverse download asks for personal details in a guestbook form, which Claude
cannot fill in. Download `student_test_data.tab` and `student_pres_data.tab`, plus their
codebooks and `READ_ME.pdf`, from
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LWFH9U into
`data/raw/`.

## Caveats

- Small, with few covariates. Tracking was randomized across schools, so only the
  tracking schools have the cutoff.
