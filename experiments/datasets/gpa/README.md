# GPA — academic probation (Lindo, Sanders, Oreopoulos 2010)

| Field | Value |
|---|---|
| **Q** | `dist_from_cut` (first-year GPA distance from probation cutoff) |
| **Threshold** | 0 |
| **Treatment** | `1{Q < 0}` — below the cutoff = on probation |
| **X** | `hsgrade_pct`, `totcredits_year1`, `loc_campus1`, `loc_campus2`, `male`, `bpl_north_america`, `age_at_entry`, `english` |
| **Y** | `nextGPA`: GPA at the next recorded evaluation minus the applicable cutoff |
| **n** | 40,582 with `nextGPA`; 44,362 in the full processed sample |
| **Citation** | Lindo, Sanders, Oreopoulos (2010), AEJ:Applied |

## Data

The processed CSV `Dep_Data/final_processed_data.csv` and the original AEJ
replication package live under `Dep_Data/`. The `.csv` file is git-LFS-tracked
(see the repo-level `.gitattributes`); run `git lfs pull` after cloning.

The original data is governed by the AEJ replication terms; see
`Dep_Data/LICENSE.txt`.

## Layout

- `adapter.py` — exports `load() -> RDDSample`. Sole required entry point.
- `redesign.py` — full-population persistence outcomes, an explicitly selected
  GPA diagnostic, and a transparent composite-outcome sensitivity analysis.
- `HARD_TRIM_RESULTS.md` — reproducible status, verified point estimates, and limitations
  for the exact hard-trim application.
- `welfare.py` — prespecified 16-outcome welfare menu separating direct outcomes,
  inherited missing-GPA sensitivities, and status-adjusted stress tests.
- `WELFARE_RESULTS.md` — full welfare-menu results, skeptical audit, and publication
  limitations.
- `../../scripts/gpa_redesign_hard_trim.py` — exact hard-support-trimmed application.
  It reports a locked full-sample ridge grid and an unregularized five-fold cross-fit
  robustness estimate using one pilot-fixed nuisance support for every outcome.
- `Dep_Data/` — raw + intermediate files (preserved from the original
  replication package layout).
- `general/` — legacy GPA-specific analysis scripts and notebooks. New
  per-dataset analysis should go in `analysis/` or `notes/`.

## Outcome-selection warning

Academic probation affects whether students continue at the university, so
the observed-`nextGPA` sample is selected after treatment. It must not be
described as a full-population policy outcome. The redesign keeps three
questions separate:

1. full-population persistence (`fallreg_year2`, `1-left_school`, and whether
   a subsequent GPA is recorded);
2. subsequent GPA among students with a recorded evaluation, labeled as a
   selected diagnostic; and
3. a full-population composite that assigns the no-subsequent-GPA state a
   stated absolute GPA value before subtracting the student's cutoff.

The composite is a sensitivity estimand, not a recovered missing GPA. The
default assumed absolute GPAs (0.0, 0.8, 0.9, and 1.1, plus a 1.5 stress test)
are motivated by imputed-outcome variable names retained in comments in the
original replication do-file. A second sensitivity holds the no-record GPA at
zero and subtracts an explicit GPA-equivalent welfare penalty. The penalty is
not a literal GPA; it quantifies how strongly a planner values the loss of a
subsequent academic record.

## Hard-trim application status

The application uses the exact indicator
`1{l_hat <= eta_hat <= u_hat}` with `eps=0.1`; no smooth gate remains. The nuisance
spline support `(-2, 0)` is held fixed across outcomes and specifications. It was rounded
outward from an August 2026 pilot diagnostic and is therefore exploratory rather than a
confirmatory scientific choice. Results include full-sample ridge scales 0, 0.0001,
0.001, and 0.01 plus a five-fold unregularized cross-fit check. The code currently
reports point estimates only, not the theorem's boundary-aware influence-function
variance.

The broader welfare audit is run by `../../scripts/gpa_welfare_hard_trim.py`. It reports
the entire prespecified menu and an expanded-policy-grid cost audit. It must not be used
to select whichever valuation happens to produce a preferred threshold.
