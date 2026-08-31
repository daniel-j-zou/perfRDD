# GPA welfare-menu hard-trim results

## Bottom line

A prespecified menu of 16 welfare outcomes does not identify an interior GPA-probation
threshold. Every no-cost full-sample and cross-fit optimum is at a policy-grid boundary.
On the expanded policy grid `[-1.2, 1.2]`, all 64 combinations of 16 outcomes and direct
costs `{0, 0.025, 0.05, 0.10}` also select a boundary.

Reasonable welfare choices therefore clarify which boundary is preferred, but they do
not create a stable interior optimum. Selecting a more extreme calibration because it
does produce one would be outcome-driven tuning.

## Reproduction

```sh
/Users/zoudj/miniconda3/envs/srsd/bin/python \
  -m experiments.scripts.gpa_welfare_hard_trim
```

The complete machine-readable output and figures are written to the ignored directory
`experiments/runs/gpa_welfare_hard_trim/`. The prespecified outcomes and formulas are in
`welfare.py`.

## Why the menu is structured this way

The original academic-probation study reports the same basic tension found here:
probation discourages some students from returning while improving GPA among those who
remain. It also reports negative graduation effects for some groups. The original
replication code names missing-GPA sensitivity variables assigning absolute GPAs 0.0,
0.8, 0.9, and 1.1, but provides no argument that any one value is the correct welfare
calibration.

No subsequent GPA record is not the same state as leaving:

- 3,780 students have no subsequent GPA record;
- 1,929 of those are coded as voluntary leavers;
- 1,851 are not coded as voluntary leavers; and
- another 246 leavers nevertheless have a subsequent GPA record.

Consequently, a missing-record assignment and a leave penalty are kept distinct. An
assignment `a` followed by a penalty on every missing record is redundant because only
`a - penalty` enters the outcome. The status-adjusted stress tests instead apply a small
penalty or bonus by observed leave/return status.

## Prespecified menu and estimates

`Hard alpha` is the hard-window average treatment-effect nuisance under the full-sample
ridge-0.001 display specification. `CF alpha` is the five-fold unregularized estimate.
The ridge grid is `{0, 0.0001, 0.001, 0.01}`. All rows use exact hard trimming with
`eps=0.1` and fixed nuisance support `(-2, 0)`.

| # | Welfare outcome | Role | Hard alpha | CF alpha | No-cost phi* |
|---:|---|---|---:|---:|---:|
| 1 | Fall-year-2 enrollment | Primary direct | -0.0833 | -0.0852 | -0.6 |
| 2 | Not voluntarily leaving | Primary direct | -0.0165 | -0.0178 | -0.6 |
| 3 | Any subsequent GPA record | Primary direct | -0.0277 | -0.0293 | -0.6 |
| 4 | Year-2 credits earned | Primary direct | -0.3719 | -0.3757 | -0.6 |
| 5 | Good standing in year 2 | Secondary; coding provenance open | -0.2584 | -0.2589 | -0.6 |
| 6 | Composite GPA; no record = 0.0 | Physical lower bound | 0.1433 | 0.1447 | 0.6 |
| 7 | Composite GPA; no record = 0.8 | Inherited sensitivity | 0.1662 | 0.1681 | 0.6 |
| 8 | Composite GPA; no record = 0.9 | Inherited sensitivity | 0.1691 | 0.1711 | 0.6 |
| 9 | Composite GPA; no record = 1.1 | Inherited sensitivity | 0.1748 | 0.1769 | 0.6 |
| 10 | Composite GPA; no record = 1.5 | Cutoff benchmark | 0.1863 | 0.1886 | 0.6 |
| 11 | `a=.8` minus `0.10 × leave` | Stress test | 0.1645 | 0.1663 | 0.6 |
| 12 | `a=.8` minus `0.25 × leave` | Stress test | 0.1619 | 0.1637 | 0.6 |
| 13 | `a=1.1` minus `0.10 × leave` | Stress test | 0.1731 | 0.1751 | 0.6 |
| 14 | `a=1.1` minus `0.25 × leave` | Stress test | 0.1705 | 0.1725 | 0.6 |
| 15 | `a=.8` plus `0.10 × fall return` | Stress test | 0.1579 | 0.1596 | 0.6 |
| 16 | `a=.8` plus `0.25 × fall return` | Stress test | 0.1454 | 0.1468 | 0.6 |

The full-sample and cross-fit estimates are close in every row. Ridge sensitivity does
not change any no-cost policy direction. The direct progression outcomes all favor the
lower policy boundary, while every physical or modest status-adjusted GPA composite
favors the upper boundary.

## Skeptical audit

An independent skeptic agent reviewed the menu and ran a broader scratch audit with
constant direct costs from -2 to 2 and a policy grid `[-1.2, 1.2]`. It found no interior
global optimum for the GPA, persistence, or recording outcomes: the recommendation
jumps from one boundary to the other as costs change.

The skeptic identified the following publication red flags:

- tuning a missing-GPA value, leave penalty, treatment cost, or policy domain to obtain
  an interior result;
- calling the earlier penalty-five break-even a physical GPA value—it implies a welfare
  outcome far outside the GPA scale and is only an extreme stress test;
- treating the selected observed-GPA effect as a full-population effect;
- double-counting return, recording, and leaving in one composite;
- mixing absolute and cutoff-centered GPA units;
- presenting correlated welfare variants as independent confirmations;
- treating the pilot-fixed nuisance support as confirmatory; or
- claiming policy optimality before boundary-aware inference and confidence sets exist.

## Interpretation and next work

This dataset supports a robust substantive tradeoff: probation improves the GPA-based
outcomes while reducing multiple measures of continuation and academic progress. It
does not support a data-driven interior threshold under the current linear welfare class.

An interior result would require an independently justified nonlinear welfare component,
capacity constraint, or policy cost—not a calibration chosen after inspecting `phi*`.
The publication-ready next steps are to implement the application influence-function
variance, report utility gaps/regret and policy confidence sets, document the year-2 good
standing construction, and show a two-dimensional frontier that assigns separate values
to leavers and non-leavers without a GPA record.
