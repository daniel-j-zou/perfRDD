# Workers' compensation weekly benefits, Kentucky and Michigan (Meyer, Viscusi and Durbin 1995)

| Field | Value |
|---|---|
| **Q** | log pre-injury weekly wage (`prewage`, 1982 dollars) |
| **Threshold** | the cap kink: 1.5 x the state maximum |
| **Treatment** | Dose A = min(2/3 x wage, cap(state, period)), 1982 dollars. Caps (median benefit of high earners): KY 149.9 before, 246.8 after; MI 218.3 before, 324.0 after |
| **Rule class** | Kinked schedule (linear, then capped); two schedules per state (before and after a cap increase) |
| **X** | age (+ square), sex, married, hospitalised, injury site (7), industry, injury type, state, after-change |
| **Y** | `ldurat`, log weeks of temporary total disability benefits (also `durat`, weeks; max 182, 1.2% at the max) |
| **n** | 7,150 claims: KY 5,626 (2,938 before, 2,688 after), MI 1,524 |
| **Source** | R package `wooldridge` 1.4-7 (CRAN, GPL-3), data frame `injury`; "Professor Meyer kindly provided the data" |

**Fit: weak for the dose design (screened 2026-11-12, not run through the estimator).**
- **The rule holds.** Low earners: benefit = 2/3 of the wage within $1 for 80.8%, within 5%
  for 88.1%. High earners: within 10% of the state-period cap for 89.1% (within 5%: 48.8%;
  real caps vary because the files deflate by the injury date, which is not given).
- **The score has a gap.** The sample keeps low earners (KY up to $250 a week, MI $225-325)
  and high earners (KY from $350, MI from $625); middle earners, who were partly affected,
  were dropped by MVD. R^2 of log wage on X is 0.35, but only 35% (KY) and 21% (MI) of high
  earners have eta inside the low earners' 2.5-97.5% eta range, so doses barely transport
  between the groups.
- **What identifies the effect** is the cap increase for high earners against low earners,
  MVD's difference in differences: mean log duration rose 1.38 -> 1.58 for KY high earners
  and 1.13 -> 1.13 for low earners (MI 1.58 -> 1.87 and 1.41 -> 1.51). The dose design adds
  little beyond that.
- **Outcome** is continuous (log weeks), but duration is capped at 182 weeks for 1.2%.

## Getting the data

No account or terms needed. CRAN's robots.txt disallows only package DESCRIPTION pages.

```bash
mkdir -p data/raw/cran_wooldridge data/processed
curl -o data/raw/cran_wooldridge/wooldridge_1.4-7.tar.gz https://cran.r-project.org/src/contrib/wooldridge_1.4-7.tar.gz
tar -xzf data/raw/cran_wooldridge/wooldridge_1.4-7.tar.gz -C data/raw/cran_wooldridge wooldridge/data/injury.RData wooldridge/man/injury.Rd
Rscript --vanilla -e "e <- new.env(); load('data/raw/cran_wooldridge/wooldridge/data/injury.RData', envir=e); write.csv(e\$injury, 'data/processed/injury.csv', row.names=FALSE)"
```

## Caveats

- `totmed` (medical costs) is measured after the injury and is not used as a covariate.
- Other workers' compensation microdata found by the 2026-11-12 desk search: Oregon's
  administrative claims (used by Hansen, Nguyen and Waddell, IZA DP 10621; obtained from the
  state agency, no public request route found), the federal FECA public-use file (no
  pre-injury wage), and WCRI and NCCI claim data (no academic access route found).
