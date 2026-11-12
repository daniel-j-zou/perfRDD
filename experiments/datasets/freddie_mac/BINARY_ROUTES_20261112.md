# Freddie Mac credit-score band edges, binary-outcome routes (Claude, 2026-11-12)

Exploratory. Runner: `experiments/scripts/binary_freddie_lines.py` (results in
`experiments/runs/binary_freddie_lines/e{660,680,720,740}/`). The folder has no `adapter.py`, so the runner builds
the frame from the sample files with `screen.ORIG`. Cache: `data/cache/freddie_default36_2010_2019.parquet`
(untracked, not covered by the ignore rules; do not commit).

## Sample

- Originations 2010-2019, 50,000 loans per year: 500,000; 496,854 have performance records.
- Valid score (300-850), LTV (1-105), DTI (0-65) and performance: n = 418,642.
- Y: default = any month with age <= 36 and delinquency status >= 3 (90+ days) or RA, or a zero-balance
  code 02, 03 or 09 within 36 months. Default rate 1.995 percent (90+ days alone: 1.995 percent, so the
  termination codes add nothing). Used as Y = 1 - default (survival). Prepaid loans count as no default.
- X: LTV, DTI, log loan amount, two borrowers, first-time buyer, dummies for purpose, property type,
  occupancy, channel, state, origination year; standardized. R-squared of the score on X = 0.101
  (the earlier screen found 0.117 with more variables). sd(score) = 43.9, mean 753.6.
- LTV 75-80 subsample used for all edges: n = 147,658, default rate 1.82 percent. Effect covariate: DTI.
- Settings: candidates edge-30 to edge+30, eps = 0.1, mirror=False, survival=True, grid 121, seed 1.
  Treated = score at or above the edge.
- Base default rate near each edge is 2.4-5 percent, so the binary outcome is rare and effects are small.

## Routes (n = 147,658 in every row; tau in probability units)

| edge | route | cost | treated | window share | optimum | boundary | tau -30 | tau -15 | tau 0 | tau +15 | tau +30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 680 | logit_emp | 0 | 0.935 | 0.066 | 691.2 | no | -0.0343 | -0.0162 | -0.0047 | +0.0013 | +0.0039 |
| 680 | lpm | 0 | 0.935 | 0.066 | 710.0 | yes | +0.0011 | +0.0004 | -0.0004 | -0.0015 | -0.0028 |
| 680 | logit_emp | 0.005 | 0.935 | 0.066 | 710.0 | yes | -0.0393 | -0.0212 | -0.0097 | -0.0037 | -0.0011 |
| 680 | lpm | 0.005 | 0.935 | 0.066 | 710.0 | yes | -0.0039 | -0.0046 | -0.0054 | -0.0065 | -0.0078 |
| 720 | logit_emp | 0 | 0.772 | 0.175 | 750.0 | yes | -0.0153 | -0.0079 | -0.0040 | -0.0020 | -0.0009 |
| 720 | lpm | 0 | 0.772 | 0.175 | 750.0 | yes | +0.0064 | +0.0015 | -0.0040 | -0.0105 | -0.0183 |
| 720 | logit_emp | 0.005 | 0.772 | 0.175 | 750.0 | yes | -0.0203 | -0.0129 | -0.0090 | -0.0070 | -0.0059 |
| 720 | lpm | 0.005 | 0.772 | 0.175 | 750.0 | yes | +0.0014 | -0.0035 | -0.0090 | -0.0155 | -0.0233 |
| 740 | logit_emp | 0 | 0.659 | 0.243 | 721.2 | no | -0.0064 | +0.0012 | +0.0039 | +0.0037 | +0.0024 |
| 740 | lpm | 0 | 0.659 | 0.243 | 710.0 | yes | +0.0087 | +0.0062 | +0.0038 | +0.0011 | -0.0030 |
| 740 | logit_emp | 0.005 | 0.659 | 0.243 | 770.0 | yes | -0.0114 | -0.0038 | -0.0011 | -0.0013 | -0.0026 |
| 740 | lpm | 0.005 | 0.659 | 0.243 | 770.0 | yes | +0.0037 | +0.0012 | -0.0012 | -0.0039 | -0.0080 |
| 660 | logit_emp | 0 | 0.971 | 0.029 | 668.3 | no | -0.0147 | -0.0114 | -0.0031 | +0.0022 | +0.0068 |
| 660 | lpm | 0 | 0.971 | 0.029 | 630.0 | yes | +0.0129 | +0.0057 | +0.0030 | +0.0008 | -0.0016 |
| 660 | logit_emp | 0.005 | 0.971 | 0.029 | 685.1 | no | -0.0197 | -0.0164 | -0.0081 | -0.0028 | +0.0018 |
| 660 | lpm | 0.005 | 0.971 | 0.029 | 690.0 | yes | +0.0079 | +0.0007 | -0.0020 | -0.0042 | -0.0066 |

("optimum 710.0 / 750.0 / 770.0 / 630.0 / 690.0 boundary" values are the policy-set ends: grid edge, not an interior maximum.)
Gain over the deployed edge is at most 0.0013 (lpm, 720, cost 0.005) and 0.0002-0.0006 elsewhere.
Effect at the cutoff averaged over the window (logit_emp, cost 0): +0.011 (680), -0.003 (720), +0.004 (740), +0.001 (660).
Seconds: logit_emp 130-230 s, lpm 40-60 s per run. No failures.

## Local jump in the default rate at the edge (LTV 75-80, uniform kernel, local linear, HC1)

Jump = rate above minus rate below; a beneficial higher band would give a negative number.

| edge | bw | n | jump | s.e. | mean below | mean above |
|---|---|---|---|---|---|---|
| 680 | 10 | 7,343 | -0.0022 | 0.0104 | 0.0512 | 0.0439 |
| 680 | 20 | 14,993 | +0.0015 | 0.0073 | 0.0508 | 0.0401 |
| 720 | 10 | 15,530 | +0.0000 | 0.0050 | 0.0270 | 0.0241 |
| 720 | 20 | 31,008 | +0.0041 | 0.0036 | 0.0290 | 0.0230 |
| 740 | 10 | 17,862 | +0.0053 | 0.0040 | 0.0220 | 0.0178 |
| 740 | 20 | 37,180 | +0.0000 | 0.0029 | 0.0230 | 0.0154 |
| 660 | 10 | 4,076 | +0.0266 | 0.0148 | 0.0583 | 0.0504 |
| 660 | 20 | 8,100 | +0.0099 | 0.0106 | 0.0682 | 0.0508 |

No jump differs from zero by two s.e. (largest: 660, bw 10, 1.8 s.e., the wrong sign).

## Bunching: loan counts by score (LTV 75-80; all LTV in brackets)

| edge | edge-5 ... edge-1 | edge ... edge+4 |
|---|---|---|
| 680 | 288, 294, 254, 296, 286 (834, 809, 818, 827, 876) | 418, 419, 444, 493, 455 (1256, 1305, 1259, 1395, 1370) |
| 720 | 724, 831, 715, 630, 661 (1983, 2232, 1937, 1799, 1879) | 868, 813, 878, 795, 849 (2307, 2174, 2372, 2221, 2372) |
| 740 | 809, 818, 843, 846, 848 (2191, 2235, 2362, 2409, 2473) | 906, 907, 988, 952, 972 (2525, 2555, 2638, 2556, 2660) |
| 660 | 178, 178, 135, 176, 140 (483, 451, 448, 512, 442) | 246, 233, 229, 237, 255 (668, 660, 664, 685, 736) |

Count at edge-1 / edge / edge+1: 680: 286 / 418 / 419; 720: 661 / 868 / 813; 740: 848 / 906 / 907; 660: 140 / 246 / 233.

## Reading

- The score distribution has a step at 680, 660 and 720: 286 to 418 (+46 percent) at 680, 140 to 246 at 660,
  661 to 868 at 720, and the higher count persists above the edge rather than being a one-point spike. The step
  is a level shift, which fits borrowers or lenders sorting just above the band edge (or an origination
  mix that changes at the edge), so the no-bunching condition fails at 660, 680 and 720. 740 is smooth
  (848 to 906, in line with the slope), the only edge that passes.
- The local jump in the default rate is zero within noise at every edge (s.e. 0.3 to 1.5 points), so the
  data do not show a visible effect of the better band on default at the edge. At the base rate of 2-5
  percent, a fee difference acting through rates of about 8 bp would not be expected to move default
  by more than a few tenths of a point, below the detection limit.
- Routes: all 8 lpm runs end at the boundary of the candidate set (tau is close to linear). logit_emp has
  interior optima in 4 of 8 runs (680, 740 and 660 at cost 0; 660 at cost 0.005); the other 4 are at the boundary.
  Where interior, tau crosses zero inside the set (680: between 0 and +15; 660: between 0 and +15; 740: between
  -30 and -15), but lpm has the opposite slope at those edges, and the gain over the deployed edge is only
  0.00001-0.0002 in default probability.
- The covariates explain 10 percent of the score, so eta is nearly the whole score and the model's
  separation of Q = gamma'X + eta is weak; the fee grid also changed by year and LTV band, which the
  runner does not model.

Verdict: not a usable setting. The routes cross zero inside the candidate set only for logit_emp, the
crossing is not supported by the local jump (zero at every edge) or by the other route, and three of four edges show
bunching at the edge. 740 has no bunching but the weakest evidence of any effect (jump +0.005 at bw 10, 0.000 at bw 20).
