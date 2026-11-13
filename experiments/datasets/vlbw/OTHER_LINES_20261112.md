# Other birth-weight lines, same estimator (Claude, 2026-11-12)

Script `experiments/scripts/binary_vlbw_lines.py`; outputs `experiments/runs/binary_vlbw_lines/line_<L>.json`
(git-ignored). Pooled NCHS 1995-2002, Y = survival to one year, D = 1{weight < L}, score Q = -weight,
X = standardized X_COLS, effect covariate gestation, eps = 0.1, candidate thresholds L-300 to L+300 g,
routes `logit_emp` and `lpm`, costs 0 and 0.01 (tau is net of the cost).

**Status: all runs finished (1000, 1250, 2000, 2500 g, plus 1500 g as a reference). No failures; no line needed the narrower +-200 set.**

## Sample

| L | weight window | n | treated share | share within 50 g of L | make_design |
|---|---|---|---|---|---|
| 1000 | 500-2500 | 2,138,145 | 0.0749 | 0.0178 (38,136) | ok at +-300 |
| 1250 | 500-2500 | 2,138,145 | 0.1188 | 0.0169 (36,238) | ok at +-300 |
| 2000 | 1000-3500 | 17,679,669 | 0.0356 | 0.0084 (147,643) | ok at +-300 |
| 2500 | 1500-3500 | 17,475,922 | 0.101 | 0.0241 (420,879) | ok at +-300 |

## Local linear jump at L (triangular kernel, HC1 se, below minus above)

| L | death h=50 | death h=100 | ventilation h=50 | ventilation h=100 |
|---|---|---|---|---|
| 1000 | -0.0168 (0.0081) | -0.0005 (0.0050) | +0.0345 (0.0114) | +0.0257 (0.0076) |
| 1250 | -0.0126 (0.0062) | -0.0062 (0.0037) | +0.0346 (0.0108) | +0.0128 (0.0067) |
| 2000 | -0.0062 (0.0018) | -0.0037 (0.0013) | +0.0027 (0.0035) | +0.0019 (0.0026) |
| 2500 | -0.0037 (0.0009) | -0.0013 (0.0004) | -0.0022 (0.0016) | +0.0005 (0.0009) |
| 1500 (ref) | -0.0071 (0.0051) | -0.0053 (0.0028) | -0.0205 (0.0097) | -0.0011 (0.0054) |

## Method results (optimum un-mirrored, grams; tau = marginal survival effect at L-300, L-150, L, L+150, L+300)

| L | route | cost | optimum | boundary | window share | tau at the five grid points |
|---|---|---|---|---|---|---|
| 1000 | logit_emp | 0 | 1300.0 | yes (top) | 0.068 | -0.1038 -0.0459 -0.0059 +0.0186 +0.0328 |
| 1000 | lpm | 0 | 1300.0 | yes (top) | 0.068 | -0.1615 -0.0624 +0.0785 +0.2138 +0.3816 |
| 1000 | logit_emp | 0.01 | 700.0 | yes (bottom) | 0.068 | -0.1138 -0.0559 -0.0159 +0.0086 +0.0228 |
| 1000 | lpm | 0.01 | 1300.0 | yes (top) | 0.068 | -0.1715 -0.0724 +0.0685 +0.2038 +0.3716 |
| 1250 | logit_emp | 0 | 1550.0 | yes (top) | 0.194 | +0.0119 +0.0192 +0.0220 +0.0228 +0.0240 |
| 1250 | lpm | 0 | 1550.0 | yes (top) | 0.194 | -0.0563 +0.0165 +0.1174 +0.2142 +0.3373 |
| 1250 | logit_emp | 0.01 | 1550.0 | yes (top) | 0.194 | +0.0019 +0.0092 +0.0120 +0.0128 +0.0140 |
| 1250 | lpm | 0.01 | 1550.0 | yes (top) | 0.194 | -0.0663 +0.0065 +0.1074 +0.2042 +0.3273 |
| 2000 | logit_emp | 0 | 1793.9 | no | 0.012 | +0.0062 -0.0032 -0.0118 -0.0252 -0.0462 |
| 2000 | lpm | 0 | 2300.0 | yes (top) | 0.012 | +0.0001 +0.0019 +0.0030 +0.0057 +0.0091 |
| 2000 | logit_emp | 0.01 | 1700.0 | yes (bottom) | 0.012 | -0.0038 -0.0132 -0.0218 -0.0352 -0.0562 |
| 2000 | lpm | 0.01 | 1700.0 | yes (bottom) | 0.012 | -0.0099 -0.0081 -0.0070 -0.0043 -0.0009 |
| 2500 | logit_emp | 0 | 2377.4 | no | 0.122 | +0.0108 +0.0019 -0.0097 -0.0261 -0.0513 |
| 2500 | lpm | 0 | 2800.0 | yes (top) | 0.122 | -0.0012 -0.0006 +0.0005 +0.0013 +0.0017 |
| 2500 | logit_emp | 0.01 | 2213.5 | no | 0.122 | +0.0008 -0.0081 -0.0197 -0.0361 -0.0613 |
| 2500 | lpm | 0.01 | 2200.0 | yes (bottom) | 0.122 | -0.0112 -0.0106 -0.0095 -0.0087 -0.0083 |
| 1500 (ref) | logit_emp | 0 | 1800.0 | yes (top) | 0.462 | +0.0868 +0.0479 +0.0247 +0.0110 +0.0024 |
| 1500 (ref) | lpm | 0 | 1800.0 | yes (top) | 0.462 | -0.0003 +0.0515 +0.1109 +0.1785 +0.2405 |
| 1500 (ref) | logit_emp | 0.01 | 1663.2 | no | 0.462 | +0.0768 +0.0379 +0.0147 +0.0010 -0.0076 |
| 1500 (ref) | lpm | 0.01 | 1800.0 | yes (top) | 0.462 | -0.0103 +0.0415 +0.1009 +0.1685 +0.2305 |

Run times: 2500 g 25,522 s (logit c=0 8802 s, c=0.01 6335 s, lpm 344 s and 9918 s; heavily loaded machine), 1500 g 511 s, 1000 g 710 s (logit c=0 488 s), 1250 g 282 s, 2000 g 6312 s (logit c=0.01 4676 s, on a loaded machine). No failures; no line needed the narrower +-200 set.

## Plain reading per line

- 1000 g: the method finds no effect at the line (tau = -0.006 at 1000 g, -0.016 net of cost), and it is negative below and positive above; the optimum sits at an edge. The local jump gives death down 1.7 pp at h=50 (about 2 se) but 0.0 at h=100, and ventilation up 3 pp. Weak, unstable evidence; the method does not claim an effect here. The lpm reads +8 pp at the line and +38 pp above it, which is the extrapolation failure seen at 1500 g.
- 1250 g (placebo): not flat. The logit tau is positive (+1 to +2.4 pp) at every grid point, and the optimum is the top edge, 1550 g. The candidate set reaches the real 1500 g line (L+250), so the estimator is picking up the 1500 g effect, not a 1250 g effect. The local jump is -1.3 pp at h=50 (2 se), -0.6 pp at h=100, and ventilation +3 pp at h=50, so 1250 g is not a clean placebo in this file (it is inside the 1000-1500 g care range). A placebo must use a set that excludes 1500 g.
- 2000 g (placebo): the logit tau is near zero at 2000 g (-1.2 pp) and becomes more negative above (-4.6 pp at 2300 g); the interior optimum (1794 g, c=0) is where tau crosses from positive to negative, and at c=0.01 tau is negative everywhere (optimum at the bottom edge). Not a positive effect; no support for treating at 2000 g. The local jump is a small death decline (-0.6 pp, 3 se at h=50; -0.4 pp at h=100) with no ventilation jump. The size is tiny against the method's tau, but it is not exactly zero.
- 2500 g: the logit tau is +1.1 pp at 2200 g, near zero at 2300, and -1.0 pp at 2500 g, falling to -5 pp above. The interior optimum (2377 g at c=0, 2214 g at c=0.01) is where tau crosses zero, so the method finds no effect at the line. The local jump is a small death drop: -0.37 pp (4 se) at h=50 and -0.13 pp (3 se) at h=100, no ventilation jump. Consistent with little at 2500 g, though the tiny death jump is statistically nonzero. (1500 g reference: logit tau +2.5 pp at the line, optimum 1800 g at the top edge at c=0 and 1663 g interior at c=0.01; local jump -0.7 pp (1.4 se) at h=50, -0.5 pp (1.9 se) at h=100, so the logit effect at 1500 g is larger than the local jump supports.)

Caveats: the lpm results are driven by extrapolation and should not be read as effects. Window shares are small at 2000 g (1.2%), so tau there rests on few births. "Placebo" lines 1250 and 2000 are not clean: 1250 g neighbors the 1500 g line within the candidate set, and 2000 g shows a small local death jump.
