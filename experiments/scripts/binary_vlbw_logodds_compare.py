"""VLBW: the effect on the probability scale against the effect on the log-odds scale (2026-11-12).

Both use the same logit fit on both arms (pooled 1995-2002, candidate thresholds 1200-2200 g,
seed 1, four rotations). The probability route (logit_emp) scores W = F(a1) - F(a0) with a
cost c in survival points; the log-odds route (logit_lo) scores W = a1 - a0 = c1(eta) - c0(eta)
+ X_e'beta2, affine in X, with the paper's estimator and a cost in log-odds units.

Reads experiments/runs/binary_outcome_routes_vlbw_wide{,_logodds}_c*/summary.json and
writes binary_vlbw_logodds_compare.png and a table of optima to the folder given:

    python -m experiments.scripts.binary_vlbw_logodds_compare experiments/runs/binary_vlbw_logodds_compare
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
BG = "#fcfcfb"
COLS = ["#2a78d6", "#eb6834", "#2c9a5f", "#b03a48", "#7a5bb5", "#8a8a85"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#888",
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.labelcolor": "#444", "xtick.color": "#555", "ytick.color": "#555",
                     "lines.linewidth": 1.3, "savefig.facecolor": BG})


def runs(prefix, route):
    out = {}
    for f in glob.glob(str(R / f"{prefix}_c*" / "summary.json")):
        c = Path(f).parent.name.split("_c")[-1]
        rows = [r for r in json.load(open(f)) if r["route"] == route]
        if rows:
            out[float(c)] = rows[0]
    return dict(sorted(out.items()))


def crossings(w, tau, c):
    """Weights where the net effect tau - c changes sign, read with the weight increasing:
    'falls' (net effect turns negative as the line moves up: a local optimum) or 'rises'."""
    o = np.argsort(w)
    w, d = np.asarray(w)[o], np.asarray(tau)[o] - c
    out = []
    for i in range(len(w) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and np.sign(d[i]) != np.sign(d[i + 1]):
            x = w[i] + (w[i + 1] - w[i]) * d[i] / (d[i] - d[i + 1])
            out.append((float(x), "falls" if d[i] > d[i + 1] else "rises"))
    return out


prob = runs("binary_outcome_routes_vlbw_wide", "logit_emp")
# the main costs were run on the 1200-1800 g candidate set; same data, folds and fit
for c, r in runs("binary_outcome_routes_vlbw_pooled", "logit_emp").items():
    prob.setdefault(c, r)
prob = dict(sorted(prob.items()))
lo = runs("binary_outcome_routes_vlbw_wide_logodds", "logit_lo")
p0, l0 = prob[0.0], lo[0.0]
w = -np.array(p0["curves"])                       # threshold in grams, treated below
tp, tl = np.array(p0["tau"]), np.array(l0["tau"])
i15 = int(np.argmin(abs(w - 1500)))
kappa = tp[i15] / tl[i15]                         # survival points per log-odds unit at 1500 g

lines = [f"conversion at 1500 g: tau_prob = {tp[i15]:.4f}, tau_logodds = {tl[i15]:.4f}, "
         f"ratio = {kappa:.4f} survival points per log-odds unit", ""]
lines.append("tau on both scales (c = 0)")
lines.append(f"{'grams':>6} {'prob':>8} {'logodds':>8} {'ratio':>8}")
for g in (1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200):
    i = int(np.argmin(abs(w - g)))
    lines.append(f"{w[i]:6.0f} {tp[i]:8.4f} {tl[i]:8.4f} {tp[i] / tl[i]:8.4f}")
lines.append("")
lines.append("optima from explicit runs (argmax of the pooled criterion)")
for name, d in (("probability", prob), ("log-odds", lo)):
    for c, r in d.items():
        lines.append(f"{name:11s} cost {c:<8g} optimum {-r['phi_pooled']:6.0f} g  boundary {r['at_boundary']!s:5s} "
                     f"gain {r['gain_vs_deployed']:+.5f}  crossings of tau_0 = c: "
                     + ", ".join(f"{x:.0f} ({s})" for x, s in crossings(w, (tp if d is prob else tl), c)))


def derived(d, c):
    """Optimum at cost c from runs at 0 and at one cost c1 > 0 of a linear route: the cost only
    enters as -c times the treated share, so S - S(phi0) = (U_0 - U_c1) / c1 and
    U_c = U_0 - c (S - S(phi0)); the grid argmax is refined to the nearest downward crossing."""
    c1 = max(d)
    U0, U1 = np.array(d[0.0]["U"]), np.array(d[c1]["U"])
    Uc = U0 - c * (U0 - U1) / c1
    i = int(np.argmax(Uc))
    down = [x for x, sgn in crossings(w, np.array(d[0.0]["tau"]), c) if sgn == "falls"]
    near = [x for x in down if abs(x - w[i]) <= 1.5 * abs(w[1] - w[0])]
    return (near[0] if near else float(w[i])), Uc


matched = {}
if len(lo) > 1:
    lines.append("")
    lines.append("log-odds optima at costs matched to the probability costs at 1500 g (c_lo = c_prob / ratio)")
    for cp in sorted(c for c in prob if c > 0):
        cl = cp / kappa
        x, Uc = derived(lo, cl)
        matched[cp] = (cl, x, Uc)
        lines.append(f"  prob cost {cp:<7g} -> log-odds cost {cl:.4f}: optimum {x:6.0f} g "
                     f"(probability scale {-prob[cp]['phi_pooled']:.0f} g)")
    c1 = max(lo)
    x1, _ = derived(lo, c1)
    lines.append(f"  check: derived optimum at the run cost {c1:g} is {x1:.0f} g, the run gives "
                 f"{-lo[c1]['phi_pooled']:.0f} g")
text = "\n".join(lines)
print(text)
(OUT / "optima.txt").write_text(text + "\n")

# ---------------------------------------------------------------- figure
SHOW = [c for c in (0.0005, 0.01, 0.02, 0.03) if c in prob]
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4))
# probability scale
ax = axs[0, 0]
ax.plot(w, 100 * tp, color="#222", lw=1.6, label="effect at the threshold")
ax.plot(-p0["phi_pooled"], 0, "o", color="#222", ms=4)
for k, c in enumerate(SHOW):
    ax.axhline(100 * c, color=COLS[k], lw=0.9, ls="--",
               label=f"cost {100 * c:g} pt: optimum {-prob[c]['phi_pooled']:.0f} g")
    ax.plot(-prob[c]["phi_pooled"], 100 * c, "o", color=COLS[k], ms=4)
ax.set_title("Probability scale: W = F(a1) - F(a0)", fontsize=10)
ax.set_ylabel("effect at the threshold, survival points")
ax = axs[1, 0]
for k, c in enumerate([0.0] + SHOW):
    r = prob[c]
    U = 100 * np.array(r["U"])
    col = "#222" if c == 0 else COLS[k - 1]
    ax.plot(-np.array(r["curves"]), U - U.max(), color=col,
            label=f"cost {100 * c:g} pt" + (" (1200-1800 g set)" if r["curves"][0] > -2000 else ""))
    ax.plot(-r["phi_pooled"], 0, "o", color=col, ms=4)
ax.set_ylabel("utility minus its maximum, survival points per infant")
# log-odds scale, costs matched at 1500 g
ax = axs[0, 1]
ax.plot(w, tl, color="#222", lw=1.6, label="effect at the threshold")
ax.plot(-l0["phi_pooled"], 0, "o", color="#222", ms=4)
for k, c in enumerate(SHOW):
    if c in matched:
        cl, x, _ = matched[c]
        ax.axhline(cl, color=COLS[k], lw=0.9, ls="--", label=f"cost {cl:.2f} (= {100 * c:g} pt at 1500 g): optimum {x:.0f} g")
        ax.plot(x, cl, "o", color=COLS[k], ms=4)
ax.set_title("Log-odds scale: W = a1 - a0, costs matched at 1500 g", fontsize=10)
ax.set_ylabel("effect at the threshold, log-odds units")
ax = axs[1, 1]
U = np.array(l0["U"])
ax.plot(w, U - U.max(), color="#222", label="cost 0")
ax.plot(-l0["phi_pooled"], 0, "o", color="#222", ms=4)
for k, c in enumerate(SHOW):
    if c in matched:
        cl, x, Uc = matched[c]
        ax.plot(w, Uc - Uc.max(), color=COLS[k], label=f"cost {cl:.2f}")
        ax.plot(w[int(np.argmax(Uc))], 0, "o", color=COLS[k], ms=4)
ax.set_ylabel("utility minus its maximum, log-odds units per infant")
for ax in axs.flat:
    ax.axvline(1500, color="#999", lw=0.7, ls=":")
    ax.set_xlabel("threshold, grams (treated below)")
    ax.legend(frameon=False, fontsize=7.2)
for ax in axs[0]:
    ax.axhline(0, color="#aaa", lw=0.7)
fig.tight_layout()
fig.savefig(OUT / "binary_vlbw_logodds_compare.png", dpi=140)
print("wrote", OUT / "binary_vlbw_logodds_compare.png")
