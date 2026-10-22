"""Summarise mc.py output against the population targets in target_results.json."""
import common as C
import numpy as np, json, sys
from collections import defaultdict

tgt = json.load(open("target_results.json"))["targets"]
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = defaultdict(list)
for r in recs:
    g[(r["dist"], r["n"])].append(r)
tol = 1e-9


def atb(x):
    x = np.asarray(x)
    return (np.abs(x - C.PHI_LO) < tol) | (np.abs(x - C.PHI_HI) < tol)


print(f"{'dist':8s} {'n':>6s} {'R':>4s} {'phi*':>7s} {'mean':>7s} {'bias':>8s} {'(mcse)':>7s} "
      f"{'sd':>7s} {'rn*sd':>6s} {'rn*bias':>7s} {'RMSE':>7s} {'bnd_rot':>7s} {'bnd_avg':>7s}")
for key in sorted(g, key=lambda k: (k[0] != "normal", k[0], k[1])):
    dist, n = key
    rs = g[key]
    ps = tgt[dist]["phi_star"]
    for est in ["ls", "add"]:
        if est not in rs[0]:
            continue
        x = np.array([r[est] for r in rs])
        rot = np.array([r[est + "_rot"] for r in rs])
        R = len(x)
        bias = x.mean() - ps
        sd = x.std(ddof=1)
        rn = np.sqrt(n)
        print(f"{dist[:8]:8s} {n:6d} {R:4d} {ps:7.4f} {x.mean():7.4f} {bias:8.4f} {sd/np.sqrt(R):7.4f} "
              f"{sd:7.4f} {rn*sd:6.2f} {rn*bias:7.2f} {np.sqrt(np.mean((x-ps)**2)):7.4f} "
              f"{atb(rot).mean():7.3f} {atb(x).mean():7.3f}  [{est}; median {np.median(x):.4f}]")
