"""Quick checks for the README: rule agreement, which index binds, R^2 of each score on X,
and the local-linear jump in Y at the cutoff.

    python -m experiments.datasets.tx_index_2013.analysis.checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.datasets.tx_index_2013 import adapter as A


def r2(y: np.ndarray, X: np.ndarray) -> float:
    Z = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta
    return float(1 - e.var() / y.var())


def local_jump(q: np.ndarray, y: np.ndarray, d: np.ndarray, cluster: np.ndarray, h: float,
               X: np.ndarray | None = None) -> dict:
    """Local-linear fit of y on q within |q| < h (triangular kernel), separate slopes on
    each side; returns the coefficient on d (the jump at q = 0, treated minus control)
    with a cluster-robust standard error."""
    k = np.abs(q) < h
    w = (1 - np.abs(q[k]) / h)
    cols = [np.ones(k.sum()), d[k], q[k], d[k] * q[k]]
    if X is not None:
        Xk = X[k]
        Xk = Xk[:, Xk.std(axis=0) > 0]
        cols.append(Xk - Xk.mean(axis=0))
    Z = np.column_stack(cols)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Z * sw[:, None], y[k] * sw, rcond=None)
    e = y[k] - Z @ beta
    bread = np.linalg.pinv((Z * w[:, None]).T @ Z)
    s = pd.DataFrame(Z * (w * e)[:, None]).groupby(cluster[k]).sum().to_numpy()
    V = bread @ (s.T @ s) @ bread
    g = len(np.unique(cluster[k]))
    V *= g / (g - 1)
    return {"h": h, "jump": float(beta[1]), "se": float(np.sqrt(V[1, 1])), "n": int(k.sum()),
            "n_treated": int(d[k].sum())}


def report(s, two, label: str) -> None:
    ex = s.extras
    print(f"\n== {label}: n = {s.n}, IR by rule = {int(s.D.sum())}, "
          f"campuses = {len(set(ex['campus']))}, districts = {len(set(ex['district']))}")
    print("agreement D vs published:", round(float((s.D == ex["eligible_published"]).mean()), 5),
          " misses:", int((s.D != ex["eligible_published"]).sum()),
          " of which granted appeals:",
          int(((s.D != ex["eligible_published"]) & (ex["appeal"] == 1)).sum()))
    print("binding index:", pd.Series(ex["binding"]).value_counts().to_dict())
    for y in sorted(set(ex["year"])):
        m = ex["year"] == y
        M = ex["margins"][m]
        d = s.D[m] == 1
        print(f"  {y}: n={m.sum()}, IR={int(d.sum())}, binding="
              f"{pd.Series(ex['binding'][m]).value_counts().to_dict()}, "
              f"IR missing m1/m2/m3/m4 = {[int((M[d, k] < -A.EPS).sum()) for k in range(4)]}, "
              f"M12<0 {int((ex['m12'][m][d] < -A.EPS).sum())}")
    for pair in A.PAIRS:
        tt = two(pair)
        print(f"  pair {pair}: D agrees with full rule for "
              f"{tt.extras['d_pair_vs_full']:.4f}; with published "
              f"{float((tt.D == ex['eligible_published']).mean()):.4f}; R^2 of Q2 on X "
              f"{r2(tt.Q[:, 1], tt.X):.3f}; corr(Q1, Q2) {np.corrcoef(tt.Q.T)[0, 1]:.3f}")
    M = ex["margins"]

    def r2f(v):
        f = np.isfinite(v)
        return r2(v[f], s.X[f]) if f.any() else np.nan

    print("R^2 on X: q_min %.3f, M12 %.3f, m1 %.3f, m2 %.3f, m3 %.3f, m4 %.3f, Y %.3f" % (
        r2(s.Q, s.X), r2(ex["m12"], s.X), r2f(M[:, 0]), r2f(M[:, 1]), r2f(M[:, 2]),
        r2f(M[:, 3]), r2(s.Y, s.X)))
    q = s.Q
    print("counts of Q in 1-point bins from -3 to 3:",
          [int(((q >= b) & (q < b + 1)).sum()) for b in range(-3, 3)],
          " share |Q| < 2:", round(float((np.abs(q) < 2).mean()), 4))
    print("sd: q_min %.2f M12 %.2f m3 %.2f m4 %.2f; Y mean %.2f sd %.2f; corr(Y, Q) %.3f" % (
        s.Q.std(), ex["m12"].std(), M[:, 2].std(), np.nanstd(M[:, 3]), s.Y.mean(), s.Y.std(),
        np.corrcoef(s.Y, s.Q)[0, 1]))
    cl = ex["campus"]
    D = s.D.astype(float)
    for h in (3.0, 5.0, 8.0):
        print("  jump (no X):", local_jump(s.Q, s.Y, D, cl, h))
        print("  jump (X):   ", local_jump(s.Q, s.Y, D, cl, h, s.X))
    print("  jump, district clusters (X, h=5):",
          local_jump(s.Q, s.Y, D, ex["district"], 5.0, s.X))
    if s.extras.get("y_definition", "").startswith("next rated year's Index 1"):
        gain = s.Y - ex["index1_now"]
        for h in (3.0, 5.0, 8.0):
            print("  jump in Y - current Index 1 (X):", local_jump(s.Q, gain, D, cl, h, s.X))
    print("  continuity of current Index 1 (X, h=5):",
          local_jump(s.Q, ex["index1_now"], D, cl, 5.0, s.X))
    j = list(s.feature_names).index("pct_econ_dis")
    print("  continuity of % econ. disadvantaged (h=5):",
          local_jump(s.Q, s.X[:, j], D, cl, 5.0))


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(A.rule_check().to_string())
    for years in (A.MAIN_YEARS, (2013, 2014, 2015)):
        report(A.load(years), lambda pr, y=years: A.load_two_score(y, pr),
               f"load(years={years})")
    report(A.load_2017(), lambda pr: A.load_2017("two", pr), "load_2017()")


if __name__ == "__main__":
    main()
