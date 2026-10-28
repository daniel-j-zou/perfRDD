"""Apply the dose-schedule estimator (experiments/methods/dose_schedule.py) to real data.

For each dataset: readiness (R^2 of Q on X, eta spread by dose), the transported dose
response psi_hat(q, a) - psi_hat(q, a_ref) by score bin with its spread over the three role
rotations and its support coverage, and the score-direction modifier test.

    python -m experiments.scripts.dose_schedule_apply erate taxi
    python -m experiments.scripts.dose_schedule_apply taxi --x cubic

Results are printed and written to runs/dose_schedule/<name>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.methods import dose_schedule as ds

OUT = Path(__file__).resolve().parents[1] / "runs" / "dose_schedule"


def erate(args):
    from experiments.datasets.erate import adapter as ad
    s = ad.load()
    A = s.extras["discount_c1"].astype(float)
    fn = s.feature_names
    Z = s.X[:, [fn.index(c) for c in ("log_students", "private", "rural")]]
    levels = np.array([20, 25, 40, 60, 70, 80, 90.0])
    prob = ds.DoseProblem(s.Q, s.X, A, s.Y, Z, ds.DoseBasis(50.0, 90.0, levels=levels), a_lo=0.0)
    return dict(prob=prob, cluster=s.extras["ben"], levels=levels, edges=None, n_bins=10,
                a_grid=None, note="C1 discount (%) on the NSLP share; reference 50%; urban/rural schedules")


def taxi(args):
    from experiments.datasets.taxi import adapter as ad
    s = ad.load_haggag_paci()
    Q = s.Q
    A = np.where(Q >= 15, 0.25 * Q, 3.0)          # VTS middle suggestion: $3, or 25% of the fare
    X = s.X
    if args.x == "cubic":
        d = s.X[:, 0]
        X = np.column_stack([s.X, d ** 2, d ** 3])
    Z = s.X[:, [1, 2]]                              # passenger count, hour of day
    prob = ds.DoseProblem(Q, X, A, s.Y, Z, ds.DoseBasis(3.0, 6.25, l=3), a_lo=3.0)
    return dict(prob=prob, cluster=None, levels=None, edges=np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.0]),
                n_bins=None, a_grid=np.array([3.0, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25]),
                note="VTS middle tip suggestion ($) on the fare; reference $3 (fares below $15)")


def taxi2(args):
    """Both vendors: VTS ($3 below $15, else 25% of the fare) and CMT (20% at every fare)."""
    from experiments.datasets.taxi import adapter as ad
    v, c = ad.load_haggag_paci_vendor("VTS"), ad.load_haggag_paci_vendor("CMT")
    Q = np.concatenate([v.Q, c.Q])
    cmt = np.r_[np.zeros(v.n), np.ones(c.n)]
    A = np.where(cmt == 0, np.where(Q >= 15, 0.25 * Q, 3.0), 0.20 * Q)
    X = np.column_stack([np.vstack([v.X, c.X]), cmt])
    prob = ds.DoseProblem(Q, X, A, np.concatenate([v.Y, c.Y]), X[:, [1, 2]],
                          ds.DoseBasis(3.0, 6.25, l=4, a_lo=1.0), a_lo=1.0)
    return dict(prob=prob, cluster=None, levels=None, edges=np.array([5, 8, 11, 14, 17, 20, 25.0]), n_bins=None,
                a_grid=np.array([1.0, 2.0, 3.0, 3.75, 4.5, 5.25, 6.25]),
                note="middle tip suggestion ($), two schedules (VTS, CMT); reference $3")


def ui(args):
    """Upjohn WA + IL claimants: weekly benefit as a share of the cap, on log(score / cap kink)."""
    import pandas as pd
    from experiments.datasets.ui_upjohn import adapter as ad
    s = ad.load_dose()
    fn = s.feature_names
    arms = pd.get_dummies(pd.Series(s.extras["bonus_arm"])).drop(columns=["WA_C"]).to_numpy(float)
    grp = pd.get_dummies(pd.Series(s.extras["group"])).iloc[:, 1:].to_numpy(float)
    X = np.column_stack([s.X, arms, grp])
    Y = s.extras["outcomes"][args.outcome or "weeks"]
    with np.errstate(divide="ignore"):
        lq = np.log(s.Q)
    keep = (lq > np.quantile(lq, 0.01)) & (lq < np.quantile(lq, 0.99)) & np.isfinite(Y)
    Z = s.X[:, [fn.index(c) for c in ("age", "male", "educ")]]
    prob = ds.DoseProblem(lq[keep], X[keep], s.extras["A"][keep], Y[keep], Z[keep],
                          ds.DoseBasis(1.0, 1.0, l=3, a_lo=0.24), a_lo=0.24)
    return dict(prob=prob, cluster=None, levels=None, edges=np.quantile(prob.Q, np.linspace(0, 1, 9)), n_bins=None,
                a_grid=np.array([0.3, 0.45, 0.6, 0.75, 0.9, 1.0]),
                note=f"weekly benefit / cap on log(score / cap kink); Y = {args.outcome or 'weeks'}; "
                     "bonus arms and schedule groups in X")


def lending(args):
    """Lending Club 36-month loans issued 2009-2013 (LoanStats3a, 3b), resolved.

    Q = sub-grade rank (A1 = 1, ..., G5 = 35); A = interest rate (%), set by LC's rate table for
    the sub-grade and issue month (95% of loans at their cell's modal rate); the table changed
    over time, so issue quarters are several schedules (quarter indicators in X).
    Y = 1{charged off}. Reference rate 12%.
    """
    import pandas as pd
    raw = Path(__file__).resolve().parents[1] / "datasets" / "lending_club" / "data" / "raw"
    cols = ["term", "int_rate", "sub_grade", "issue_d", "loan_status", "loan_amnt", "annual_inc", "dti",
            "emp_length", "home_ownership", "purpose", "delinq_2yrs", "inq_last_6mths", "open_acc", "revol_util",
            "total_acc", "earliest_cr_line", "verification_status", "pub_rec", "revol_bal", "total_pymnt",
            "funded_amnt"]
    df = pd.concat([pd.read_csv(raw / f, skiprows=1, usecols=cols, low_memory=False)
                    for f in ("LoanStats3a.csv", "LoanStats3b.csv")], ignore_index=True)
    df = df[df["sub_grade"].notna() & df["int_rate"].notna() & df["term"].astype(str).str.contains("36")]
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["issue"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df[df["issue"] >= "2009-01-01"]
    Q = (df["sub_grade"].str[0].map({c: i for i, c in enumerate("ABCDEFG")}) * 5
         + df["sub_grade"].str[1].astype(int)).to_numpy(float)
    A = df["int_rate"].astype(str).str.rstrip("%").astype(float).to_numpy()
    if args.outcome == "return":       # realized return per dollar over the life of the loan
        Y = (df["total_pymnt"] / df["funded_amnt"] - 1).to_numpy(float)
    else:
        Y = (df["loan_status"] == "Charged Off").to_numpy(float)
    num = pd.DataFrame({
        "log_inc": np.log1p(df["annual_inc"]), "dti": df["dti"], "delinq": df["delinq_2yrs"],
        "inq": df["inq_last_6mths"], "open_acc": df["open_acc"],
        "revol_util": pd.to_numeric(df["revol_util"].astype(str).str.rstrip("%"), errors="coerce"),
        "total_acc": df["total_acc"], "pub_rec": df["pub_rec"], "log_revol": np.log1p(df["revol_bal"]),
        "log_amt": np.log(df["loan_amnt"]),
        "emp": df["emp_length"].astype(str).str.extract(r"(\d+)")[0].astype(float),
        "hist_yrs": (df["issue"] - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")).dt.days / 365.25})
    num = num.fillna(num.median())
    cat = pd.get_dummies(df[["home_ownership", "purpose", "verification_status"]].astype(str), drop_first=True)
    qtr = pd.get_dummies(df["issue"].dt.to_period("Q").astype(str), drop_first=True)
    X = np.column_stack([num.to_numpy(float), cat.to_numpy(float), qtr.to_numpy(float)])
    prob = ds.DoseProblem(Q, X, A, Y, num[["log_inc", "dti"]].to_numpy(float),
                          ds.DoseBasis(12.0, 22.0, l=3, a_lo=6.0), a_lo=6.0)
    return dict(prob=prob, cluster=None, levels=None, edges=np.array([0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 35.5]),
                n_bins=None, a_grid=np.array([7.0, 9.0, 12.0, 15.0, 18.0, 21.0]),
                note="interest rate (%) on the sub-grade rank (bins = grades A..E, F-G); "
                     f"Y = {'realized return' if args.outcome == 'return' else 'charged off'}; ref 12%")


def esrd(args):
    """CMS ESRD QIP: payment cut (%) in steps on TPS minus the year's minimum; Y = TPS two years later."""
    from experiments.datasets.cms_esrd_qip import adapter as ad
    s = ad.load_dose() if not args.outcome else ad.load_dose(outcome=args.outcome)
    e, fn = s.extras, s.feature_names
    levels = np.array([0.5, 1.0, 1.5, 2.0])
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    prob = ds.DoseProblem(s.Q, s.X, e["A"], s.Y, Z, ds.DoseBasis(0.0, 2.0, levels=levels), a_lo=0.0)
    return dict(prob=prob, cluster=e["ccn"], levels=levels, edges=np.array([-60, -30.5, -20.5, -10.5, -0.5, 10, 20, 40.0]),
                n_bins=None, a_grid=None, note=f"payment cut (%) on TPS - minimum TPS; Y = {args.outcome or 'tps_t2'}")


def dsh(args):
    """Medicare DSH adjustment (%) on the DPP, FY 2008-13; Y = log Medicare cases in t+1 (or exit by t+2)."""
    from experiments.datasets.medicare_dsh import adapter as ad
    s = ad.load_dose(outcome=args.outcome or "log_cases")
    e, fn = s.extras, s.feature_names
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    a_bar = float(np.quantile(e["A"], 0.98))
    prob = ds.DoseProblem(s.Q, s.X, e["A"], s.Y, Z, ds.DoseBasis(0.0, a_bar, l=3), a_lo=0.0)
    return dict(prob=prob, cluster=e.get("unit_id"), levels=None, edges=np.array([0, 10, 15, 20, 25, 30, 40, 60.0]),
                n_bins=None, a_grid=np.array([0, 2.5, 5, 8, 12, 16.0]),
                note=f"DSH adjustment (%) on the DPP; Y = {args.outcome or 'log_cases'}")


def ma(args):
    """Medicare Advantage benchmark percentage (steps) on the county FFS rank, 2017-26; Y = change in penetration."""
    from experiments.datasets.medicare_ma_benchmarks import adapter as ad
    s = ad.load_dose(outcome=args.outcome or "dpen")
    e, fn = s.extras, s.feature_names
    A = np.round(e["A"], 4)
    levels = np.array(sorted(set(A) - {0.95}))
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    prob = ds.DoseProblem(s.Q, s.X, A, s.Y, Z, ds.DoseBasis(0.95, 1.15, levels=levels), a_lo=0.9)
    return dict(prob=prob, cluster=e["unit_id"], levels=levels, edges=np.linspace(0, 100, 9), n_bins=None,
                a_grid=None, note=f"MA applicable percentage on the FFS rank; Y = {args.outcome or 'dpen'}")


def lcff(args):
    """California LCFF concentration grant ($ per ADA) on UPP, 2020-21 to 2024-25; Y = next spring's CAASPP
    percent meeting the standard (ELA by default, --outcome math)."""
    from experiments.datasets.ca_lcff import adapter as ad
    s = ad.load_dose(subject=args.outcome or "ela")
    e, fn = s.extras, s.feature_names
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    a_bar = float(np.quantile(e["A"], 0.98))
    prob = ds.DoseProblem(s.Q, s.X, e["A"], s.Y, Z, ds.DoseBasis(0.0, a_bar, l=3), a_lo=0.0)
    return dict(prob=prob, cluster=e["lea_id"], levels=None, edges=np.array([0, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]),
                n_bins=None, a_grid=np.array([0, 250, 500, 1000, 1500, 2000.0]),
                note=f"LCFF concentration grant ($/ADA) on UPP; Y = CAASPP % met, {args.outcome or 'ela'}")


DATASETS = {"erate": erate, "taxi": taxi, "taxi2": taxi2, "ui": ui, "lending": lending, "esrd": esrd, "dsh": dsh,
            "ma": ma, "lcff": lcff}


def run(name, args):
    cfg = DATASETS[name](args)
    prob, cl = cfg["prob"], cfg["cluster"]
    a_ref = prob.basis.a_ref
    rd = ds.readiness(prob.Q, prob.X, prob.A, a_ref, levels=cfg["levels"])
    est = ds.estimate(prob, n_bins=cfg["n_bins"] or 10, seed=args.seed, a_grid=cfg["a_grid"],
                      score_edges=cfg["edges"], cluster=cl)
    _, eta, _ = ds.first_stage(prob.Q, prob.X, np.ones(len(prob.Q), bool))
    cov = ds.coverage(eta, prob.A, prob.Q, est["edges"], est["a_grid"], a_ref, levels=cfg["levels"], band=0.04)
    j0 = int(np.argmin(np.abs(est["a_grid"] - a_ref)))
    eff = est["psi"] - est["psi"][:, [j0]]
    sd = (est["psi_rot"] - est["psi_rot"][:, :, [j0]]).std(0)
    mt = ds.modifier_test(prob, seed=args.seed, cluster=cl)
    print(f"\n== {name}: {cfg['note']}")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rd.items() if k != "by_dose"})
    for row in rd["by_dose"]:
        print("  dose %-16s n=%7d  eta q10/50/90 = %8.2f %8.2f %8.2f" % row)
    print(f"K={est['K']} k={est['k']}; effect vs a_ref={a_ref:g} by score bin: estimate (rotation sd, coverage)")
    print("score bin        " + " ".join("%20s" % f"a={a:g}" for a in est["a_grid"]))
    for i in range(len(est["edges"]) - 1):
        cells = ["%7.3f(%5.3f,%3.0f%%)" % (e, v, 100 * c) for e, v, c in zip(eff[i], sd[i], cov[i])]
        print("%6.1f-%6.1f   " % (est["edges"][i], est["edges"][i + 1]) + " ".join("%20s" % c for c in cells))
    print("score-direction modifier test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in mt.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = name + ("" if args.x == "linear" else f"_{args.x}") + (f"_{args.outcome}" if args.outcome else "")
    json.dump({"readiness": rd, "edges": est["edges"].tolist(), "a_grid": est["a_grid"].tolist(),
               "effect": eff.tolist(), "rotation_sd": sd.tolist(), "coverage": cov.tolist(), "modifier_test": mt,
               "K": est["K"], "k": est["k"], "note": cfg["note"]}, open(OUT / f"{tag}.json", "w"), indent=1, default=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=sorted(DATASETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x", default="linear", choices=["linear", "cubic"], help="taxi: baseline in distance")
    p.add_argument("--outcome", default=None, help="ui: weeks, benefits, spell or earn_post")
    args = p.parse_args()
    for name in args.names:
        run(name, args)


if __name__ == "__main__":
    main()
