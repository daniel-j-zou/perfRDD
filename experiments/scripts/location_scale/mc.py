"""Task (c): Monte Carlo of the location-scale and additive four-role estimators.

usage: python mc.py <out.jsonl> <dist>:<n>:<reps>:<additive 0/1> [...]
Results are appended one JSON line per replication (resumable / partial-safe).
"""
import common as C
import numpy as np, json, sys, time
from multiprocessing import Pool


def one(task):
    dist, n, rep, do_add = task
    t0 = time.time()
    rng = np.random.default_rng([20261021, {"normal": 0, "t5": 1, "logistic": 2}[dist], n, rep])
    d = C.simulate(n, rng, dist=dist)
    d["folds"] = rng.permutation(n) % 4
    ls, ls_rot = C.four_role(d, scale=True)
    rec = dict(dist=dist, n=n, rep=rep, ls=ls, ls_rot=ls_rot.tolist())
    if do_add:
        ad, ad_rot = C.four_role(d, scale=False)
        rec.update(add=ad, add_rot=ad_rot.tolist())
    rec["secs"] = time.time() - t0
    return rec


if __name__ == "__main__":
    outp = sys.argv[1]
    tasks = []
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        dist, n, reps, add = parts[:4]
        start = int(parts[4]) if len(parts) > 4 else 0
        tasks += [(dist, int(n), r, int(add)) for r in range(start, start + int(reps))]
    # biggest first for better load balance
    tasks.sort(key=lambda t: -t[1])
    t0 = time.time()
    with Pool(2) as pool, open(outp, "a") as f:
        for k, rec in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(tasks) - 1:
                print(f"{k+1}/{len(tasks)} {rec['dist']} n={rec['n']} rep={rec['rep']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)
