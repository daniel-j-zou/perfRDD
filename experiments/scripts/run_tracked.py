"""Run perfrdd in a chosen first_stage mode across all datasets, with progress.

Prints one line per dataset start and end, including elapsed time. Aggregates
results into a JSON file at the end.

Usage:
    python -m experiments.scripts.run_tracked q_nonlinear
    python -m experiments.scripts.run_tracked all_nonlinear
    python -m experiments.scripts.run_tracked linear
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from experiments._core.registry import list_datasets, load


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["linear", "q_nonlinear", "all_nonlinear"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-n", type=int, default=None,
                   help="cap PLM training subsample (default: perfrdd's DEFAULT_MAX_N=30k)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    import gc
    from experiments.methods.perfrdd import perfrdd, DEFAULT_MAX_N

    names = [n for n in list_datasets() if (args.only is None or n in args.only)]
    max_n = args.max_n if args.max_n is not None else DEFAULT_MAX_N

    print(f"[runner] mode={args.mode}  datasets={len(names)}  "
          f"max_n={max_n}", flush=True)

    out_root = Path(__file__).resolve().parent.parent / "runs" / f"perfrdd_{args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    t_total0 = time.time()
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}  loading", flush=True)
        t_load0 = time.time()
        try:
            sample = load(name)
        except FileNotFoundError as e:
            print(f"[{i}/{len(names)}] {name}  SKIP  (data missing)", flush=True)
            continue
        n_full = len(sample.Y)
        print(f"[{i}/{len(names)}] {name}  start  (n_full={n_full}, load={time.time()-t_load0:.1f}s)",
              flush=True)
        t0 = time.time()
        try:
            res = perfrdd(
                sample, out_root / name,
                first_stage=args.mode, max_n=max_n,
            )
            dt = time.time() - t0
            r2 = res.get("first_stage_R2")
            print(f"[{i}/{len(names)}] {name}  DONE  "
                  f"n_used={res['n_used']}  R²={r2:.3f}  t={dt:.1f}s", flush=True)
            results[name] = res
        except Exception as e:
            dt = time.time() - t0
            print(f"[{i}/{len(names)}] {name}  FAIL  t={dt:.1f}s  err={e!r}", flush=True)
            results[name] = {"error": repr(e)}
        # Free memory before next dataset.
        del sample
        gc.collect()

    print(f"[runner] all done  total={time.time() - t_total0:.1f}s", flush=True)

    out = Path(args.out) if args.out else (out_root / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[runner] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
