import sys, json, time
sys.argv = ["x"]
import mc_decomp as M
D = "/Users/zoudj/Documents/Codex/PerfRDD/code/experiments/runs/ds_four_role_coverage"
for n in (3200, 12800):
    stored = {}
    for l in open(f"{D}/nonlinear_n{n}.jsonl"):
        if l.strip():
            r = json.loads(l); 
            if r["rep"] < 3: stored[r["rep"]] = r["est"]
    for rep in range(3):
        t0 = time.time(); o = M.one((n, rep)); dt = time.time() - t0
        print(n, rep, f"{dt:.2f}s", "avg", o["avg"], stored[rep]["avg"], "pooled", o["pooled"], stored[rep]["pooled"], "fixed", o["rot"][0], stored[rep]["fixed"])
        print("   Zhat", [round(z,6) for z in o["Zhat"]], " sum parts", [round(a+b+c+d,6) for a,b,c,d in zip(o["eval"],o["out"],o["dens"],o["end"])])
        print("   lin", [round(x,5) for x in o["lin"]], "rot", [round(x,5) for x in o["rot"]])
