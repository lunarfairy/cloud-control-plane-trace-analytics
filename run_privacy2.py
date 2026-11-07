import sys, os, json, gzip, glob
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("pylib"))
import numpy as np
import cptrace.study as ST, cptrace.study2 as S2
recs=[]
for fn in sorted(glob.glob("sim/main/shard*.jsonl.gz")):
    with gzip.open(fn,"rt",encoding="utf8") as f: recs+=[json.loads(l) for l in f]
eps_by={r["sid"]:ST.truth_episodes(r) for r in recs}
out={}
out["utility"]=S2.utility2(recs, eps_by)
json.dump(out, open("results/privacy2.json","w"), default=float)
for name,(coh,sd,tol) in {"base":(700,600.,1200.),"cohort400":(400,600.,1200.),"precise":(700,300.,600.),"coarse":(700,1800.,3600.)}.items():
    out[name]=S2.privacy2(recs, cohort=coh, noise_sd=sd, tol=tol)
    json.dump(out, open("results/privacy2.json","w"), default=float)
    print(name, flush=True)
print("done")
