import sys, os, json, gzip, glob
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("pylib"))
import numpy as np
import cptrace.study2 as S2
recs=[]
for fn in sorted(glob.glob("sim/main/shard*.jsonl.gz")):
    with gzip.open(fn,"rt",encoding="utf8") as f: recs+=[json.loads(l) for l in f]
out=S2.hmm_analysis(recs, ks=(3,), restarts=6, boot=300)
json.dump(out, open("results/hmm_k3.json","w"), default=float)
print("done")
