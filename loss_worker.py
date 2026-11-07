import sys, os, json, gzip, glob
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("pylib"))
import cptrace.study2 as S2
mode, rs, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rs = None if rs == "none" else int(rs)
recs=[]
for fn in sorted(glob.glob("sim/dumps/c*.jsonl.gz")):
    with gzip.open(fn,"rt",encoding="utf8") as f: recs+=[json.loads(l) for l in f]
out=S2.event_loss2(recs, modes=(mode,), resyncs=(rs,), reps=20, seed=seed)
json.dump(out, open(f"results/loss/{mode}_{sys.argv[2]}.json","w"))
print("done", flush=True)
