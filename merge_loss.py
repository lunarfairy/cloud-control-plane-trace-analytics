"""Merge results/loss/<mode>_<resync>.json (from loss_worker.py) into results/loss_all.json."""
import glob
import json

LOSS = {}
for f in sorted(glob.glob("results/loss/*.json")):
    d = json.load(open(f))
    LOSS.update({k: v for k, v in d.items() if "|" in k})
    LOSS["_n"], LOSS["_states"], LOSS["_reps"] = d["n_learners"], d["n_states"], d["reps"]
json.dump(LOSS, open("results/loss_all.json", "w"))
print(LOSS["_n"], LOSS["_states"], LOSS["_reps"], len(LOSS) - 3, "conditions")
