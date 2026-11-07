"""Launch the RQ1 event-perturbation grid: 6 mechanisms x 4 re-sync settings, one process each
(seeds 1001-1024 in this order), 20 replications per condition, rates 0.5-5%. Outputs results/loss/*.json;
then run merge_loss.py."""
import os
import subprocess
import sys

os.makedirs("results/loss", exist_ok=True)
procs, k = [], 0
for mode in ("iid", "burst", "create_only", "delete_only", "duplicate", "reorder"):
    for rs in ("none", "15", "60", "240"):
        k += 1
        procs.append(subprocess.Popen([sys.executable, "loss_worker.py", mode, rs, str(1000 + k)],
                                      stdout=open(f"results/loss/{mode}_{rs}.log", "w"), stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
print("done", [p.returncode for p in procs])
