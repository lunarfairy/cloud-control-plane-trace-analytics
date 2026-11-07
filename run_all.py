"""Reproduce the study end to end.

    python run_all.py simulate   # 2,000 learners in 16 parallel shards -> sim/main/*.jsonl.gz (~40-70 min)
    python run_all.py analyse    # analyses A1-A8 -> results/numbers.json (~45 min)
    python run_all.py revision   # revision analyses -> results/*.json (dump cohort, loss grid, RQ2-RQ5, variants, RQ4)
    python make_figures2.py       # Figures 1-3 from supplied results

Requires the packages listed in requirements.txt; moto must be importable (pip install
"moto==5.1.4" boto3 hmmlearn).  MOTO_IAM_LOAD_MANAGED_POLICIES=true is set by the runner.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pylib"))
sys.path.insert(0, ROOT)
N, SHARDS, KEEP_DUMPS = 2000, 16, 40


def simulate():
    os.makedirs(os.path.join(ROOT, "sim", "main"), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT, MOTO_IAM_LOAD_MANAGED_POLICIES="true")
    per = N // SHARDS
    procs = []
    for s in range(SHARDS):
        log = open(os.path.join(ROOT, "sim", "main", f"log{s:02d}.txt"), "w")
        procs.append(subprocess.Popen([sys.executable, "-m", "cptrace.runner", str(s * per), str(per),
                                       os.path.join("sim", "main", f"shard{s:02d}.jsonl.gz"), str(KEEP_DUMPS)],
                                      stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
    while any(p.poll() is None for p in procs):
        time.sleep(15)
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]


def load():
    recs = []
    for s in range(SHARDS):
        with gzip.open(os.path.join(ROOT, "sim", "main", f"shard{s:02d}.jsonl.gz"), "rt", encoding="utf8") as f:
            recs += [json.loads(line) for line in f]
    recs.sort(key=lambda r: r["index"])
    return recs


def analyse():
    from cptrace import study as S
    recs = load()
    eps_by = {r["sid"]: S.truth_episodes(r) for r in recs}
    NUM = {"a1": S.a1_fidelity(recs), "a2": S.a2_landscape(recs, eps_by), "a3": S.a3_snapshots(recs, eps_by),
           "a4": S.a4_active_time(recs, eps_by)}
    rows = S.episode_strategy_table(recs, eps_by)
    NUM["a5"] = S.a5_strategy(rows)
    NUM["a1b"] = S.a1b_event_loss(recs)
    NUM["a6"] = S.a6_hmm(recs)
    NUM["a7"] = S.a7_designs(recs, eps_by, rows)
    NUM["a8"] = S.a8_privacy(recs, cohort=700, ks=(1, 2, 3), targets=300)
    NUM["a8u"] = S.a8_utility(recs, eps_by)
    fe = [e for L in eps_by.values() for e in L if e["code"].startswith("F") and not e["censored"]]
    cross = [e for e in fe if e["wall"] > e["active_true"] + 1800]
    NUM["extra"] = {"cross_share": len(cross) / len(fe),
                    "cross_wall": float(np.median([e["wall"] for e in cross]) / 3600),
                    "cross_active": float(np.median([e["active_true"] for e in cross]) / 60)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(NUM, open(os.path.join(ROOT, "results", "numbers.json"), "w"), indent=1, default=str)


def revision():
    """Revision analyses (see reproduce.md for inputs, outputs and expected runtimes)."""
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=ROOT, MOTO_IAM_LOAD_MANAGED_POLICIES="true", OMP_NUM_THREADS="1")
    steps = [[py, "sim_queue4.py"],                                   # dump cohort (400) + sensitivity variants
             [py, "run_dump_fidelity.py"],                            # RQ1 full-state fidelity
             [py, "run_loss_grid.py"], [py, "merge_loss.py"],         # RQ1 perturbation grid
             [py, "run_rev_analyses.py", "results"],                  # RQ1 census/stack, RQ2, RQ3, variants
             [py, "run_hmm_k3.py"],                                   # RQ3 HMM (K = 3, main table)
             [py, "run_h3_power.py"],                                 # H3 planning simulation
             [py, "run_privacy2.py"]]                                 # RQ5
    for cmd in steps:
        print("running", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    subprocess.run([py, "rq4_run.py", "rq4_inputs.npz", "all"], cwd=os.path.join(ROOT, "rq4"), env=env, check=True)
    shutil.copyfile(os.path.join(ROOT, "rq4", "rq4_results.json"), os.path.join(ROOT, "results", "rq4_results.json"))



if __name__ == "__main__":
    {"simulate": simulate, "analyse": analyse, "revision": revision}[sys.argv[1]]()
