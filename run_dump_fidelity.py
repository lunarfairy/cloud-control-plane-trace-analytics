"""RQ1 full-state fidelity on the state-dump cohort (sim/dumps) -> results/fidelity_dump.json.
Also checks that the re-simulated dump learners behave identically to the main corpus."""
import glob
import gzip
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from cptrace import study as ST  # noqa: E402


def load(p):
    out = []
    for fn in sorted(glob.glob(p)):
        with gzip.open(fn, "rt", encoding="utf8") as f:
            out += [json.loads(line) for line in f]
    return out


dumps = load(os.path.join(ROOT, "sim", "dumps", "c*.jsonl.gz"))
base = {r["sid"]: r for r in load(os.path.join(ROOT, "sim", "main", "shard*.jsonl.gz"))}
sig = lambda r: [(e["eventTime"], e["eventName"], e.get("errorCode")) for e in r["events"]]
mism = [r["sid"] for r in dumps if sig(r) != sig(base[r["sid"]])]
print("learners", len(dumps), "behaviour mismatches", len(mism), Counter(r.get("moto_version", "5.2.5/unrecorded") for r in dumps))
fid = ST.a1_fidelity(dumps)
json.dump(fid, open(os.path.join(ROOT, "results", "fidelity_dump.json"), "w"), indent=1, default=float)
print({k: v for k, v in fid.items() if k != "per_code"})
