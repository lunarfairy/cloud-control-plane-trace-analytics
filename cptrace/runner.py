"""Run simulated learners against moto and store traces + ground truth.

usage: python -m cptrace.runner <first_index> <count> <out.jsonl.gz> [keep_dumps_below]
"""
import datetime as dt
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("pylib"))
os.environ["MOTO_IAM_LOAD_MANAGED_POLICIES"] = "true"
import numpy as np  # noqa: E402
from moto import mock_aws  # noqa: E402

from cptrace import invariants as inv  # noqa: E402
from cptrace import learner as L  # noqa: E402
from cptrace.world import World  # noqa: E402

PARAMS = dict(p_cli=0.35, gap_days=0.8, console_pre_reads=1.2, console_post_reads=1.0,
              p_cidr_conflict=0.12, skill_slope=0.8, attempt_budget=3)
T0 = dt.datetime(2026, 3, 2, 0, 0, 0)            # project release (a Monday), UTC
DEADLINE = T0 + dt.timedelta(days=14, hours=23, minutes=59)
MASTER_SEED = 20261104


def ids_signature(st):
    return {"lb": sorted(st["lbs"]), "tg": sorted(st["tgs"]), "asg": sorted(st["asgs"]),
            "lt": sorted(st["lts"]), "nat": sorted(st["nats"]), "role": sorted(st["roles"]),
            "profile": sorted(st["profiles"])}


def run_one(i, keep_dumps):
    seed = np.random.SeedSequence([MASTER_SEED, i])
    rng = np.random.default_rng(seed)
    sid = f"S{i:04d}"
    with mock_aws():
        acct = "123456789012"
        ip = f"203.0.113.{int(rng.integers(1, 255))}"
        w = World(sid, acct, ip, "pending", T0, rng)
        imgs = w.c["ec2"].describe_images(Owners=["amazon"])["Images"]
        L.AMI = sorted(x["ImageId"] for x in imgs)[0]
        lr = L.Learner(sid, w, rng, PARAMS, DEADLINE)
        w.ua = "aws-cli/2.17.0 Python/3.11" if lr.cli else "AWS Internal"
        for e in w.events:
            e["userAgent"] = w.ua
        t_start = time.time()
        ok = lr.run()
        final = w.snapshot()
        wall = time.time() - t_start
    oracle = [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), inv.evaluate(st, bucket=lr.res.get("bucket")), ids_signature(st))
              for t, st in w.dumps]
    rec = {
        "sid": sid, "index": i,
        "latent": {"theta": lr.theta, "cli": lr.cli, "lab_weekday": lr.lab_weekday, "lab_hour": lr.lab_hour,
                   "pref_hour": lr.pref_hour, "mis": lr.mis, "strat_logits": lr.strat_logits.tolist()},
        "completed": bool(ok), "helped": lr.helped,
        "final_violations": inv.evaluate(final, bucket=lr.res.get("bucket")),
        "events": w.events, "ledger": lr.ledger, "sessions": lr.sessions, "attempts": lr.attempts, "oracle": oracle,
        "dumps": [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), st) for t, st in w.dumps] if i < keep_dumps else None,
        "sim_seconds": wall,
        "moto_version": __import__("moto").__version__,
    }
    return rec


def main():
    first, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    keep = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    if len(sys.argv) > 5:                      # optional JSON overrides of PARAMS (sensitivity variants)
        PARAMS.update(json.loads(sys.argv[5]))
    t = time.time()
    with gzip.open(out + ".part", "wt", encoding="utf8") as f:
        for i in range(first, first + count):
            rec = run_one(i, keep)
            f.write(json.dumps(rec, default=list) + "\n")
            print(i, len(rec["events"]), round(rec["sim_seconds"], 1), flush=True)
    os.replace(out + ".part", out)
    print("done", round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    main()
