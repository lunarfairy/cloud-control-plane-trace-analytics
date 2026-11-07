"""Build the public release (format R6) of the simulated corpus and its ground truth.

Usage: python build_release.py OUTDIR
Writes OUTDIR/corpus_R6.jsonl.gz (one event per line) and OUTDIR/ground_truth.jsonl.gz (one learner per line).
Times are (session_index, t_in_session_s); between-session gap lengths, dates and time of day are not released.
The pseudonymisation key is drawn at random and discarded, so pseudonyms cannot be linked to simulator ids.
"""
import glob
import gzip
import json
import os
import secrets
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from cptrace import analytics as A, release as REL, study2 as S2  # noqa: E402


def session_coords(starts, t):
    b = max(i for i, s in enumerate(starts) if s <= t + 1e-9) if starts and t >= starts[0] - 1e-9 else 0
    return {"session_index": b, "t_in_session_s": round(t - starts[b], 1) if starts else 0.0}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    key = secrets.token_bytes(32)
    n = leaks = 0
    with gzip.open(os.path.join(outdir, "corpus_R6.jsonl.gz"), "wt", encoding="utf8") as fc, \
            gzip.open(os.path.join(outdir, "ground_truth.jsonl.gz"), "wt", encoding="utf8") as fg:
        for fn in sorted(glob.glob(os.path.join(ROOT, "sim", "main", "shard*.jsonl.gz"))):
            with gzip.open(fn, "rt", encoding="utf8") as f:
                for line in f:
                    r = json.loads(line)
                    pid, rows = REL.deidentify(r, key, mode="R6")
                    leaks += len(REL.leak_check(rows, r))
                    for row in rows:
                        fc.write(json.dumps(row) + "\n")
                    times = [A.ts(e["eventTime"]) for e in r["events"]]
                    order = sorted(range(len(times)), key=lambda k: times[k])
                    seq_of = {k: s for s, k in enumerate(order)}
                    starts = [b[0] for b in A.bursts(sorted(times))]
                    lat = {k: v for k, v in r["latent"].items() if k not in ("lab_weekday", "lab_hour", "pref_hour")}
                    gt = {
                        "learner": pid, "latent": lat, "completed": r["completed"], "helped": r["helped"],
                        "final_violations": r["final_violations"],
                        "ledger": sorted(({"seq": seq_of[x["i"]], "phase": x["phase"], "strategy": x["strategy"], "attempt": x["attempt"],
                                           "background_read": x.get("background")} for x in r["ledger"]), key=lambda z: z["seq"]),
                        "attempts": [{"attempt": f"a{j}", "strategy": a["strategy"], "symptom": a["symptom"], "causes": a["causes"],
                                      "start": session_coords(starts, S2.iso_ts(a["t0"])), "end": session_coords(starts, S2.iso_ts(a["t1"]))}
                                     for j, a in enumerate(r["attempts"])],
                        "oracle": [dict(session_coords(starts, A.ts(t)), violations=v) for t, v, _ in r["oracle"]],
                    }
                    fg.write(json.dumps(gt) + "\n")
                    n += 1
    print("learners", n, "leaks", leaks)


if __name__ == "__main__":
    main(sys.argv[1])
