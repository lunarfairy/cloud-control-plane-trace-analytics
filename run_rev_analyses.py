"""Revision analyses (RQ1 census and stack check, RQ2 polling/persistence, RQ3 strategies and HMM, sensitivity variants).

Usage:  python run_rev_analyses.py OUTDIR [stage ...]
Stages: census stack rq2 strategy hmm variants (default: all). Writes JSON files into OUTDIR with the same
names as in results/ (new_part1.json, strategy.json, hmm_k2to5.json, variants.json).
Inputs: sim/main/shard*.jsonl.gz (main corpus) and sim/v2/<variant>/c*.jsonl.gz (sensitivity variants).
This script consolidates the analysis cells that were run interactively during the revision; their
outputs in results/ are the values reported in the manuscript (see reproduce.md for the comparison).
"""
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pylib"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np  # noqa: E402

try:  # some sandboxed Windows hosts deny the named pipes that joblib thread pools need
    from joblib import parallel_config
    parallel_config(backend="sequential")
except Exception:
    pass

from cptrace import analytics as A, invariants as INV, study as ST, study2 as S2  # noqa: E402
from cptrace.replay import Replayer  # noqa: E402


def load(pattern):
    out = []
    for fn in sorted(glob.glob(pattern)):
        with gzip.open(fn, "rt", encoding="utf8") as f:
            out += [json.loads(line) for line in f]
    return out


def dump(obj, outdir, name):
    with open(os.path.join(outdir, name), "w") as f:
        json.dump(obj, f, default=float)


def census(recs):
    import botocore.session
    bs = botocore.session.get_session()
    handled = {m[3:] for m in dir(Replayer) if m.startswith("op_")}
    read = re.compile(r"^(Describe|Get|List|Head|Lookup|Search|Select|Check|Validate|Preview|Estimate|Test|Simulate|Generate|Export|Download|Batch(Get|Describe))")
    c = {}
    for svc in ["ec2", "elbv2", "autoscaling", "iam", "s3"]:
        ops = bs.get_service_model(svc).operation_names
        mut = [o for o in ops if not read.match(o)]
        c[svc] = {"operations": len(ops), "mutating": len(mut), "handled": len([o for o in mut if o in handled])}
    used = Counter(e["eventName"] for r in recs for e in r["events"] if e["eventCategory"] == "Management" and not e["readOnly"])
    c["_distinct_write_ops_in_corpus"] = len(used)
    c["_handlers"] = len(handled)
    c["_unhandled_relevant_examples"] = [o for o in ["ModifySecurityGroupRules", "AuthorizeSecurityGroupEgress", "CreateNetworkAclEntry", "ReplaceNetworkAclAssociation",
                                                     "ReplaceRouteTableAssociation", "ModifyVpcAttribute", "AssociateRouteTable", "PutBucketPolicy", "PutPublicAccessBlock",
                                                     "SetDesiredCapacity", "PutRolePermissionsBoundary", "CreateTags", "AttachInstances", "ModifyLoadBalancerAttributes"] if o not in handled]
    return c


def stack_check(recs):
    multi = switch = n_states = multi_states = 0
    for r in recs:
        rp, prev, lm = Replayer(), None, False
        for e in r["events"]:
            if e.get("eventCategory") != "Management" or e.get("readOnly") or e.get("errorCode"):
                continue
            rp.apply(e)
            n_states += 1
            k = INV.locate_stack(rp.st)
            if len(rp.st["lbs"]) > 1 or len(rp.st["asgs"]) > 1:
                multi_states += 1
                lm = True
            if prev and k["lb"] and prev != k["lb"] and prev in rp.st["lbs"]:
                switch += 1
            prev = k["lb"]
        multi += lm
    return {"learners_with_coexisting_candidates": multi, "states_with_coexisting": multi_states, "states": n_states, "switches_while_previous_exists": switch}


def rq2(recs, eps_by):
    out = {"poll": S2.polling_recall(recs, eps_by, boot=300), "snapacc": S2.snapshot_episode_accuracy(recs, eps_by),
           "persistence": S2.persistence(recs, eps_by, boot=300)}
    at = {}
    for gap in (900, 1800, 2700, 3600):
        err = []
        for r in recs:
            ev = [A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management"]
            for e in eps_by[r["sid"]]:
                if e["censored"] or not e["code"].startswith("F") or e["active_true"] < 60:
                    continue
                err.append(A.active_time(e["t0"], e["t1"], ev, gap=gap) - e["active_true"])
        at[gap // 60] = {"mae_min": float(np.mean(np.abs(err)) / 60), "median_bias_min": float(np.median(err) / 60), "n": len(err)}
    out["active_gap"] = at
    return out


def strategy_rows(recs, eps_by):
    rows = ST.episode_strategy_table(recs, eps_by)
    th = {r["sid"]: r["latent"]["theta"] for r in recs}
    for x in rows:
        x["theta"] = th[x["sid"]]
    S2.add_gap_feature(recs, rows)
    S2.gap_before_writes(recs, rows)
    S2.snapshot_strategy_features(recs, rows)
    return rows


def variants(recs, eps_by, rows):
    """Sensitivity variants: transfer of main-corpus models (learners with index >= 300) to each variant."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    def prep(rs):
        eb = {r["sid"]: ST.truth_episodes(r) for r in rs}
        S2.annotate_episodes(rs, eb)
        rw = ST.episode_strategy_table(rs, eb)
        th = {r["sid"]: r["latent"]["theta"] for r in rs}
        for x in rw:
            x["theta"] = th[x["sid"]]
        S2.add_gap_feature(rs, rw)
        S2.gap_before_writes(rs, rw)
        return eb, rw

    feat = lambda rw: S2._features(rw, "O1")
    tr_recs = [r for r in recs if int(r["sid"][1:]) >= 300]
    trS = {r["sid"] for r in tr_recs}
    rows_tr = [x for x in rows if x["sid"] in trS]
    y_tr = np.array([x["label"] for x in rows_tr])
    clf4 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(feat(rows_tr), y_tr)
    y3 = np.array([x["label3"] for x in rows_tr])
    clf3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(feat(rows_tr), y3)
    seq_tr = [s for s in (S2.bin_sequences2(r) for r in tr_recs) if s is not None and len(s["y_debug"]) >= 3]
    Xh = np.concatenate([s["X"] for s in seq_tr])
    lh = [len(s["y_debug"]) for s in seq_tr]
    best = None
    for k in range(6):  # six restarts (seeds 11-16); the best log-likelihood is kept
        m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=11 + k, min_covar=1e-3).fit(Xh, lh)
        ll = m.score(Xh, lh)
        if best is None or ll > best[0]:
            best = (ll, m)
    HM = best[1]
    rankv = np.argsort(np.argsort(HM.means_[:, 4])) / 2

    def analyse(rs, eb, rws):
        out = {"n_learners": len(rs), "n_episodes": sum(len(v) for v in eb.values())}
        E = [e for L in eb.values() for e in L]
        for g in "FS":
            G = [e for e in E if e["code"][0] == g]
            out[f"open_{g}"] = float(np.mean([e["censored"] for e in G]))
            out[f"n_{g}"] = len(G)
        pr = S2.polling_recall(rs, eb, deltas=(15, 60), boot=50)
        for d in ("15", "60"):
            for q in ("recall_F", "pred_F", "recall_S", "pred_S"):
                out[f"{q}_{d}"] = pr[d][q]
        err = []
        for r in rs:
            ev = [A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management"]
            for e in eb[r["sid"]]:
                if e["censored"] or not e["code"].startswith("F") or e["active_true"] < 60:
                    continue
                err.append(A.active_time(e["t0"], e["t1"], ev) - e["active_true"])
        out["active_mae_min"] = float(np.mean(np.abs(err)) / 60)
        out["active_median_bias_min"] = float(np.median(err) / 60)
        X = feat(rws)
        y4 = np.array([x["label"] for x in rws])
        y3_ = np.array([x["label3"] for x in rws])
        out["strategy_n"] = len(rws)
        out["label_counts"] = dict(Counter(y4))
        out["f1_transfer_4"] = S2.macro_f1(list(y4), list(clf4.predict(X)), ("H", "T", "D", "help"))[0]
        out["f1_transfer_3"] = S2.macro_f1(list(y3_), list(clf3.predict(X)), ("H", "T", "D"))[0]
        yh = np.empty(len(rws), dtype=object)
        g = [x["sid"] for x in rws]
        for a, b in GroupKFold(n_splits=5).split(X, y4, g):
            yh[b] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[a], y4[a]).predict(X[b])
        out["f1_within_4"] = S2.macro_f1(list(y4), list(yh), ("H", "T", "D", "help"))[0]
        out["f1_rule_O1_3"] = S2.macro_f1(list(y3_), [A.rule_strategy(x["f1"], "O1") for x in rws], ("H", "T", "D"))[0]
        sq = [s for s in (S2.bin_sequences2(r) for r in rs) if s is not None and len(s["y_debug"]) >= 3]
        p = np.concatenate([S2.forward_filter(HM, s["X"]) @ rankv for s in sq])
        yd = np.concatenate([s["y_debug"] for s in sq])
        yu = np.concatenate([s["y_unprod"] for s in sq])
        out["hmm_auroc_debug"] = float(roc_auc_score(yd, p))
        out["hmm_auroc_unprod_within"] = float(roc_auc_score(yu[yd == 1], p[yd == 1]))
        out["debug_share"] = float(yd.mean())
        return out

    VAR = {}
    for v in ["think2", "idle", "review_hi", "console3", "weaksig", "flatskill"]:
        L = load(os.path.join("sim", "v2", v, "c*.jsonl.gz"))
        n = len(L)
        eb, rw = prep(L)
        VAR[v] = analyse(L, eb, rw)
        if f"base{n}" not in VAR:
            bsub = [r for r in recs if int(r["sid"][1:]) < n]
            VAR[f"base{n}"] = analyse(bsub, {r["sid"]: eps_by[r["sid"]] for r in bsub}, [x for x in rows if int(x["sid"][1:]) < n])
        print(v, "done", flush=True)
    return VAR


def main():
    outdir = sys.argv[1]
    stages = sys.argv[2:] or ["census", "stack", "rq2", "strategy", "hmm", "variants"]
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    recs = load(os.path.join("sim", "main", "shard*.jsonl.gz"))
    eps_by = {r["sid"]: ST.truth_episodes(r) for r in recs}
    S2.annotate_episodes(recs, eps_by)
    print("loaded", len(recs), round(time.time() - t0), flush=True)
    part1 = {}
    if "census" in stages:
        part1["census"] = census(recs)
    if "stack" in stages:
        part1["stack_ambiguity"] = stack_check(recs)
    if "rq2" in stages:
        part1.update(rq2(recs, eps_by))
    if part1:
        dump(part1, outdir, "new_part1.json")
        print("part1", round(time.time() - t0), flush=True)
    rows = None
    if "strategy" in stages or "variants" in stages:
        rows = strategy_rows(recs, eps_by)
    if "strategy" in stages:
        dump(S2.strategy_analysis(rows, boot=300), outdir, "strategy.json")
        print("strategy", round(time.time() - t0), flush=True)
    if "hmm" in stages:
        dump(S2.hmm_analysis(recs, restarts=6, boot=300), outdir, "hmm_k2to5.json")
        print("hmm", round(time.time() - t0), flush=True)
    if "variants" in stages:
        dump(variants(recs, eps_by, rows), outdir, "variants.json")
        print("variants", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
