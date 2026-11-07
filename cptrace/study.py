"""Simulation-study analyses (ADEMP estimands A1-A8)."""
import copy
import json
import math
from collections import Counter, defaultdict

import numpy as np

from . import analytics as A
from . import invariants as inv
from .replay import Replayer

DAY = 86400.0
T_END = A.ts("2026-03-16T23:59:00Z")
DELTAS_MIN = [1, 5, 15, 60, 240, 1440]
DUR_BINS = [(0, 300, "<5 min"), (300, 1800, "5-30 min"), (1800, 7200, "0.5-2 h"),
            (7200, 43200, "2-12 h"), (43200, 1e12, ">12 h")]


def mgmt_writes(rec):
    return [(k, e) for k, e in enumerate(rec["events"])
            if e.get("eventCategory") == "Management" and not e["readOnly"] and not e.get("errorCode")]


def true_sessions(rec):
    """Work sessions from session start to the learner's last action before the next session.

    This removes idle tails and forum-waiting time from the ground-truth active time.
    """
    starts = [A.ts(a) for a, _ in rec["sessions"]]
    ev = sorted(A.ts(e["eventTime"]) for e in rec["events"] if e["eventCategory"] == "Management")
    out = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else 1e18
        inside = [x for x in ev if s - 1 <= x < nxt]
        if inside:
            out.append((s, max(inside)))
    return out


def truth_episodes(rec):
    t, v = A.oracle_timeline(rec)
    eps = A.episodes(t, v, T_END)
    writes = mgmt_writes(rec)
    led = {x["i"]: x for x in rec["ledger"]}
    sess = true_sessions(rec)
    for e in eps:
        e["wall"] = e["t1"] - e["t0"]
        e["active_true"] = A.session_overlap(e["t0"], e["t1"], sess)
        e["label"] = None
        if not e["censored"]:
            k = t.index(e["t1"]) if e["t1"] in t else None
            # the clearing write = first oracle index at t1 where code is no longer violated
            for j in range(len(t)):
                if t[j] == e["t1"] and not v[j].get(e["code"]) and (j == 0 or v[j - 1].get(e["code"])):
                    k = j
                    break
            if k is not None and k < len(writes):
                L = led.get(writes[k][0], {})
                e["label"] = L.get("strategy") or L.get("phase")
    return eps


# ------------------------------------------------------------------- A1
def a1_fidelity(recs):
    agree = tot = 0
    per_code = Counter()
    per_code_tot = Counter()
    full_eq = full_tot = 0
    unknown = set()
    for r in recs:
        t, v = A.oracle_timeline(r)
        t2, v2, rp = A.replay_vectors(r)
        unknown |= rp.unknown_ops
        for a, b in zip(v, v2):
            tot += 1
            agree += int(a == b)
            for c in inv.CODES:
                per_code_tot[c] += 1
                per_code[c] += int(a.get(c) == b.get(c))
        if r.get("dumps"):
            rp = Replayer()
            k = 0
            writes = mgmt_writes(r)
            for (idx, ev) in writes:
                rp.apply(ev)
                snap = r["dumps"][k][1]
                full_tot += 1
                full_eq += int(_canon(rp.st) == _canon(snap))
                k += 1
    return {"vectors": tot, "vector_agreement": agree / tot, "unknown_ops": sorted(unknown),
            "per_code": {c: per_code[c] / per_code_tot[c] for c in inv.CODES},
            "full_states": full_tot, "full_state_agreement": full_eq / full_tot if full_tot else None,
            "n_learners_full": sum(1 for r in recs if r.get("dumps"))}


def _canon(st):
    s = json.loads(json.dumps(st, default=list))
    # implicit main route tables that were never referenced carry no information
    s["rts"] = {k: v for k, v in s["rts"].items() if v["routes"] or v["assoc"] or not v.get("main")}
    for v in s["rts"].values():
        v["routes"] = sorted(map(tuple, v["routes"]))
        v["assoc"] = sorted(v["assoc"])
        v.pop("main", None)
    for g in s["sgs"].values():
        g["ingress"] = sorted(map(tuple, g["ingress"]))
        g.pop("name", None)
    for i in s["igws"].values():
        pass
    s["igws"] = {k: v for k, v in s["igws"].items()}
    for r_ in s["roles"].values():
        r_.pop("trust_ec2", None)
    for lb in s["lbs"].values():
        lb.pop("scheme", None)
    for tg in s["tgs"].values():
        tg.pop("vpc", None)
    for sub in s["subnets"].values():
        sub.pop("cidr", None)
    s.pop("policies", None)
    return s


# ------------------------------------------------------------------- A2
def a2_landscape(recs, eps_by):
    n = len(recs)
    out = {}
    for c in inv.CODES:
        es = [e for sid, L in eps_by.items() for e in L if e["code"] == c]
        learners = len({sid for sid, L in eps_by.items() if any(e["code"] == c for e in L)})
        res = [e for e in es if not e["censored"]]
        out[c] = {"prevalence": learners / n, "episodes": len(es),
                  "median_wall_h": float(np.median([e["wall"] for e in res]) / 3600) if res else None,
                  "median_active_min": float(np.median([e["active_true"] for e in res]) / 60) if res else None,
                  "unresolved_share": (len(es) - len(res)) / len(es) if es else None}
    api_err = [sum(1 for e in r["events"] if e.get("errorCode") and e["eventCategory"] == "Management") for r in recs]
    out["_api_errors"] = {"learners_with_any": float(np.mean([x > 0 for x in api_err])),
                          "mean_per_learner": float(np.mean(api_err)),
                          "codes": Counter(e["errorCode"] for r in recs for e in r["events"]
                                           if e.get("errorCode") and e["eventCategory"] == "Management").most_common(8)}
    icmp = [any(e["eventName"] == "AuthorizeSecurityGroupIngress" and "icmp" in json.dumps(e["requestParameters"])
                for e in r["events"]) for r in recs]
    out["_icmp_marker_share"] = float(np.mean(icmp))
    # data-plane exposure for F6
    f6 = [e for sid, L in eps_by.items() for e in L if e["code"] == "F6"]
    byid = {r["sid"]: r for r in recs}
    seen = 0
    for sid, L in eps_by.items():
        dts = [A.ts(e["eventTime"]) for e in byid[sid]["events"] if e["eventCategory"] == "Data" and e.get("errorCode")]
        for e in L:
            if e["code"] == "F6" and any(e["t0"] <= x <= e["t1"] for x in dts):
                seen += 1
    out["_f6_with_denied_data_event"] = seen / len(f6) if f6 else None
    out["_n_events"] = int(sum(len(r["events"]) for r in recs))
    out["_events_per_learner_median"] = float(np.median([len(r["events"]) for r in recs]))
    out["_read_share"] = float(np.mean([e["readOnly"] for r in recs for e in r["events"] if e["eventCategory"] == "Management"]))
    out["_completed"] = float(np.mean([r["completed"] for r in recs]))
    out["_helped"] = float(np.mean([r["helped"] > 0 for r in recs]))
    return out


# ------------------------------------------------------------------- A3
def a3_snapshots(recs, eps_by):
    res = {}
    rows = []
    for dmin in DELTAS_MIN:
        dsec = dmin * 60.0
        det = Counter()
        tot = Counter()
        bins_det = Counter()
        bins_tot = Counter()
        ratio = []
        count_true = count_hat = 0
        for r in recs:
            t, v = A.oracle_timeline(r)
            if not t:
                continue
            phase = ((int(r["sid"][1:]) * 0.6180339887) % 1.0) * dsec   # deterministic poll phase per learner
            grid = np.arange(t[0] - dsec + phase, T_END + dsec, dsec)
            sv = A.snapshot_vectors(t, v, grid)
            eps = eps_by[r["sid"]]
            m = A.match_snapshot(eps, grid, sv)
            hat = A.episodes(list(grid), sv, T_END)
            count_true += len(eps)
            count_hat += len(hat)
            for e, (ok, a, b) in zip(eps, m):
                grp = e["code"][0]
                tot[grp] += 1
                det[grp] += int(ok)
                for lo, hi, lab in DUR_BINS:
                    if lo <= e["wall"] < hi:
                        bins_tot[lab] += 1
                        bins_det[lab] += int(ok)
                if ok and not e["censored"] and e["wall"] > 0:
                    ratio.append((b - a) / e["wall"])
        res[dmin] = {"recall_F": det["F"] / tot["F"], "recall_S": det["S"] / tot["S"],
                     "recall_all": (det["F"] + det["S"]) / (tot["F"] + tot["S"]),
                     "recall_by_duration": {lab: bins_det[lab] / bins_tot[lab] for _, _, lab in DUR_BINS if bins_tot[lab]},
                     "n_by_duration": {lab: bins_tot[lab] for _, _, lab in DUR_BINS},
                     "duration_ratio_median": float(np.median(ratio)) if ratio else None,
                     "duration_ratio_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))] if ratio else None,
                     "episode_count_ratio": count_hat / count_true}
    return res


# ------------------------------------------------------------------- A4
def a4_active_time(recs, eps_by):
    out = {}
    for ch in ("O1", "O2"):
        err, rel = [], []
        for r in recs:
            ev = [A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management"
                  and (ch == "O1" or not e["readOnly"])]
            for e in eps_by[r["sid"]]:
                if e["censored"] or not e["code"].startswith("F") or e["active_true"] < 60:
                    continue
                est = A.active_time(e["t0"], e["t1"], ev)
                err.append(est - e["active_true"])
                rel.append(est / e["active_true"])
        out[ch] = {"n": len(err), "median_rel": float(np.median(rel)), "mae_min": float(np.mean(np.abs(err)) / 60),
                   "bias_min": float(np.median(err) / 60),
                   "rel_iqr": [float(np.percentile(rel, 25)), float(np.percentile(rel, 75))]}
    walls = [e["wall"] / max(e["active_true"], 1) for L in eps_by.values() for e in L
             if not e["censored"] and e["code"].startswith("F") and e["active_true"] >= 60]
    out["wall_to_active_median"] = float(np.median(walls))
    return out


# ------------------------------------------------------------------- A5
def episode_strategy_table(recs, eps_by):
    rows = []
    for r in recs:
        evt = [(A.ts(e["eventTime"]), e) for e in r["events"]]
        oracle_ids = [(A.ts(t), sig) for t, _, sig in r["oracle"]]
        for e in eps_by[r["sid"]]:
            if e["censored"] or not e["code"].startswith("F") or e["label"] not in ("H", "T", "D", "help"):
                continue
            f1 = A.strategy_features(evt, e["code"], e["t0"], e["t1"], "O1")
            f2 = A.strategy_features(evt, e["code"], e["t0"], e["t1"], "O2")
            # snapshot-observable churn: resource ids replaced inside the episode
            ids_in = [s for tt, s in oracle_ids if e["t0"] <= tt <= e["t1"]]
            churn = 0
            if len(ids_in) >= 2:
                for key in ("lb", "tg", "asg", "lt", "nat", "role", "profile"):
                    churn += int(set(ids_in[0][key]) != set(ids_in[-1][key]) and bool(ids_in[-1][key]))
            rows.append({"sid": r["sid"], "code": e["code"], "label": e["label"],
                         "label3": "H" if e["label"] == "help" else e["label"],
                         "cli": r["latent"]["cli"], "f1": f1, "f2": f2, "churn": churn,
                         "wall": e["wall"], "n_changes": len(ids_in), "t0": e["t0"], "t1": e["t1"]})
    return rows


def _macro_f1(y, yhat, classes=("H", "T", "D")):
    f = []
    rec = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, yhat) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yhat) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yhat) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        q = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * q / (p + q) if p + q else 0.0)
        rec[c] = q
    return float(np.mean(f)), rec


def a5_strategy(rows, seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    y = [r["label3"] for r in rows]
    out = {"n_episodes": len(rows), "label_counts": Counter(r["label"] for r in rows)}
    for ch in ("O1", "O2"):
        yhat = [A.rule_strategy(r["f1" if ch == "O1" else "f2"], ch) for r in rows]
        mf, rec = _macro_f1(y, yhat)
        acc = float(np.mean([a == b for a, b in zip(y, yhat)]))
        out[f"rule_{ch}"] = {"macro_f1": mf, "recall": rec, "accuracy": acc,
                             "confusion": _confusion(y, yhat)}
        for iface in (True, False):
            idx = [i for i, r in enumerate(rows) if r["cli"] == iface]
            mf_i, _ = _macro_f1([y[i] for i in idx], [yhat[i] for i in idx])
            out[f"rule_{ch}"]["macro_f1_" + ("cli" if iface else "console")] = mf_i
    keys = ["n_reads", "n_reads_path", "n_writes", "n_writes_off", "n_deletes", "recreate", "n_errors", "n_fam", "dur", "world_open"]
    feats = {
        "O1": np.array([[r["f1"][k] for k in keys] + [r["cli"] * 1.0] for r in rows], float),
        "O2": np.array([[r["f2"][k] for k in keys if k not in ("n_reads", "n_reads_path")] for r in rows], float),
        "snapshot": np.array([[r["churn"], r["n_changes"], math.log1p(r["wall"])] for r in rows], float),
    }
    groups = [r["sid"] for r in rows]
    for ch, X in feats.items():
        X = np.column_stack([X, np.log1p(np.abs(X))])
        yhat = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr], np.array(y)[tr])
            yhat[te] = m.predict(X[te])
        mf, rec = _macro_f1(y, list(yhat))
        out[f"supervised_{ch}"] = {"macro_f1": mf, "recall": rec,
                                   "accuracy": float(np.mean([a == b for a, b in zip(y, yhat)]))}
    return out


def _confusion(y, yhat, classes=("H", "T", "D")):
    return {a: {b: sum(1 for p, q in zip(y, yhat) if p == a and q == b) for b in classes} for a in classes}


# ------------------------------------------------------------------- A6
def bin_sequences(r, bin_s=300):
    led = {x["i"]: x for x in r["ledger"]}
    t_rep, v_rep, _ = A.replay_vectors(r)
    t_rep = np.asarray(t_rep)
    ev = [(A.ts(e["eventTime"]), k, e) for k, e in enumerate(r["events"]) if e["eventCategory"] == "Management"]
    if not ev:
        return None
    bins = defaultdict(list)
    for t, k, e in ev:
        bins[int(t // bin_s)].append((t, k, e))
    X, Y, keys = [], [], sorted(bins)
    for b in keys:
        items = bins[b]
        nw = sum(1 for _, _, e in items if not e["readOnly"])
        nr = sum(1 for _, _, e in items if e["readOnly"])
        ne = sum(1 for _, _, e in items if e.get("errorCode"))
        nd = sum(1 for _, _, e in items if e["eventName"].startswith("Delete"))
        tend = (b + 1) * bin_s
        j = np.searchsorted(t_rep, tend, side="right") - 1
        nviol = sum(1 for c in inv.FUNCTIONAL if j >= 0 and v_rep[j].get(c))
        X.append([math.log1p(nw), math.log1p(nr), math.log1p(ne), math.log1p(nd), float(nviol)])
        ph = [led.get(k, {}).get("phase") for _, k, _ in items]
        Y.append(int(sum(p == "debug" for p in ph) > len(ph) / 2))
    return np.array(X), np.array(Y)


def a6_hmm(recs, seed=11):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(seed)
    seqs = [s for s in (bin_sequences(r) for r in recs) if s is not None and len(s[1]) >= 3]
    idx = rng.permutation(len(seqs))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    out = {"n_learners": len(seqs), "n_bins": int(sum(len(s[1]) for s in seqs)),
           "stuck_share": float(np.mean(np.concatenate([s[1] for s in seqs])))}
    for name, cols in (("activity_only", [0, 1, 2, 3]), ("activity_plus_state", [0, 1, 2, 3, 4])):
        Xtr = np.concatenate([seqs[i][0][:, cols] for i in tr]); ltr = [len(seqs[i][1]) for i in tr]
        best, bestll = None, -np.inf
        for k in range(4):
            m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=seed + k, min_covar=1e-3)
            m.fit(Xtr, ltr)
            ll = m.score(Xtr, ltr)
            if ll > bestll:
                best, bestll = m, ll
        ytr = np.concatenate([seqs[i][1] for i in tr])
        ptr = best.predict_proba(Xtr, ltr)
        stuck_rate = (ptr * ytr[:, None]).sum(0) / ptr.sum(0)          # P(stuck | state) on training set
        pte, yte = [], []
        for i in te:
            p = best.predict_proba(seqs[i][0][:, cols])
            pte.append(p @ stuck_rate)
            yte.append(seqs[i][1])
        pte, yte = np.concatenate(pte), np.concatenate(yte)
        thr = 0.5
        out[name] = {"auroc": float(roc_auc_score(yte, pte)),
                     "balanced_accuracy": float(balanced_accuracy_score(yte, pte > thr)),
                     "state_stuck_rates": sorted(map(float, stuck_rate))}
    Xte = np.concatenate([seqs[i][0] for i in te]); yte = np.concatenate([seqs[i][1] for i in te])
    rule = (Xte[:, 4] > 0).astype(int)
    out["rule_any_violation"] = {"balanced_accuracy": float(balanced_accuracy_score(yte, rule))}
    Xtr = np.concatenate([seqs[i][0] for i in tr]); ytr = np.concatenate([seqs[i][1] for i in tr])
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    out["supervised_logistic_upper"] = {"auroc": float(roc_auc_score(yte, p)),
                                        "balanced_accuracy": float(balanced_accuracy_score(yte, p > 0.5))}
    return out


# ------------------------------------------------------------------- A7
def early_features(r, eps, rows_by_sid, horizon_days=7.0):
    ev = [e for e in r["events"] if e["eventCategory"] == "Management"]
    t_first = A.ts(ev[0]["eventTime"])
    h = t_first + horizon_days * DAY
    E = [e for e in eps if e["t0"] <= h and e["code"].startswith("F")]
    pers = sum(min(e["t1"], h) - e["t0"] for e in E) / 3600
    rows = [x for x in rows_by_sid.get(r["sid"], []) if x["t1"] <= h]     # only episodes resolved inside the window
    hyp = [A.rule_strategy(x["f1"], "O1") == "H" for x in rows]
    n_err = sum(1 for e in ev if e.get("errorCode") and A.ts(e["eventTime"]) <= h)
    n_r = sum(1 for e in ev if e["readOnly"] and A.ts(e["eventTime"]) <= h)
    n_w = sum(1 for e in ev if not e["readOnly"] and A.ts(e["eventTime"]) <= h)
    churn = sum(x["churn"] > 0 for x in rows)
    o1 = [len(E), math.log1p(pers), float(np.mean(hyp)) if hyp else 0.5, math.log1p(n_err), n_r / max(n_w, 1), churn]
    snap = [len(E), math.log1p(pers), churn]
    return o1, snap


def make_outcomes(recs, eps_by, lam, seed):
    rng = np.random.default_rng(seed)
    th = np.array([r["latent"]["theta"] for r in recs])
    gain = np.array([sum(1 for e in eps_by[r["sid"]] if e["code"].startswith("F") and e["label"] in ("H", "help"))
                     for r in recs], float)
    zg = (gain - gain.mean()) / gain.std()
    G = 0.6 * th + 0.8 * rng.normal(size=len(th))
    P = rng.binomial(30, 1 / (1 + np.exp(-(0.2 + 0.9 * th))))
    C = rng.binomial(30, 1 / (1 + np.exp(-(0.4 + 0.9 * th + lam * zg))))
    return G, P.astype(float), C.astype(float), zg


def _ftest(y, X0, X1):
    n = len(y)
    Z0 = np.column_stack([np.ones(n), X0])
    Z1 = np.column_stack([Z0, X1])
    r0 = y - Z0 @ np.linalg.lstsq(Z0, y, rcond=None)[0]
    r1 = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    q = X1.shape[1]
    df = n - Z1.shape[1]
    F = ((r0 @ r0 - r1 @ r1) / q) / (r1 @ r1 / df)
    from scipy.stats import f as fdist
    return float(fdist.sf(F, q, df)), float(1 - (r1 @ r1) / ((y - y.mean()) @ (y - y.mean()))), \
        float(1 - (r0 @ r0) / ((y - y.mean()) @ (y - y.mean())))


def a7_power(recs, eps_by, rows, lams=(0.0, 0.15, 0.3), Ns=(100, 200, 400, 700), reps=1000, seed=3):
    rows_by = defaultdict(list)
    for x in rows:
        rows_by[x["sid"]].append(x)
    F1, FS = [], []
    for r in recs:
        a, b = early_features(r, eps_by[r["sid"]], rows_by)
        F1.append(a); FS.append(b)
    F1, FS = np.array(F1), np.array(FS)
    out = {"population": len(recs)}
    rng = np.random.default_rng(seed)
    for lam in lams:
        G, P, C, zg = make_outcomes(recs, eps_by, lam, seed=101)
        out[f"lam{lam}"] = {"corr_theta_G": float(np.corrcoef([r["latent"]["theta"] for r in recs], G)[0, 1]),
                            "corr_theta_P": float(np.corrcoef([r["latent"]["theta"] for r in recs], P)[0, 1])}
        for covname, cov in (("prior_grade", G[:, None]), ("pretest", P[:, None]), ("both", np.column_stack([G, P]))):
            for chname, X1 in (("O1", F1), ("snapshot60", FS)):
                pfull, r2f, r2r = _ftest(C, cov, X1)
                key = f"{covname}|{chname}"
                out[f"lam{lam}"][key] = {"dR2_population": r2f - r2r}
                for N in Ns:
                    rej = 0
                    for _ in range(reps):
                        s = rng.choice(len(C), size=N, replace=False)
                        p, _, _ = _ftest(C[s], cov[s], X1[s])
                        rej += p < 0.05
                    out[f"lam{lam}"][key][f"power_N{N}"] = rej / reps
    return out


# ------------------------------------------------------------------- A8
def release_starts(r, variant, rng):
    ev = sorted(A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management")
    starts = np.array([a for a, _ in A.bursts(ev)])
    if variant == "R0":
        return starts
    if variant == "R1":
        return np.floor(starts / 3600) * 3600
    if variant == "R2":
        return np.floor(starts / DAY) * DAY
    if variant == "R3":
        return starts + DAY * int(rng.integers(-14, 15))
    if variant == "R4":
        return starts - starts[0]
    if variant == "R5":
        return np.floor((starts - starts[0]) / DAY) * DAY
    raise ValueError(variant)


def release_starts_from(starts, variant, rng):
    """Release transformation of an array of session starts (shared by revised analyses)."""
    if variant == "R0":
        return starts
    if variant == "R1":
        return np.floor(starts / 3600) * 3600
    if variant == "R2":
        return np.floor(starts / DAY) * DAY
    if variant == "R3":
        return starts + DAY * int(rng.integers(-14, 15))
    if variant == "R4":
        return starts - starts[0]
    if variant == "R5":
        return np.floor((starts - starts[0]) / DAY) * DAY
    raise ValueError(variant)


def _score(obs, rel, variant, tol=1200.0):
    if variant == "R0":
        return sum(np.any(np.abs(rel - o) <= tol) for o in obs)
    if variant == "R1":
        return sum(np.any(np.abs(rel - np.floor(o / 3600) * 3600) <= 3600) for o in obs)
    if variant == "R2":
        return sum(np.any(rel == np.floor(o / DAY) * DAY) for o in obs)
    if variant == "R3":
        best = 0
        for d in range(-14, 15):
            best = max(best, sum(np.any(np.abs(rel - (o + d * DAY)) <= tol) for o in obs))
        return best
    if variant in ("R4", "R5"):
        best = 0
        for a in obs:
            for x in rel:
                if variant == "R4":
                    sh = x - a
                    best = max(best, sum(np.any(np.abs(rel - (o + sh)) <= tol) for o in obs))
                else:
                    sh = x - np.floor(a / DAY) * DAY
                    best = max(best, sum(np.any(rel == np.floor(o / DAY) * DAY + sh) for o in obs))
        return best
    raise ValueError(variant)


BIG = 1.0e8   # offset separating candidates in the flattened, sorted session array


def _index(rels):
    vals = np.concatenate([r + j * BIG for j, r in enumerate(rels)])
    order = np.argsort(vals)
    return vals[order]


def _hit(sorted_vals, q, tol):
    """For query array q (already offset by candidate), is there a value within tol?"""
    pos = np.searchsorted(sorted_vals, q)
    lo = np.clip(pos - 1, 0, len(sorted_vals) - 1)
    hi = np.clip(pos, 0, len(sorted_vals) - 1)
    return (np.abs(sorted_vals[lo] - q) <= tol) | (np.abs(sorted_vals[hi] - q) <= tol)


def scores_all(obs, rels, sv, variant, tol=1200.0):
    n = len(rels)
    cand = np.arange(n) * BIG
    if variant in ("R0", "R1", "R2"):
        if variant == "R1":
            obs_t, t_ = [np.floor(o / 3600) * 3600 for o in obs], 3600.0
        elif variant == "R2":
            obs_t, t_ = [np.floor(o / DAY) * DAY for o in obs], 0.5
        else:
            obs_t, t_ = obs, tol
        return sum(_hit(sv, cand + o, t_).astype(int) for o in obs_t)
    if variant == "R3":
        best = np.zeros(n, int)
        for d in range(-14, 15):
            best = np.maximum(best, sum(_hit(sv, cand + o + d * DAY, tol).astype(int) for o in obs))
        return best
    # R4 / R5: unknown anchor -> align each observation with each released session
    best = np.zeros(n, int)
    lens = np.array([len(r) for r in rels])
    owner = np.repeat(np.arange(n), lens)
    flat = np.concatenate(rels)
    for a in obs:
        if variant == "R4":
            shift = flat - a
            sc = np.zeros(len(flat), int)
            for o in obs:
                sc += _hit(sv, owner * BIG + o + shift, tol).astype(int)
        else:
            shift = flat - np.floor(a / DAY) * DAY
            sc = np.zeros(len(flat), int)
            for o in obs:
                sc += _hit(sv, owner * BIG + np.floor(o / DAY) * DAY + shift, 0.5).astype(int)
        np.maximum.at(best, owner, sc)
    return best


def a8_privacy(recs, cohort=700, ks=(1, 2, 3), targets=300, seed=5):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), size=cohort, replace=False)
    C = [recs[i] for i in pick]
    out = {"cohort": cohort, "targets": targets}
    tgt = rng.choice(cohort, size=targets, replace=False)
    obs_pool = {}
    for ti in tgt:
        sess = [A.ts(a) for a, _ in C[ti]["sessions"]]
        ev = sorted(A.ts(e["eventTime"]) for e in C[ti]["events"] if e["eventCategory"] == "Management")
        act = [s for s in sess if any(s - 60 <= x <= s + 3 * 3600 for x in ev)] or sess
        obs_pool[ti] = (act, ev)
    for variant in ("R0", "R1", "R2", "R3", "R4", "R5"):
        rel = [release_starts(r, variant, rng) for r in C]
        sv = _index(rel)
        out[variant] = {}
        for k in ks:
            succ = []
            for ti in tgt:
                act, ev = obs_pool[ti]
                if len(act) < k:
                    continue
                chosen = rng.choice(len(act), size=k, replace=False)
                obs = [_first_after(ev, act[j]) + rng.normal(0, 600) for j in chosen]
                sc = scores_all(obs, rel, sv, variant)
                ties = np.flatnonzero(sc == sc.max())
                succ.append((1.0 / len(ties)) if ti in ties else 0.0)
            out[variant][f"k{k}"] = float(np.mean(succ))
            out[variant][f"n_k{k}"] = len(succ)
    out["chance"] = 1.0 / cohort
    out["sessions_per_learner_median"] = float(np.median([len(release_starts(r, "R0", rng)) for r in C]))
    return out


def _first_after(ev, s):
    for x in ev:
        if x >= s - 60:
            return x
    return s


# ------------------------------------------------------------------- A1b
def a1b_event_loss(recs, rates=(0.005, 0.01, 0.02, 0.05), seed=17, resync_min=60):
    """Replay with randomly lost write events, with and without periodic snapshot resync."""
    rng = np.random.default_rng(seed)
    sub = [r for r in recs if r.get("dumps")]
    out = {"n_learners": len(sub)}
    for rate in rates:
        agree = {"replay": [0, 0], "replay+resync": [0, 0]}
        ep_rec = {"replay": [0, 0], "replay+resync": [0, 0]}
        for r in sub:
            writes = mgmt_writes(r)
            drop = rng.random(len(writes)) < rate
            t_or, v_or = A.oracle_timeline(r)
            true_eps = A.episodes(t_or, v_or, T_END)
            for mode in ("replay", "replay+resync"):
                rp = Replayer()
                vecs = []
                next_sync = t_or[0] + resync_min * 60 if t_or else 0
                for k, (idx, ev) in enumerate(writes):
                    tk = A.ts(ev["eventTime"])
                    if mode == "replay+resync" and tk >= next_sync and k > 0:
                        rp.st = copy.deepcopy(_lists_to_tuples(r["dumps"][k - 1][1]))
                        while next_sync <= tk:
                            next_sync += resync_min * 60
                    if not drop[k]:
                        rp.apply(ev)
                    vecs.append(inv.evaluate(rp.st))
                agree[mode][0] += sum(a == b for a, b in zip(v_or, vecs))
                agree[mode][1] += len(vecs)
                hat = A.episodes(t_or, vecs, T_END)
                for e in true_eps:
                    ep_rec[mode][1] += 1
                    ep_rec[mode][0] += int(any(h["code"] == e["code"] and h["t0"] < e["t1"] and h["t1"] > e["t0"] for h in hat))
        out[f"rate{rate}"] = {m: {"vector_agreement": agree[m][0] / agree[m][1],
                                  "episode_recall": ep_rec[m][0] / ep_rec[m][1]} for m in agree}
    return out


def _lists_to_tuples(st):
    s = json.loads(json.dumps(st))
    for rt in s["rts"].values():
        rt["routes"] = [tuple(x) for x in rt["routes"]]
    for g in s["sgs"].values():
        g["ingress"] = [tuple(x) for x in g["ingress"]]
    return s


# ------------------------------------------------------------------- A7 (v2)
def _wald_hc1(y, X, Z, test_cols):
    """2SLS (or OLS when Z is X) with HC1 sandwich; Wald test of the coefficients in test_cols."""
    n, k = X.shape
    PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    Xh = PZ @ X
    A_ = np.linalg.pinv(Xh.T @ Xh)
    b = A_ @ Xh.T @ y
    u = y - X @ b
    meat = (Xh * u[:, None] ** 2).T @ Xh
    V = A_ @ meat @ A_ * n / (n - k)
    bb = b[test_cols]
    W = float(bb @ np.linalg.pinv(V[np.ix_(test_cols, test_cols)]) @ bb)
    from scipy.stats import chi2
    return float(chi2.sf(W, len(test_cols)))


def make_outcomes2(recs, eps_by, lam, seed):
    rng = np.random.default_rng(seed)
    th = np.array([r["latent"]["theta"] for r in recs])
    gain = np.array([sum(1 for e in eps_by[r["sid"]] if e["code"].startswith("F") and e["label"] in ("H", "help"))
                     for r in recs], float)
    zg = (gain - gain.mean()) / gain.std()
    G = 0.6 * th + 0.8 * rng.normal(size=len(th))
    lp = 0.2 + 0.9 * th
    P1 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    P2 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    C = rng.binomial(30, 1 / (1 + np.exp(-(0.4 + 0.9 * th + lam * zg)))).astype(float)
    return th, G, P1, P2, C, zg


def a7_designs(recs, eps_by, rows, lams=(0.0, 0.3, 0.5), Ns=(100, 200, 400, 700), reps=1000, seed=3):
    rows_by = defaultdict(list)
    for x in rows:
        rows_by[x["sid"]].append(x)
    F1, FS = [], []
    for r in recs:
        a, b = early_features(r, eps_by[r["sid"]], rows_by, horizon_days=1.0)
        F1.append(a); FS.append(b)
    F1, FS = np.array(F1), np.array(FS)
    out = {"population": len(recs), "features_O1": ["n_functional_episodes", "log_persistence_h", "share_rule_H",
                                                  "log_api_errors", "read_write_ratio", "n_recreate"],
           "features_snapshot": ["n_functional_episodes", "log_persistence_h", "n_recreate"]}
    rng = np.random.default_rng(seed)
    for lam in lams:
        th, G, P1, P2, C, zg = make_outcomes2(recs, eps_by, lam, seed=101)
        P = P1 + P2
        rel = np.corrcoef(P1, P2)[0, 1]
        res = {"corr_theta_G": float(np.corrcoef(th, G)[0, 1]), "corr_theta_P": float(np.corrcoef(th, P)[0, 1]),
               "split_half_r": float(rel), "corr_gain_theta": float(np.corrcoef(zg, th)[0, 1])}
        for chname, X1 in (("O1", F1), ("snapshot60", FS)):
            q = X1.shape[1]
            for design in ("prior_grade_OLS", "pretest_OLS", "pretest_IV"):
                pw = {}
                for N in Ns:
                    rej = 0
                    for _ in range(reps):
                        s = rng.choice(len(C), size=N, replace=False)
                        one = np.ones((N, 1))
                        if design == "prior_grade_OLS":
                            X = np.column_stack([one, G[s], X1[s]]); Z = X
                        elif design == "pretest_OLS":
                            X = np.column_stack([one, P[s], X1[s]]); Z = X
                        else:
                            X = np.column_stack([one, P1[s], X1[s]]); Z = np.column_stack([one, P2[s], X1[s]])
                        p = _wald_hc1(C[s], X, Z, list(range(2, 2 + q)))
                        rej += p < 0.05
                    pw[f"N{N}"] = rej / reps
                res[f"{design}|{chname}"] = pw
        out[f"lam{lam}"] = res
    return out


# ------------------------------------------------------------------- A8 utility
def _time_map(r, variant, rng):
    ev = sorted(A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management")
    bs = A.bursts(ev)
    starts = np.array([b[0] for b in bs])
    shift = DAY * int(rng.integers(-14, 15))
    s0 = starts[0]

    def f(t):
        if variant == "R0":
            return t
        if variant == "R1":
            return math.floor(t / 3600) * 3600
        if variant == "R2":
            return math.floor(t / DAY) * DAY
        if variant == "R3":
            return t + shift
        if variant == "R4":
            return t - s0
        b = int(np.searchsorted(starts, t + 1e-9, side="right") - 1)
        b = max(b, 0)
        return math.floor((starts[b] - s0) / DAY) * DAY + (t - starts[b])
    return f, ev


def a8_utility(recs, eps_by, n=600, seed=9):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), size=n, replace=False)
    out = {}
    for variant in ("R0", "R1", "R2", "R3", "R4", "R5"):
        ea, ew, tod = [], [], []
        for i in pick:
            r = recs[i]
            f, ev = _time_map(r, variant, rng)
            evm = [f(x) for x in ev]
            for e in eps_by[r["sid"]]:
                if e["censored"] or not e["code"].startswith("F"):
                    continue
                a0 = A.active_time(e["t0"], e["t1"], ev)
                a1 = A.active_time(f(e["t0"]), f(e["t1"]), evm)
                ea.append(abs(a1 - a0) / 60)
                ew.append(abs((f(e["t1"]) - f(e["t0"])) - (e["t1"] - e["t0"])) / 3600)
        out[variant] = {"active_mae_min": float(np.mean(ea)), "wall_mae_h": float(np.mean(ew)),
                        "active_exact_share": float(np.mean(np.array(ea) < 1e-6))}
    return out
