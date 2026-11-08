"""Revision analyses (round 2): definitions requested by reviewers.

All functions take the parsed corpus (list of learner records) and the true episodes
(study.truth_episodes) and return JSON-serialisable dictionaries.
"""
import datetime as dt
import math
from collections import Counter, defaultdict

import numpy as np

from . import analytics as A
from . import invariants as inv
from . import study as ST

DAY = 86400.0
T_END = ST.T_END


# ------------------------------------------------------------ helpers
def iso_ts(s):
    """Parse ledger/attempt ISO times (naive, UTC) or CloudTrail 'Z' times."""
    if s.endswith("Z"):
        return A.ts(s)
    return (dt.datetime.fromisoformat(s) - A.EPOCH).total_seconds()     # same epoch as analytics.ts


def poll_grid(r, t, dsec):
    phase = ((int(r["sid"][1:]) * 0.6180339887) % 1.0) * dsec   # deterministic poll phase per learner
    return np.arange(t[0] - dsec + phase, T_END + dsec, dsec)


def viol_segments(t, v, e):
    """Sub-intervals of an episode in which the invariant is evaluated and violated."""
    c, segs, on = e["code"], [], None
    i0 = int(np.searchsorted(np.asarray(t), e["t0"], "left"))
    for j in range(i0, len(t)):
        if t[j] >= e["t1"] and j > i0:
            break
        x = v[j].get(c)
        if x and on is None:
            on = t[j]
        if (not x) and on is not None:
            segs.append((on, t[j]))
            on = None
    if on is not None:
        segs.append((on, e["t1"]))
    return segs


def annotate_episodes(recs, eps_by):
    """Add configuration-violation duration ('cfg') and violated segments to every true episode."""
    for r in recs:
        t, v = A.oracle_timeline(r)
        for e in eps_by[r["sid"]]:
            e["segs"] = viol_segments(t, v, e)
            e["cfg"] = sum(b - a for a, b in e["segs"])


def cluster_boot(values_by_unit, stat, reps=1000, seed=0):
    """Percentile CI of stat(list_of_units) resampling units (learners) with replacement."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_unit)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        out.append(stat([values_by_unit[keys[k]] for k in pick]))
    out = np.asarray(out, float)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


# ------------------------------------------------------------ RQ2 polling
def polling_detection_probability(segments, delta):
    """Length of the union of violated intervals modulo the polling period."""
    projected = []
    for a, b in segments:
        length = b - a
        if length >= delta:
            return 1.0
        if length <= 0:
            continue
        start = a % delta
        end = start + length
        if end <= delta:
            projected.append((start, end))
        else:
            projected.extend(((start, delta), (0.0, end - delta)))
    if not projected:
        return 0.0
    projected.sort()
    left, right = projected[0]
    covered = 0.0
    for a, b in projected[1:]:
        if a > right:
            covered += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return (covered + right - left) / delta


def polling_recall(recs, eps_by, deltas=(1, 5, 15, 60, 240, 1440), boot=400):
    """Detection iff a poll falls inside a violated (applicable) segment; analytic
    uniform-phase prediction from the union of segments modulo Delta."""
    out = {}
    for dmin in deltas:
        dsec = dmin * 60.0
        per = defaultdict(list)          # sid -> list of (group, wall, cfg, hit, pred, censored)
        for r in recs:
            t, _ = A.oracle_timeline(r)
            if not t:
                continue
            g = poll_grid(r, t, dsec)
            for e in eps_by[r["sid"]]:
                hit = any(np.any((g >= a) & (g < b)) for a, b in e["segs"])
                pred = polling_detection_probability(e["segs"], dsec)
                per[r["sid"]].append((e["code"][0], e["wall"], e["cfg"], hit, pred, e["censored"]))
        allr = [x for L in per.values() for x in L]
        res = {}
        for gname in ("F", "S"):
            sel = [x for x in allr if x[0] == gname]
            res[f"recall_{gname}"] = float(np.mean([x[3] for x in sel]))
            res[f"pred_{gname}"] = float(np.mean([x[4] for x in sel]))
            res[f"ci_{gname}"] = cluster_boot(
                per, lambda U, gname=gname: np.mean([x[3] for L in U for x in L if x[0] == gname]), reps=boot, seed=dmin)
        for key, col in (("by_wall", 1), ("by_cfg", 2)):
            d = {}
            for lo, hi, lab in ST.DUR_BINS:
                sel = [x for x in allr if lo <= x[col] < hi]
                d[lab] = {"n": len(sel), "recall": float(np.mean([x[3] for x in sel])) if sel else None,
                          "pred": float(np.mean([x[4] for x in sel])) if sel else None}
            res[key] = d
        out[str(dmin)] = res
    return out


def snapshot_episode_accuracy(recs, eps_by, dmin=60):
    """Detected vs true episode counts and duration error (configuration-violation duration)."""
    dsec = dmin * 60.0
    n_true = n_hat = 0
    ratio, ratio_by = [], defaultdict(list)
    for r in recs:
        t, v = A.oracle_timeline(r)
        if not t:
            continue
        g = poll_grid(r, t, dsec)
        sv = A.snapshot_vectors(t, v, g)
        hat = A.episodes(list(g), sv, T_END)
        n_true += len(eps_by[r["sid"]])
        n_hat += len(hat)
        for e in eps_by[r["sid"]]:
            if e["censored"] or e["cfg"] <= 0:
                continue
            m = [h for h in hat if h["code"] == e["code"] and h["t0"] < e["t1"] and h["t1"] > e["t0"]]
            if not m:
                continue
            d_hat = sum(h["t1"] - h["t0"] for h in m)
            q = d_hat / e["wall"]
            ratio.append(q)
            for lo, hi, lab in ST.DUR_BINS:
                if lo <= e["wall"] < hi:
                    ratio_by[lab].append(q)
    pct = lambda x, p: float(np.percentile(x, p)) if len(x) else None
    return {"count_ratio": n_hat / n_true, "n_true": n_true, "n_hat": n_hat,
            "ratio_median": pct(ratio, 50), "ratio_iqr": [pct(ratio, 25), pct(ratio, 75)],
            "ratio_by_duration": {k: {"n": len(v), "median": pct(v, 50), "iqr": [pct(v, 25), pct(v, 75)]}
                                  for k, v in ratio_by.items()}}


# ------------------------------------------------------------ persistence (censoring-aware)
def km_curve(d, ev):
    d = np.asarray(d, float)
    ev = np.asarray(ev, bool)
    o = np.argsort(d, kind="stable")
    d, ev = d[o], ev[o]
    n, S, tt, ss, i = len(d), 1.0, [], [], 0
    while i < n:
        j = i
        while j < n and d[j] == d[i]:
            j += 1
        k = int(ev[i:j].sum())
        if k:
            S *= 1 - k / (n - i)
        tt.append(d[i]); ss.append(S); i = j
    return np.array(tt), np.array(ss)


def km_median(d, ev):
    tt, ss = km_curve(d, ev)
    k = np.flatnonzero(ss <= 0.5)
    return float(tt[k[0]]) if len(k) else None


def rmst(d, ev, tau):
    tt, ss = km_curve(d, ev)
    tt = np.concatenate([[0.0], tt]); ss = np.concatenate([[1.0], ss])
    m = tt < tau
    t2 = np.append(tt[m], tau)
    return float(np.sum(np.diff(t2) * ss[m]))


def persistence(recs, eps_by, tau_h=72.0, boot=300):
    out = {}
    for c in inv.CODES:
        by = defaultdict(list)
        for sid, L in eps_by.items():
            for e in L:
                if e["code"] == c:
                    by[sid].append(e)
        E = [e for L in by.values() for e in L]
        res = [e for e in E if not e["censored"]]
        w = lambda x: [e["wall"] for e in x]
        km = km_median(w(E), [not e["censored"] for e in E])
        out[c] = {
            "n": len(E), "n_resolved": len(res), "n_censored": len(E) - len(res),
            "open_share": float(np.mean([e["censored"] for e in E])),
            "open_ci": cluster_boot(by, lambda U: np.mean([e["censored"] for L in U for e in L]), reps=boot, seed=1),
            "resolved_median_wall_h": float(np.median(w(res)) / 3600),
            "resolved_iqr_wall_h": [float(np.percentile(w(res), q) / 3600) for q in (25, 75)],
            "resolved_median_active_min": float(np.median([e["active_true"] for e in res]) / 60),
            "resolved_iqr_active_min": [float(np.percentile([e["active_true"] for e in res], q) / 60) for q in (25, 75)],
            "km_median_wall_h": km / 3600 if km is not None else None,
            "rmst_h": rmst(w(E), [not e["censored"] for e in E], tau_h * 3600) / 3600,
            "rmst_ci_h": [x / 3600 for x in cluster_boot(
                by, lambda U: rmst([e["wall"] for L in U for e in L], [not e["censored"] for L in U for e in L],
                                   tau_h * 3600), reps=boot, seed=2)],
            "cfg_share_of_wall": float(np.sum([e["cfg"] for e in E]) / max(np.sum(w(E)), 1e-9)),
            "share_with_absent_component": float(np.mean([e["wall"] - e["cfg"] > 1e-6 for e in E])),
        }
    out["_tau_h"] = tau_h
    return out


# ------------------------------------------------------------ RQ3 strategies
KEYS = ["n_reads", "n_reads_path", "n_writes", "n_writes_off", "n_deletes", "recreate", "n_errors", "n_fam", "dur",
        "world_open"]


def gap_before_burst(evt, t0, t1):
    """Observable: idle time preceding the final work burst of the episode (forum wait signature)."""
    win = [t for t, e in evt if t0 <= t <= t1 and e.get("eventCategory") == "Management"]
    if not win:
        return 0.0
    tail = [win[-1]]
    for t in reversed(win[:-1]):
        if tail[-1] - t > A.GAP:
            return float(tail[-1] - t)
        tail.append(t)
    return 0.0


def add_gap_feature(recs, rows):
    byid = {r["sid"]: r for r in recs}
    cache = {}
    for x in rows:
        sid = x["sid"]
        if sid not in cache:
            cache[sid] = [(A.ts(e["eventTime"]), e) for e in byid[sid]["events"]]
        x["gap_before"] = gap_before_burst(cache[sid], x["t0"], x["t1"])


def macro_f1(y, yhat, classes):
    f, rec, prec = [], {}, {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, yhat) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yhat) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yhat) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        q = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * q / (p + q) if p + q else 0.0)
        rec[c], prec[c] = q, p
    return float(np.mean(f)), rec, prec


def confusion(y, yhat, classes):
    return {a: {b: int(sum(1 for p, q in zip(y, yhat) if p == a and q == b)) for b in list(classes) + ["undetected"]}
            for a in classes}


def _features(rows, ch):
    if ch == "O1":
        X = np.array([[r["f1"][k] for k in KEYS] + [r["cli"] * 1.0, math.log1p(r["gap_before"])] for r in rows], float)
    elif ch == "O2":
        X = np.array([[r["f2"][k] for k in KEYS if k not in ("n_reads", "n_reads_path")] +
                      [math.log1p(r["gap_before_w"])] for r in rows], float)
    else:
        X = np.array([r["snap"] for r in rows], float)
    return np.column_stack([X, np.log1p(np.abs(X))])


def snapshot_strategy_features(recs, rows, dmin=60):
    """Features observable from hourly polls for each episode, using DETECTED episodes only."""
    dsec = dmin * 60.0
    byid = {r["sid"]: r for r in recs}
    by_sid = defaultdict(list)
    for x in rows:
        by_sid[x["sid"]].append(x)
    for sid, L in by_sid.items():
        r = byid[sid]
        t, v = A.oracle_timeline(r)
        sigs = [s for _, _, s in r["oracle"]]
        g = poll_grid(r, t, dsec)
        idx = np.searchsorted(np.asarray(t), g, side="right") - 1
        sv = A.snapshot_vectors(t, v, g)
        hat = A.episodes(list(g), sv, T_END, codes=inv.FUNCTIONAL)
        for x in L:
            m = [h for h in hat if h["code"] == x["code"] and h["t0"] < x["t1"] and h["t1"] > x["t0"] and not h["censored"]]
            if not m:
                x["snap"], x["detected60"] = None, False
                continue
            h = m[0]
            ks = [k for k in range(len(g)) if h["t0"] <= g[k] <= h["t1"] and idx[k] >= 0]
            ch = nchg = 0
            if len(ks) >= 2:
                s0, s1 = sigs[idx[ks[0]]], sigs[idx[ks[-1]]]
                ch = sum(int(set(s0[k]) != set(s1[k]) and bool(s1[k])) for k in s0)
                nchg = sum(int(sigs[idx[ks[i]]] != sigs[idx[ks[i - 1]]]) for i in range(1, len(ks)))
            x["snap"] = [ch, nchg, math.log1p(h["t1"] - h["t0"]), len(ks)]
            x["detected60"] = True


def gap_before_writes(recs, rows):
    byid = {r["sid"]: r for r in recs}
    cache = {}
    for x in rows:
        sid = x["sid"]
        if sid not in cache:
            cache[sid] = [(A.ts(e["eventTime"]), e) for e in byid[sid]["events"] if not e["readOnly"]]
        x["gap_before_w"] = gap_before_burst(cache[sid], x["t0"], x["t1"])


def strategy_analysis(rows, boot=500, seed=7):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {"n_episodes": len(rows), "label_counts": dict(Counter(r["label"] for r in rows))}
    groups = np.array([r["sid"] for r in rows])
    schemes = {
        "merged3": (np.array([r["label3"] for r in rows]), ("H", "T", "D"), np.ones(len(rows), bool)),
        "four": (np.array([r["label"] for r in rows]), ("H", "T", "D", "help"), np.ones(len(rows), bool)),
        "unassisted3": (np.array([r["label"] for r in rows]), ("H", "T", "D"),
                        np.array([r["label"] != "help" for r in rows])),
    }

    def summarise(y, yhat, classes, mask, sids):
        mf, rec, prec = macro_f1(list(y[mask]), list(yhat[mask]), classes)
        by = defaultdict(list)
        for a, b, s in zip(y[mask], yhat[mask], sids[mask]):
            by[s].append((a, b))
        ci = cluster_boot(by, lambda U: macro_f1([a for L in U for a, _ in L], [b for L in U for _, b in L],
                                                 classes)[0], reps=boot, seed=seed)
        return {"macro_f1": mf, "macro_f1_ci": ci, "recall": rec, "precision": prec,
                "accuracy": float(np.mean(y[mask] == yhat[mask])), "n": int(mask.sum()),
                "confusion": confusion(list(y[mask]), list(yhat[mask]), classes)}

    for sname, (y, classes, mask) in schemes.items():
        res = {}
        # a-priori rules (3 classes only; 'help' never predicted)
        if sname != "four":
            for ch in ("O1", "O2"):
                yhat = np.array([A.rule_strategy(r["f1" if ch == "O1" else "f2"], ch) for r in rows], dtype=object)
                res[f"rule_{ch}"] = summarise(y, yhat, classes, mask, groups)
        for ch in ("O1", "O2", "O3"):
            sub = mask & (np.array([r.get("snap") is not None for r in rows]) if ch == "O3" else True)
            idx = np.flatnonzero(sub)
            X = _features([rows[i] for i in idx], ch)
            yy, gg = y[idx], groups[idx]
            yhat_sub = np.empty(len(idx), dtype=object)
            yhat_dum = np.empty(len(idx), dtype=object)
            for tr, te in GroupKFold(n_splits=5).split(X, yy, gg):
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(X[tr], yy[tr])
                yhat_sub[te] = m.predict(X[te])
                yhat_dum[te] = DummyClassifier(strategy="most_frequent").fit(X[tr], yy[tr]).predict(X[te])
            full = np.array(["undetected"] * len(rows), dtype=object)
            full[idx] = yhat_sub
            if ch == "O3":
                res["clf_O3_conditional"] = summarise(yy, yhat_sub, classes, np.ones(len(idx), bool), gg)
                res["clf_O3_end_to_end"] = summarise(y, full, classes, mask, groups)
                res["O3_detected_share"] = float(len(idx) / mask.sum())
            else:
                res[f"clf_{ch}"] = summarise(y, full, classes, mask, groups)
                if ch == "O1":
                    res["majority_baseline"] = summarise(yy, yhat_dum, classes, np.ones(len(idx), bool), gg)
                    # stratified performance
                    strat = {}
                    cli = np.array([rows[i]["cli"] for i in idx])
                    th = np.array([rows[i]["theta"] for i in idx])
                    terc = np.digitize(th, np.quantile(th, [1 / 3, 2 / 3]))
                    for nm, msk in (("cli", cli), ("console", ~cli), ("skill_low", terc == 0),
                                    ("skill_mid", terc == 1), ("skill_high", terc == 2)):
                        strat[nm] = {"macro_f1": macro_f1(list(yy[msk]), list(yhat_sub[msk]), classes)[0],
                                     "n": int(msk.sum())}
                    res["clf_O1_strata"] = strat
        out[sname] = res
    out["features"] = {"O1": KEYS + ["cli", "log_gap_before_final_burst"],
                       "O2": [k for k in KEYS if k not in ("n_reads", "n_reads_path")] + ["log_gap_before_final_burst"],
                       "O3": ["n_component_groups_replaced", "n_polls_with_id_change", "log_detected_duration", "n_polls"],
                       "transform": "each feature x and log1p(|x|); standardised; L2 logistic C=1; 5-fold GroupKFold by learner"}
    return out


# ------------------------------------------------------------ RQ3 debugging-phase bins / HMM
def attempt_success(r):
    """Attempt id -> True if every causal invariant of the symptom was cleared at attempt end."""
    t, v = A.oracle_timeline(r)
    tt = np.asarray(t)
    ok = {}
    for a in r["attempts"]:
        j = int(np.searchsorted(tt, iso_ts(a["t1"]) + 1e-6, side="right") - 1)
        vec = v[j] if j >= 0 else {}
        ok[a["attempt"]] = all(not vec.get(c) for c in a["causes"]) if a["causes"] else True
    return ok


def bin_sequences2(r, bin_s=300):
    """Bins of 5 min containing >=1 management event. Features as in study.bin_sequences plus labels:
    y_debug  = majority of the bin's events carry ledger phase 'debug' (the original target);
    y_unprod = y_debug and the majority attempt of the bin did not clear its causal invariants."""
    led = {x["i"]: x for x in r["ledger"]}
    succ = attempt_success(r)
    t_rep, v_rep, _ = A.replay_vectors(r)
    t_rep = np.asarray(t_rep)
    ev = [(A.ts(e["eventTime"]), k, e) for k, e in enumerate(r["events"]) if e["eventCategory"] == "Management"]
    if not ev:
        return None
    bins = defaultdict(list)
    for t, k, e in ev:
        bins[int(t // bin_s)].append((t, k, e))
    X, Yd, Yu, T = [], [], [], []
    for b in sorted(bins):
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
        dbg = int(sum(p == "debug" for p in ph) > len(ph) / 2)
        att = Counter(led.get(k, {}).get("attempt") for _, k, _ in items if led.get(k, {}).get("attempt"))
        un = 0
        if dbg and att:
            a = att.most_common(1)[0][0]
            un = int(not succ.get(a, True))
        Yd.append(dbg); Yu.append(un); T.append(b * bin_s)
    return {"sid": r["sid"], "X": np.array(X), "y_debug": np.array(Yd), "y_unprod": np.array(Yu), "t": np.array(T),
            "cli": bool(r["latent"]["cli"]), "theta": float(r["latent"]["theta"])}


def forward_filter(model, X):
    """Filtered state posteriors P(z_t | x_1..t) for a fitted hmmlearn GaussianHMM (causal, online)."""
    from scipy.special import logsumexp
    logB = model._compute_log_likelihood(X)
    logA = np.log(np.maximum(model.transmat_, 1e-300))
    la = np.log(np.maximum(model.startprob_, 1e-300)) + logB[0]
    out = np.empty_like(logB)
    out[0] = la - logsumexp(la)
    for t in range(1, len(X)):
        la = logsumexp(out[t - 1][:, None] + logA, axis=0) + logB[t]
        out[t] = la - logsumexp(la)
    return np.exp(out)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def hmm_analysis(recs, seed=11, ks=(2, 3, 4, 5), restarts=10, boot=500):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(seed)
    seqs = [s for s in (bin_sequences2(r) for r in recs) if s is not None and len(s["y_debug"]) >= 3]
    idx = rng.permutation(len(seqs))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    out = {"n_learners": len(seqs), "n_bins": int(sum(len(s["y_debug"]) for s in seqs)),
           "debug_share": float(np.mean(np.concatenate([s["y_debug"] for s in seqs]))),
           "unprod_share": float(np.mean(np.concatenate([s["y_unprod"] for s in seqs]))),
           "unprod_share_within_debug": float(np.mean(np.concatenate([s["y_unprod"][s["y_debug"] == 1] for s in seqs]))),
           "n_train_learners": len(tr), "n_test_learners": len(te)}
    yte_d = np.concatenate([seqs[i]["y_debug"] for i in te])
    yte_u = np.concatenate([seqs[i]["y_unprod"] for i in te])
    sid_te = np.concatenate([[i] * len(seqs[i]["y_debug"]) for i in te])
    scores = {}

    def lagged(X):
        L1 = np.vstack([np.zeros((1, X.shape[1])), X[:-1]])
        L2 = np.vstack([np.zeros((2, X.shape[1])), X[:-2]])[: len(X)]
        return np.column_stack([X, L1, L2])

    for name, cols in (("activity_only", [0, 1, 2, 3]), ("activity_plus_state", [0, 1, 2, 3, 4])):
        Xtr = np.concatenate([seqs[i]["X"][:, cols] for i in tr])
        ltr = [len(seqs[i]["y_debug"]) for i in tr]
        sel = {}
        best_by_k = {}
        for K in ks:
            fits = []
            for k in range(restarts):
                m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, random_state=seed + k, min_covar=1e-3)
                m.fit(Xtr, ltr)
                fits.append((m.score(Xtr, ltr), m))
            fits.sort(key=lambda z: -z[0])
            ll, m = fits[0]
            npar = K - 1 + K * (K - 1) + 2 * K * len(cols)
            bic = -2 * ll + npar * math.log(len(Xtr))
            sel[str(K)] = {"loglik": float(ll), "bic": float(bic),
                           "loglik_spread_restarts": float(fits[0][0] - fits[-1][0])}
            best_by_k[K] = m
        out[name + "_model_selection"] = sel
        for K in ks:
            m = best_by_k[K]
            ytr = np.concatenate([seqs[i]["y_debug"] for i in tr])
            ptr_s = m.predict_proba(Xtr, ltr)
            rate = (ptr_s * ytr[:, None]).sum(0) / ptr_s.sum(0)            # label-calibrated mapping (training labels)
            ptr_f = np.concatenate([forward_filter(m, seqs[i]["X"][:, cols]) for i in tr])
            rate_f = (ptr_f * ytr[:, None]).sum(0) / ptr_f.sum(0)
            # label-free orientation: state ordering by mean emission (violations if available, else errors+writes)
            mu = m.means_
            key = mu[:, 4] if len(cols) == 5 else mu[:, 2] + mu[:, 0]
            rank = np.argsort(np.argsort(key)) / (K - 1)
            sm = [m.predict_proba(seqs[i]["X"][:, cols]) for i in te]
            ff = [forward_filter(m, seqs[i]["X"][:, cols]) for i in te]
            sc = {"smooth_calibrated": np.concatenate([p @ rate for p in sm]),
                  "filter_calibrated": np.concatenate([p @ rate_f for p in ff]),
                  "smooth_labelfree": np.concatenate([p @ rank for p in sm]),
                  "filter_labelfree": np.concatenate([p @ rank for p in ff])}
            res = {}
            for sn, p in sc.items():
                res[sn] = {"auroc_debug": _auc(yte_d, p), "auroc_unprod": _auc(yte_u, p),
                           "auroc_unprod_within_debug": _auc(yte_u[yte_d == 1], p[yte_d == 1])}
            # threshold for operating metrics chosen on TRAINING learners (filtered, calibrated): max balanced accuracy
            pf_tr = ptr_f @ rate_f
            grid = np.unique(np.round(pf_tr, 4))
            ba = [balanced_accuracy_score(ytr, pf_tr >= g) for g in grid]
            thr = float(grid[int(np.argmax(ba))])
            res["filter_calibrated"]["threshold_train"] = thr
            res["filter_calibrated"]["balanced_accuracy_test"] = float(balanced_accuracy_score(yte_d, sc["filter_calibrated"] >= thr))
            res["smooth_calibrated"]["balanced_accuracy_test_at_0.5"] = float(
                balanced_accuracy_score(yte_d, sc["smooth_calibrated"] > 0.5))
            res["state_rates_calibrated"] = sorted(map(float, rate))
            out[f"{name}_K{K}"] = res
            if K == 3:
                scores[name] = sc
                scores[name + "_thr"] = thr
    # baselines (training learners only for fitting)
    Xtr5 = np.concatenate([seqs[i]["X"] for i in tr]); ytr = np.concatenate([seqs[i]["y_debug"] for i in tr])
    Xte5 = np.concatenate([seqs[i]["X"] for i in te])
    XtrL = np.concatenate([lagged(seqs[i]["X"]) for i in tr]); XteL = np.concatenate([lagged(seqs[i]["X"]) for i in te])
    base = {}
    base["rule_any_violation"] = {"balanced_accuracy": float(balanced_accuracy_score(yte_d, Xte5[:, 4] > 0)),
                                  "auroc_debug": _auc(yte_d, Xte5[:, 4]), "auroc_unprod_within_debug":
                                      _auc(yte_u[yte_d == 1], Xte5[yte_d == 1, 4])}
    lr = LogisticRegression(max_iter=3000).fit(Xtr5, ytr)
    p_lr = lr.predict_proba(Xte5)[:, 1]
    lrL = LogisticRegression(max_iter=3000).fit(XtrL, ytr)
    p_lrL = lrL.predict_proba(XteL)[:, 1]
    gb = HistGradientBoostingClassifier(random_state=seed).fit(XtrL, ytr)
    p_gb = gb.predict_proba(XteL)[:, 1]
    lrS = LogisticRegression(max_iter=3000).fit(Xtr5[:, [4]], ytr)
    p_S = lrS.predict_proba(Xte5[:, [4]])[:, 1]
    for nm, p in (("logistic_bin", p_lr), ("logistic_lagged", p_lrL), ("gbm_lagged", p_gb), ("state_only_logistic", p_S)):
        base[nm] = {"auroc_debug": _auc(yte_d, p), "balanced_accuracy": float(balanced_accuracy_score(yte_d, p > 0.5)),
                    "auroc_unprod": _auc(yte_u, p), "auroc_unprod_within_debug": _auc(yte_u[yte_d == 1], p[yte_d == 1])}
        scores["base_" + nm] = p
    out["baselines"] = base
    # learner-clustered CIs and paired differences (test learners resampled)
    units = defaultdict(list)
    for j, s in enumerate(sid_te):
        units[s].append(j)
    keys = list(units)
    rng2 = np.random.default_rng(seed + 99)
    cmp = {"hmm_state_smooth": scores["activity_plus_state"]["smooth_calibrated"],
           "hmm_state_filter": scores["activity_plus_state"]["filter_calibrated"],
           "hmm_state_filter_labelfree": scores["activity_plus_state"]["filter_labelfree"],
           "hmm_activity_filter": scores["activity_only"]["filter_calibrated"],
           "logistic_bin": scores["base_logistic_bin"], "gbm_lagged": scores["base_gbm_lagged"]}
    boots = {k: [] for k in cmp}
    diffs = {"filter_minus_smooth": [], "hmm_filter_minus_logistic": [], "state_minus_activity_filter": []}
    for _ in range(boot):
        pick = np.concatenate([units[keys[k]] for k in rng2.integers(0, len(keys), len(keys))])
        a = {k: _auc(yte_d[pick], v[pick]) for k, v in cmp.items()}
        for k in cmp:
            boots[k].append(a[k])
        diffs["filter_minus_smooth"].append(a["hmm_state_filter"] - a["hmm_state_smooth"])
        diffs["hmm_filter_minus_logistic"].append(a["hmm_state_filter"] - a["logistic_bin"])
        diffs["state_minus_activity_filter"].append(a["hmm_state_filter"] - a["hmm_activity_filter"])
    out["auroc_debug_point"] = {k: _auc(yte_d, v) for k, v in cmp.items()}
    out["auroc_debug_ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    out["auroc_debug_diff_ci"] = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                  for k, v in diffs.items()}
    # operating characteristics of the online (filtered, calibrated, activity+state) detector per test learner
    p = scores["activity_plus_state"]["filter_calibrated"]
    thr = scores["activity_plus_state_thr"]
    alerts = p >= thr
    prec = float(np.mean(yte_d[alerts])) if alerts.any() else None
    sens = float(np.mean(alerts[yte_d == 1]))
    fa_per_hour = []
    delays = []
    pos = 0
    for k in keys:
        j = np.array(units[k])
        y_, a_ = yte_d[j], alerts[j]
        tt = seqs[k]["t"]
        fa_per_hour.append(float(np.sum(a_ & (y_ == 0)) / (len(j) * 300 / 3600)))
        # runs of consecutive debug bins (adjacent in active-bin order)
        i = 0
        while i < len(y_):
            if y_[i] == 1:
                s = i
                while i < len(y_) and y_[i] == 1:
                    i += 1
                pos += 1
                hit = np.flatnonzero(a_[s:i])
                delays.append(float((tt[s + hit[0]] - tt[s]) / 60) if len(hit) else None)
            else:
                i += 1
    det = [d for d in delays if d is not None]
    out["online_operating"] = {"threshold": thr, "precision": prec, "sensitivity": sens,
                               "false_alerts_per_active_hour_median": float(np.median(fa_per_hour)),
                               "false_alerts_per_active_hour_p90": float(np.percentile(fa_per_hour, 90)),
                               "debug_runs": pos, "runs_detected_share": len(det) / max(pos, 1),
                               "delay_min_median": float(np.median(det)) if det else None,
                               "delay_min_p90": float(np.percentile(det, 90)) if det else None}
    # stratified AUROC (online detector)
    cli = np.array([seqs[i]["cli"] for i in sid_te]); th = np.array([seqs[i]["theta"] for i in sid_te])
    terc = np.digitize(th, np.quantile(th, [1 / 3, 2 / 3]))
    out["online_strata_auroc"] = {nm: _auc(yte_d[m], p[m]) for nm, m in
                                  (("cli", cli), ("console", ~cli), ("skill_low", terc == 0), ("skill_mid", terc == 1),
                                   ("skill_high", terc == 2))}
    return out


# ------------------------------------------------------------ RQ5 privacy (revised)
def release_starts2(r, variant, rng, gap=None):
    ev = sorted(A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management")
    starts = np.array([a for a, _ in A.bursts(ev, gap or A.GAP)])
    if variant == "R6":                       # session-ordinal re-basing: session b placed on day b
        return np.arange(len(starts)) * DAY
    return ST.release_starts_from(starts, variant, rng)


def _scores(obs, rels, sv, variant, tol):
    if variant == "R6":
        variant = "R5"
    return ST.scores_all(obs, rels, sv, variant, tol=tol)


def privacy2(recs, cohort=700, variants=("R0", "R1", "R2", "R3", "R4", "R5", "R6"), ks=(1, 2, 3),
             noise_sd=600.0, tol=1200.0, seed=5, boot=1000):
    """All cohort members are targets; results for all eligible targets per k and for the common set
    eligible at k=3. Success = 1/#ties if the true trace is among the best-scoring candidates."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), size=cohort, replace=False)
    C = [recs[i] for i in pick]
    obs_pool = []
    for r in C:
        sess = [A.ts(a) for a, _ in r["sessions"]]
        ev = sorted(A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management")
        act = [s for s in sess if any(s - 60 <= x <= s + 3 * 3600 for x in ev)] or sess
        obs_pool.append((act, ev))
    draws = {}
    for ti in range(cohort):
        act, ev = obs_pool[ti]
        for k in ks:
            if len(act) >= k:
                ch = rng.choice(len(act), size=k, replace=False)
                draws[(ti, k)] = [ST._first_after(ev, act[j]) + rng.normal(0, noise_sd) for j in ch]
    common = [ti for ti in range(cohort) if (ti, max(ks)) in draws]
    out = {"cohort": cohort, "noise_sd_min": noise_sd / 60, "tol_min": tol / 60, "n_common": len(common),
           "n_eligible": {f"k{k}": sum(1 for ti in range(cohort) if (ti, k) in draws) for k in ks},
           "chance": 1.0 / cohort}
    for variant in variants:
        rel = [release_starts2(r, variant, rng) for r in C]
        sv = ST._index(rel)
        res = {}
        for k in ks:
            succ = {}
            for ti in range(cohort):
                if (ti, k) not in draws:
                    continue
                sc = _scores(draws[(ti, k)], rel, sv, variant, tol)
                ties = np.flatnonzero(sc == sc.max())
                succ[ti] = (1.0 / len(ties)) if ti in ties else 0.0
            vals = np.array(list(succ.values()))
            bs = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(boot)]
            cv = np.array([succ[t] for t in common])
            bc = [float(np.mean(rng.choice(cv, len(cv), replace=True))) for _ in range(boot)]
            res[f"k{k}"] = float(vals.mean()); res[f"k{k}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res[f"k{k}_common"] = float(cv.mean()); res[f"k{k}_common_ci"] = [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]
            res[f"n_k{k}"] = len(vals)
        out[variant] = res
    return out


def time_map2(r, variant, rng):
    ev = sorted(A.ts(e["eventTime"]) for e in r["events"] if e["eventCategory"] == "Management")
    if variant != "R6":
        return ST._time_map(r, variant, rng)
    bs = A.bursts(ev)
    starts = np.array([b[0] for b in bs])

    def f(t):
        b = max(int(np.searchsorted(starts, t + 1e-9, side="right") - 1), 0)
        return b * DAY + (t - starts[b])
    return f, ev


def utility2(recs, eps_by, n=600, seed=9, variants=("R0", "R1", "R2", "R3", "R4", "R5", "R6")):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), size=n, replace=False)
    out = {}
    for variant in variants:
        ea, ew, cross_ok, order_ok = [], [], [], []
        for i in pick:
            r = recs[i]
            f, ev = time_map2(r, variant, rng)
            evm = [f(x) for x in ev]
            bs = A.bursts(ev)
            for e in eps_by[r["sid"]]:
                if e["censored"] or not e["code"].startswith("F"):
                    continue
                a0 = A.active_time(e["t0"], e["t1"], ev)
                a1 = A.active_time(f(e["t0"]), f(e["t1"]), sorted(evm))
                ea.append(abs(a1 - a0) / 60)
                ew.append(abs((f(e["t1"]) - f(e["t0"])) - (e["t1"] - e["t0"])) / 3600)
                cross = sum(1 for s, _ in bs if e["t0"] < s <= e["t1"]) > 0
                cross_m = (f(e["t1"]) - f(e["t0"])) > 12 * 3600
                cross_ok.append(cross == cross_m)
            order_ok.append(all(np.diff(evm) >= -1e-6))
        ea, ew = np.array(ea), np.array(ew)
        out[variant] = {"active_mae_min": float(ea.mean()), "active_p95_min": float(np.percentile(ea, 95)),
                        "active_exact_share": float(np.mean(ea < 1e-6)), "wall_mae_h": float(ew.mean()),
                        "wall_p90_h": float(np.percentile(ew, 90)), "wall_max_h": float(ew.max()),
                        "cross_session_flag_agreement": float(np.mean(cross_ok)),
                        "event_order_preserved_share": float(np.mean(order_ok))}
    return out


# ------------------------------------------------------------ RQ1 event loss (revised)
def _perturb(writes, mode, rate, rng):
    """Return list of (index_in_writes, event) actually delivered, in delivery order."""
    n = len(writes)
    idx = list(range(n))
    if mode == "iid":
        keep = rng.random(n) >= rate
        return [i for i in idx if keep[i]]
    if mode == "burst":                       # contiguous outage covering ~rate of the writes (expected)
        L = max(1, int(round(rate * n)))
        if rng.random() < min(1.0, rate * n / L):
            s = int(rng.integers(0, max(1, n - L + 1)))
            return [i for i in idx if not (s <= i < s + L)]
        return idx
    if mode == "create_only":
        keep = [not (writes[i][1]["eventName"].startswith("Create") and rng.random() < rate) for i in idx]
        return [i for i in idx if keep[i]]
    if mode == "delete_only":
        keep = [not (writes[i][1]["eventName"].startswith(("Delete", "Revoke", "Detach", "Disassociate", "Remove"))
                     and rng.random() < rate) for i in idx]
        return [i for i in idx if keep[i]]
    if mode == "duplicate":
        out = []
        for i in idx:
            out.append(i)
            if rng.random() < rate:
                out.append(i)
        return out
    if mode == "reorder":                     # adjacent swaps (delivery order differs from event order)
        out = idx[:]
        for j in range(n - 1):
            if rng.random() < rate:
                out[j], out[j + 1] = out[j + 1], out[j]
        return out
    raise ValueError(mode)


def event_loss2(recs, modes=("iid", "burst", "create_only", "delete_only", "duplicate", "reorder"),
                rates=(0.005, 0.01, 0.02, 0.05), resyncs=(None, 15, 60, 240), reps=20, seed=17):
    """Replay under perturbed delivery; state agreement after every true write, episode recall and
    precision (spurious episodes), and median time to recover after a divergence."""
    import copy as _copy
    from .replay import Replayer
    sub = [r for r in recs if r.get("dumps")]
    out = {"n_learners": len(sub), "n_states": int(sum(len(ST.mgmt_writes(r)) for r in sub)), "reps": reps}
    rng = np.random.default_rng(seed)
    for mode in modes:
        for rate in rates:
            for rs in resyncs:
                agree, rec_, prec_, rec_t, frs = [], [], [], [], []
                for rep in range(reps):
                    a_n = a_d = r_n = r_d = p_n = p_d = fr_n = 0
                    for r in sub:
                        writes = ST.mgmt_writes(r)
                        if not writes:
                            continue
                        t_or, v_or = A.oracle_timeline(r)
                        true_eps = A.episodes(t_or, v_or, T_END)
                        order = _perturb(writes, mode, rate, rng)
                        # arrival model: lost/duplicated events arrive at their own position; reordered
                        # events are processed in delivery order (one arrival per true write)
                        if mode == "reorder":
                            arrivals = [(pos, i) for pos, i in enumerate(order)]
                        else:
                            arrivals = [(i, i) for i in order]
                        rp = Replayer()
                        vecs = []
                        ptr = 0
                        next_sync = (t_or[0] + rs * 60) if rs else None
                        div_start = None
                        for k, (_, ev) in enumerate(writes):
                            tk = A.ts(ev["eventTime"])
                            if rs and tk >= next_sync and k > 0:
                                rp.resync(_copy.deepcopy(ST._lists_to_tuples(r["dumps"][k - 1][1])))
                                while next_sync <= tk:
                                    next_sync += rs * 60
                            while ptr < len(arrivals) and arrivals[ptr][0] <= k:
                                rp.apply(writes[arrivals[ptr][1]][1])
                                ptr += 1
                            vec = inv.evaluate(rp.st)
                            vecs.append(vec)
                            ok = vec == v_or[k]
                            if not ok and div_start is None:
                                div_start = tk
                            if ok and div_start is not None:
                                rec_t.append((tk - div_start) / 60)
                                div_start = None
                        a_n += sum(x == y for x, y in zip(v_or, vecs)); a_d += len(vecs); fr_n += rp.failed_refs
                        hat = A.episodes(t_or, vecs, T_END)
                        for e in true_eps:
                            r_d += 1
                            r_n += int(any(h["code"] == e["code"] and h["t0"] < e["t1"] and h["t1"] > e["t0"] for h in hat))
                        for h in hat:
                            p_d += 1
                            p_n += int(any(h["code"] == e["code"] and h["t0"] < e["t1"] and h["t1"] > e["t0"] for e in true_eps))
                    agree.append(a_n / a_d); rec_.append(r_n / r_d); prec_.append(p_n / max(p_d, 1)); frs.append(fr_n)
                key = f"{mode}|{rate}|{rs or 'none'}"
                q = lambda x: [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
                out[key] = {"agreement": q(agree), "episode_recall": q(rec_), "episode_precision": q(prec_),
                            "recovery_min_median": float(np.median(rec_t)) if rec_t else None,
                            "recovery_min_p90": float(np.percentile(rec_t, 90)) if rec_t else None,
                            "failed_refs_per_rep": float(np.mean(frs))}
    return out
