"""Trace analytics: episodes, channels, active time, strategy features, HMM bins."""
import datetime as dt

import numpy as np

from . import invariants as inv
from .replay import Replayer

EPOCH = dt.datetime(2026, 3, 2)
GAP = 30 * 60          # inactivity threshold (s) for active-time estimation
CODES = inv.CODES

# resource families touched by API calls (for on-/off-path features)
FAMILY = {
    "Vpc": "net", "Subnet": "net", "InternetGateway": "route", "RouteTable": "route", "Route": "route",
    "NatGateway": "route", "Address": "route", "SecurityGroup": "sg", "SecurityGroupIngress": "sg",
    "Role": "iam", "RolePolicy": "iam", "InstanceProfile": "iam", "RoleToInstanceProfile": "iam",
    "RoleFromInstanceProfile": "iam", "Policy": "iam", "Bucket": "iam", "LaunchTemplate": "compute",
    "LaunchTemplateVersion": "compute", "AutoScalingGroup": "compute", "LoadBalancerTargetGroups": "lb",
    "TargetGroup": "lb", "LoadBalancer": "lb", "Listener": "lb", "Subnets": "lb", "SecurityGroups": "lb",
    "SubnetAttribute": "net",
}
PATH = {"F1": {"route", "lb", "net"}, "F2": {"sg", "lb"}, "F7": {"lb", "compute"}, "F3": {"sg"},
        "F4": {"route", "compute", "net"}, "F5": {"lb"}, "F6": {"iam", "compute"},
        "S1": {"sg"}, "S2": {"compute", "route"}, "S3": {"compute"}, "S4": {"iam"}, "S5": {"sg"}}
VERBS = ("Create", "Delete", "Modify", "Authorize", "Revoke", "Put", "Attach", "Detach", "Add", "Remove",
         "Associate", "Disassociate", "Replace", "Update", "Set", "Allocate", "Release")


def ts(s):
    return (dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ") - EPOCH).total_seconds()


def family(op):
    for v in VERBS:
        if op.startswith(v):
            return FAMILY.get(op[len(v):], "other")
    return "read"


def resource_kind(op):
    for v in VERBS:
        if op.startswith(v):
            return op[len(v):]
    return None


# --------------------------------------------------------------- episodes
def episodes(times, vecs, t_end, codes=CODES):
    """Maximal intervals in which each invariant is violated.

    An episode ends only when the invariant is evaluated and satisfied; while the
    components it refers to are absent (None) the last evaluated status persists.
    """
    out = []
    for c in codes:
        on = None
        bad = False
        for t, v in zip(times, vecs):
            x = v.get(c)
            if x is None:          # component absent (e.g. mid delete-and-recreate): status carried forward
                continue
            bad = bool(x)
            if bad and on is None:
                on = t
            elif not bad and on is not None:
                out.append({"code": c, "t0": on, "t1": t, "censored": False})
                on = None
        if on is not None:
            out.append({"code": c, "t0": on, "t1": max(t_end, on), "censored": True})
    return out


def oracle_timeline(rec):
    times = [ts(t) for t, _, _ in rec["oracle"]]
    vecs = [v for _, v, _ in rec["oracle"]]
    return times, vecs


def replay_vectors(rec, bucket=None):
    rp = Replayer()
    times, vecs = [], []
    for ev in rec["events"]:
        if ev.get("eventCategory") != "Management" or ev.get("readOnly") or ev.get("errorCode"):
            continue
        rp.apply(ev)
        times.append(ts(ev["eventTime"]))
        vecs.append(inv.evaluate(rp.st, bucket=bucket))
    return times, vecs, rp


def snapshot_vectors(times, vecs, grid):
    """Vector observed by a describe-* poll at each grid time (state is piecewise constant)."""
    idx = np.searchsorted(np.asarray(times), grid, side="right") - 1
    empty = {c: None for c in CODES}
    return [vecs[i] if i >= 0 else empty for i in idx]


def match_snapshot(true_eps, grid, svecs):
    """A true episode is detected by polling iff some poll time falls inside it."""
    g = np.asarray(grid)
    res = []
    for e in true_eps:
        lo, hi = np.searchsorted(g, e["t0"], "left"), np.searchsorted(g, e["t1"], "left")
        hits = [k for k in range(lo, hi) if svecs[k].get(e["code"])]
        if hits:
            k0 = hits[0]
            k1 = k0
            while k1 < len(g) and svecs[k1].get(e["code"]):
                k1 += 1
            t1_hat = g[k1] if k1 < len(g) else g[-1]
            res.append((True, g[k0], t1_hat))
        else:
            res.append((False, None, None))
    return res


# ------------------------------------------------------------ active time
def session_overlap(t0, t1, sessions):
    tot = 0.0
    for a, b in sessions:
        lo, hi = max(t0, a), min(t1, b)
        if hi > lo:
            tot += hi - lo
    return tot


def active_time(t0, t1, ev_times, gap=GAP):
    """Kovanovic-style estimate: sum of inter-event gaps <= gap inside [t0, t1]."""
    x = [t for t in ev_times if t0 <= t <= t1]
    x = [t0] + x + [t1]
    d = np.diff(np.asarray(x))
    return float(d[d <= gap].sum())


def bursts(ev_times, gap=GAP):
    """Observable work sessions: clusters of events separated by > gap."""
    if not ev_times:
        return []
    out, s, prev = [], ev_times[0], ev_times[0]
    for t in ev_times[1:]:
        if t - prev > gap:
            out.append((s, prev))
            s = t
        prev = t
    out.append((s, prev))
    return out


# ----------------------------------------------------- strategy features
def strategy_features(events_t, code, t0, t1, channel="O1"):
    """Features of the last observable work burst before the episode cleared.

    events_t: list of (time, event) sorted.  channel O1 = reads+writes, O2 = writes only.
    """
    win = [(t, e) for t, e in events_t if t0 <= t <= t1 and e.get("eventCategory") == "Management"]
    if channel == "O2":
        win = [(t, e) for t, e in win if not e["readOnly"]]
    # restrict to the final burst (events within GAP chains ending at t1)
    tail = []
    for t, e in reversed(win):
        if tail and tail[-1][0] - t > GAP:
            break
        tail.append((t, e))
    tail.reverse()
    reads = [e for _, e in tail if e["readOnly"]]
    writes = [e for _, e in tail if not e["readOnly"]]
    fam_w = [family(e["eventName"]) for e in writes]
    path = PATH[code]
    deletes = [e for e in writes if e["eventName"].startswith("Delete")]
    created = {resource_kind(e["eventName"]) for e in writes if e["eventName"].startswith("Create")}
    recreate = any(resource_kind(d["eventName"]) in created for d in deletes)
    reads_path = sum(1 for e in reads if any(k in e["eventName"] for k in _READ_HINT[code]))
    return {
        "n_reads": len(reads), "n_reads_path": reads_path, "n_writes": len(writes),
        "n_writes_off": sum(1 for f in fam_w if f not in path), "n_deletes": len(deletes),
        "recreate": int(recreate), "n_errors": sum(1 for _, e in tail if e.get("errorCode")),
        "n_fam": len(set(fam_w)), "dur": (tail[-1][0] - tail[0][0]) if tail else 0.0,
        "world_open": sum(1 for e in writes if "0.0.0.0/0" in str(e.get("requestParameters")) and
                          e["eventName"] == "AuthorizeSecurityGroupIngress"),
    }


_READ_HINT = {"F1": ("RouteTable", "LoadBalancer", "Subnet"), "F2": ("SecurityGroup",),
              "F7": ("Listener", "AutoScaling"), "F3": ("SecurityGroup", "TargetHealth"),
              "F4": ("RouteTable", "NatGateway"), "F5": ("TargetGroup", "TargetHealth"),
              "F6": ("RolePolic", "LaunchTemplate", "InstanceProfile"),
              "S1": ("SecurityGroup",), "S2": ("AutoScaling",), "S3": ("AutoScaling",),
              "S4": ("Role",), "S5": ("SecurityGroup",)}


def rule_strategy(f, channel="O1"):
    """A-priori rule classifier (H / T / D) fixed before looking at results."""
    if f["recreate"] or f["n_deletes"] >= 2:
        return "D"
    if channel == "O1":
        if f["n_reads_path"] >= 1 and f["n_writes_off"] == 0 and f["world_open"] == 0:
            return "H"
        return "T"
    # write-only: no evidence of probing; fall back on write pattern
    if f["n_writes"] <= 2 and f["n_writes_off"] == 0 and f["world_open"] == 0:
        return "H"
    return "T"
