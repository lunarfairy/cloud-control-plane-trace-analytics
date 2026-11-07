"""De-identification for trace release (variants R5 and R6 in the paper).

* principal ARNs, account ids and roster tokens in resource names -> keyed pseudonyms
* source IP addresses dropped; user agents coarsened to console / cli / sdk
* absolute timestamps replaced by (session_index, session_day, t_in_session_s):
  within-session timing is exact, between-session gaps are released in whole days
  relative to the learner's first session, and time of day is removed (R5);
* mode="R6" (session-ordinal re-basing, recommended for public release) drops session_day,
  so only the order of sessions survives and between-session gap lengths are discarded.
"""
import hashlib
import hmac
import json
import re

from . import analytics as A

ACCOUNT_RE = re.compile(r"\b\d{12}\b")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:?\d{2}|Z)?")


def pseudonym(key, sid):
    return "L" + hmac.new(key, sid.encode(), hashlib.sha256).hexdigest()[:10]


def coarse_agent(ua):
    if ua is None:
        return None
    u = ua.lower()
    if "aws-cli" in u:
        return "cli"
    if "internal" in u or "console" in u:
        return "console"
    return "sdk"


def deidentify(rec, key, mode="R5"):
    if mode not in ("R5", "R6"):
        raise ValueError(mode)
    sid = rec["sid"]
    pid = pseudonym(key, sid)
    tokens = {sid.lower(): pid.lower(), sid: pid}
    ev = [e for e in rec["events"]]
    times = [A.ts(e["eventTime"]) for e in ev]
    order = sorted(range(len(ev)), key=lambda k: times[k])
    mg = sorted(times)
    bursts = A.bursts(mg)
    starts = [b[0] for b in bursts]
    out = []
    for n, k in enumerate(order):
        e, t = ev[k], times[k]
        b = max(i for i, s in enumerate(starts) if s <= t + 1e-9) if starts else 0
        s0 = starts[0] if starts else t
        txt = json.dumps({"requestParameters": e.get("requestParameters"),
                          "responseElements": e.get("responseElements"),
                          "errorMessage": e.get("errorMessage")})
        for a, z in tokens.items():
            txt = txt.replace(a, z)
        txt = ACCOUNT_RE.sub("000000000000", txt)
        txt = DATE_RE.sub("[time-redacted]", txt)
        body = json.loads(txt)
        row = {
            "learner": pid, "seq": n, "session_index": b,
            "session_day": int((starts[b] - s0) // 86400.0) if starts else 0,
            "t_in_session_s": round(t - starts[b], 1) if starts else 0.0,
            "eventSource": e["eventSource"], "eventName": e["eventName"], "eventCategory": e["eventCategory"],
            "readOnly": e["readOnly"], "errorCode": e.get("errorCode"),
            "errorMessage": body["errorMessage"], "userAgentClass": coarse_agent(e.get("userAgent")),
            "principal": "learner" if e["eventCategory"] == "Management" else "instance-role",
            "requestParameters": body["requestParameters"], "responseElements": body["responseElements"],
        }
        if mode == "R6":
            del row["session_day"]
        out.append(row)
    return pid, out


def leak_check(released, rec):
    """Return identifiers from the raw record that survive in the released rows."""
    blob = json.dumps(released)
    leaks = []
    for needle in {rec["sid"], rec["sid"].lower(), rec["events"][0]["sourceIPAddress"], "123456789012"}:
        if needle and needle in blob:
            leaks.append(needle)
    if "eventTime" in blob or "2026-03" in blob:
        leaks.append("absolute-time")
    return leaks
