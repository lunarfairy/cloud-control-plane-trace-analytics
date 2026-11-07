"""Declarative error-state invariants over a normalised configuration state.

Each invariant is a predicate on the reconstructed resource graph of the
target architecture (internet-facing ALB in public subnets -> target group ->
Auto Scaling group in private subnets with NAT egress -> instance role able
to write uploads to S3).  The application stack is located by *function*
(what is wired to what), never by resource names, because learners name
and re-create resources freely.

Codes: F* = functional (produce a user-visible symptom), S* = silent
(security / resilience defects with no symptom).
"""
import fnmatch

APP_PORT = 80
APP_PATHS = {"/", "/health"}
WORLD = "0.0.0.0/0"

CATALOGUE = {
    "F1": ("ALB_NOT_PUBLIC", "A load-balancer subnet has no default route to an attached internet gateway, or the ALB is internal"),
    "F2": ("ALB_SG_CLOSED", "No ALB security group admits TCP 80 from 0.0.0.0/0"),
    "F3": ("APP_SG_BLOCKS_ALB", "Instance security group does not admit TCP 80 from the ALB security group or VPC range"),
    "F4": ("NO_EGRESS_PATH", "An ASG subnet has no working egress (NAT in a public subnet, or IGW route plus public IP)"),
    "F5": ("HEALTH_CHECK_MISMATCH", "Target-group health check path or port does not match the application"),
    "F6": ("S3_WRITE_DENIED", "Instance profile/role cannot s3:PutObject into the upload bucket"),
    "F7": ("NOT_WIRED", "No port-80 listener forwards to a target group that an ASG is registered with"),
    "S1": ("APP_SG_OPEN_WORLD", "Instance security group admits TCP 80 directly from 0.0.0.0/0"),
    "S2": ("INSTANCES_PUBLIC", "ASG launches instances into subnets routed to the internet gateway"),
    "S3": ("SINGLE_AZ_OR_MIN1", "ASG spans fewer than two AZs or has MinSize < 2"),
    "S4": ("IAM_OVERBROAD", "Instance role grants s3:* / * on all resources (or a full-access managed policy)"),
    "S5": ("SSH_OPEN_WORLD", "Instance security group admits TCP 22 from 0.0.0.0/0"),
}
CODES = list(CATALOGUE)
FUNCTIONAL = [c for c in CODES if c.startswith("F")]
SILENT = [c for c in CODES if c.startswith("S")]

MANAGED = {
    "arn:aws:iam::aws:policy/AmazonS3FullAccess": {"Statement": [{"Effect": "Allow", "Action": ["s3:*", "s3-object-lambda:*"], "Resource": "*"}]},
    "arn:aws:iam::aws:policy/AdministratorAccess": {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]},
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess": {"Statement": [{"Effect": "Allow", "Action": ["s3:Get*", "s3:List*", "s3:Describe*"], "Resource": "*"}]},
}


# ---------------------------------------------------------------- helpers
def _lst(x):
    return x if isinstance(x, list) else [x]


def effective_rt(st, subnet):
    for rid, rt in st["rts"].items():
        if subnet in rt["assoc"]:
            return rt
    vpc = st["subnets"].get(subnet, {}).get("vpc")
    for rid, rt in st["rts"].items():
        if rt.get("main") and rt.get("vpc") == vpc:
            return rt
    return None


def _default_route(rt):
    if rt is None:
        return None
    for dest, kind, tgt in rt["routes"]:
        if dest == WORLD:
            return kind, tgt
    return None


def igw_routed(st, subnet):
    r = _default_route(effective_rt(st, subnet))
    if not r or r[0] != "igw":
        return False
    g = st["igws"].get(r[1])
    return g is not None and g.get("vpc") == st["subnets"].get(subnet, {}).get("vpc")


def nat_routed(st, subnet):
    r = _default_route(effective_rt(st, subnet))
    if not r or r[0] != "nat":
        return False
    n = st["nats"].get(r[1])
    return n is not None and n["subnet"] in st["subnets"] and igw_routed(st, n["subnet"])


def _port_ok(rule, port):
    proto, fp, tp = rule[0], rule[1], rule[2]
    if proto == "-1":
        return True
    if proto not in ("tcp", "6"):
        return False
    try:
        return int(fp) <= port <= int(tp)
    except (TypeError, ValueError):
        return False


def sg_rules(st, sgids):
    out = []
    for g in sgids:
        out.extend(st["sgs"].get(g, {}).get("ingress", []))
    return out


def admits_world(st, sgids, port):
    return any(_port_ok(r, port) and r[3] == "cidr" and r[4] == WORLD for r in sg_rules(st, sgids))


def admits_from(st, sgids, port, src_sgs, vpc_cidr):
    for r in sg_rules(st, sgids):
        if not _port_ok(r, port):
            continue
        if r[3] == "sg" and r[4] in src_sgs:
            return True
        if r[3] == "cidr" and r[4] in (WORLD, vpc_cidr):
            return True
    return False


def _match(pattern, value):
    return fnmatch.fnmatchcase(value.lower(), pattern.lower())


def policy_docs(st, role):
    r = st["roles"].get(role)
    if r is None:
        return []
    docs = list(r["inline"].values())
    for arn in r["attached"]:
        docs.append(MANAGED.get(arn) or st["policies"].get(arn) or {"Statement": []})
    return docs


def allows(docs, action, resource):
    allowed = False
    for d in docs:
        for s in _lst(d.get("Statement", [])):
            acts = _lst(s.get("Action", []))
            ress = _lst(s.get("Resource", []))
            hit = any(_match(a, action) for a in acts) and any(_match(x, resource) for x in ress)
            if hit and s.get("Effect") == "Deny":
                return False
            if hit and s.get("Effect") == "Allow":
                allowed = True
    return allowed


def overbroad(docs):
    for d in docs:
        for s in _lst(d.get("Statement", [])):
            if s.get("Effect") != "Allow":
                continue
            acts = _lst(s.get("Action", []))
            ress = _lst(s.get("Resource", []))
            if any(a in ("*", "s3:*") for a in acts) and "*" in ress:
                return True
    return False


# ------------------------------------------------------------ stack finder
def locate_stack(st):
    """Return dict(lb, listener, tg, asg, lt) chosen by wiring, or Nones."""
    asg_by_tg = {}
    for name, a in st["asgs"].items():
        for t in a["tgs"]:
            asg_by_tg.setdefault(t, name)
    best = None
    for arn in st["lbs"]:
        score, li_arn, tg = 0, None, None
        for la, li in st["listeners"].items():
            if li["lb"] == arn:
                score = max(score, 1)
                li_arn = li_arn or la
                if li["port"] == APP_PORT and li["tg"] in st["tgs"]:
                    score, li_arn, tg = max(score, 2), la, li["tg"]
                    if li["tg"] in asg_by_tg:
                        score = 3
                        break
        cand = (score, arn, li_arn, tg)
        if best is None or cand[0] >= best[0]:
            best = cand
    lb = best[1] if best else None
    tg = best[3] if best else None
    asg = asg_by_tg.get(tg) if tg else None
    if asg is None and st["asgs"]:
        asg = list(st["asgs"])[-1]
    if tg is None and st["tgs"]:
        tg = list(st["tgs"])[-1]
    lt = st["asgs"][asg]["lt"] if asg else None
    if lt not in st["lts"]:
        lt = None
    return {"lb": lb, "listener": best[2] if best else None, "tg": tg, "asg": asg, "lt": lt}


# --------------------------------------------------------------- evaluate
def evaluate(st, bucket=None):
    """Return {code: True (violated) | False (satisfied) | None (not applicable)}."""
    k = locate_stack(st)
    lb = st["lbs"].get(k["lb"]) if k["lb"] else None
    asg = st["asgs"].get(k["asg"]) if k["asg"] else None
    lt = st["lts"].get(k["lt"]) if k["lt"] else None
    tg = st["tgs"].get(k["tg"]) if k["tg"] else None
    out = {c: None for c in CODES}
    vpc_cidr = None
    if lb and lb["subnets"]:
        vpc = st["subnets"].get(lb["subnets"][0], {}).get("vpc")
        vpc_cidr = st["vpcs"].get(vpc, {}).get("cidr")
    if lb:
        out["F1"] = lb.get("scheme") == "internal" or not all(igw_routed(st, s) for s in lb["subnets"]) or not lb["subnets"]
        out["F2"] = not admits_world(st, lb["sgs"], APP_PORT)
        wired = any(li["lb"] == k["lb"] and li["port"] == APP_PORT and li["tg"] in st["tgs"]
                    and any(li["tg"] in a["tgs"] for a in st["asgs"].values())
                    for li in st["listeners"].values())
        out["F7"] = (not wired) if st["asgs"] else None   # only once an ASG exists (not a build transient)
    if tg:
        out["F5"] = tg.get("hc_path") not in APP_PATHS or tg.get("port") != APP_PORT
    if lt:
        out["S1"] = admits_world(st, lt["sgs"], APP_PORT)
        out["S5"] = admits_world(st, lt["sgs"], 22)
        if lb:
            out["F3"] = not admits_from(st, lt["sgs"], APP_PORT, set(lb["sgs"]), vpc_cidr)
    if asg:
        subs = [s for s in asg["subnets"] if s in st["subnets"]]
        egress = [nat_routed(st, s) or (igw_routed(st, s) and st["subnets"][s]["public_ip"]) for s in subs]
        out["F4"] = (not subs) or not all(egress)
        out["S2"] = any(igw_routed(st, s) for s in subs)
        azs = {st["subnets"][s]["az"] for s in subs}
        out["S3"] = len(azs) < 2 or asg["min"] < 2
    if lt is not None and asg:
        prof = st["profiles"].get(lt["profile"]) if lt["profile"] else None
        roles = prof["roles"] if prof else []
        bkt = bucket or (st["buckets"][0] if st["buckets"] else None)
        if bkt is not None:
            ok = any(allows(policy_docs(st, r), "s3:PutObject", f"arn:aws:s3:::{bkt}/uploads/probe.jpg") for r in roles)
            out["F6"] = not ok
        if roles:
            out["S4"] = any(overbroad(policy_docs(st, r)) for r in roles)
    return out


def symptom(v):
    """Top-layer symptom a learner sees in the browser (layered masking)."""
    if any(v.get(c) for c in ("F1", "F2", "F7")):
        return "unreachable"
    if any(v.get(c) for c in ("F3", "F4", "F5")):
        return "unhealthy"
    if v.get("F6"):
        return "upload_denied"
    return "ok"
