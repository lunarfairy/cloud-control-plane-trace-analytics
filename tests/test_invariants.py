"""Independently specified semantic test cases for the twelve invariants.

Each case is a hand-built configuration whose expected verdict is derived from the
AWS documentation semantics stated in the comment, not from the evaluator's code.
Run:  python -m pytest -q tests
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cptrace import invariants as inv  # noqa: E402

PUT = {"Statement": [{"Effect": "Allow", "Action": "s3:PutObject", "Resource": "arn:aws:s3:::bkt/*"}]}


def reference():
    """A correct reference deployment (all invariants satisfied)."""
    return {
        "vpcs": {"vpc-1": {"cidr": "10.0.0.0/16"}},
        "subnets": {
            "pubA": {"vpc": "vpc-1", "cidr": "10.0.1.0/24", "az": "a", "public_ip": True},
            "pubB": {"vpc": "vpc-1", "cidr": "10.0.2.0/24", "az": "b", "public_ip": True},
            "prvA": {"vpc": "vpc-1", "cidr": "10.0.3.0/24", "az": "a", "public_ip": False},
            "prvB": {"vpc": "vpc-1", "cidr": "10.0.4.0/24", "az": "b", "public_ip": False},
        },
        "igws": {"igw-1": {"vpc": "vpc-1"}},
        "nats": {"nat-1": {"subnet": "pubA"}},
        "rts": {
            "rtb-main": {"vpc": "vpc-1", "routes": [], "assoc": [], "main": True},
            "rtb-pub": {"vpc": "vpc-1", "routes": [("0.0.0.0/0", "igw", "igw-1")], "assoc": ["pubA", "pubB"], "main": False},
            "rtb-prv": {"vpc": "vpc-1", "routes": [("0.0.0.0/0", "nat", "nat-1")], "assoc": ["prvA", "prvB"], "main": False},
        },
        "sgs": {
            "sg-alb": {"vpc": "vpc-1", "ingress": [("tcp", 80, 80, "cidr", "0.0.0.0/0")]},
            "sg-app": {"vpc": "vpc-1", "ingress": [("tcp", 80, 80, "sg", "sg-alb")]},
        },
        "roles": {"app-role": {"inline": {"s3-uploads": PUT}, "attached": []}},
        "policies": {},
        "profiles": {"app-profile": {"roles": ["app-role"]}},
        "buckets": ["bkt"],
        "lts": {"lt-1": {"name": "app", "sgs": ["sg-app"], "profile": "app-profile"}},
        "tgs": {"tg-1": {"vpc": "vpc-1", "port": 80, "hc_path": "/health"}},
        "lbs": {"lb-1": {"scheme": "internet-facing", "subnets": ["pubA", "pubB"], "sgs": ["sg-alb"]}},
        "listeners": {"li-1": {"lb": "lb-1", "port": 80, "tg": "tg-1"}},
        "asgs": {"asg-1": {"lt": "lt-1", "subnets": ["prvA", "prvB"], "min": 2, "tgs": ["tg-1"]}},
    }


def ev(st):
    return inv.evaluate(st, bucket="bkt")


def violated(st):
    return {c for c, x in ev(st).items() if x}


def test_reference_is_clean():
    v = ev(reference())
    assert all(x is False for x in v.values()), v


# ---- F1: ALB reachable only via IGW-routed subnets and an internet-facing scheme
def test_f1_internal_scheme():
    st = reference(); st["lbs"]["lb-1"]["scheme"] = "internal"
    assert "F1" in violated(st)


def test_f1_subnet_falls_back_to_main_table_without_igw():
    st = reference(); st["rts"]["rtb-pub"]["assoc"] = ["pubA"]   # pubB now uses the VPC main table
    assert "F1" in violated(st)


def test_f1_igw_not_attached_to_vpc():
    st = reference(); st["igws"]["igw-1"]["vpc"] = None
    assert "F1" in violated(st)


def test_implicit_main_table_makes_unassociated_subnet_public():
    st = reference()
    st["rts"]["rtb-main"]["routes"] = [("0.0.0.0/0", "igw", "igw-1")]
    st["rts"]["rtb-prv"]["assoc"] = ["prvA"]                      # prvB now uses the main table
    v = violated(st)
    assert "S2" in v and "F4" in v                                # IGW route but no public IP -> no egress


# ---- F2 / F3 / S1 / S5: security groups are allow-only; any matching rule admits
def test_f2_alb_sg_closed():
    st = reference(); st["sgs"]["sg-alb"]["ingress"] = [("tcp", 443, 443, "cidr", "0.0.0.0/0")]
    assert "F2" in violated(st)


def test_f2_all_traffic_rule_admits():
    st = reference(); st["sgs"]["sg-alb"]["ingress"] = [("-1", -1, -1, "cidr", "0.0.0.0/0")]
    assert "F2" not in violated(st)


def test_icmp_rule_does_not_admit_http():
    st = reference(); st["sgs"]["sg-alb"]["ingress"] = [("icmp", -1, -1, "cidr", "0.0.0.0/0")]
    assert "F2" in violated(st)


def test_f3_app_sg_self_reference():
    st = reference(); st["sgs"]["sg-app"]["ingress"] = [("tcp", 80, 80, "sg", "sg-app")]
    assert "F3" in violated(st)


def test_f3_vpc_cidr_source_is_sufficient():
    st = reference(); st["sgs"]["sg-app"]["ingress"] = [("tcp", 80, 80, "cidr", "10.0.0.0/16")]
    v = violated(st)
    assert "F3" not in v and "S1" not in v


def test_s1_world_open_app_tier_is_silent_only():
    st = reference(); st["sgs"]["sg-app"]["ingress"].append(("tcp", 80, 80, "cidr", "0.0.0.0/0"))
    assert violated(st) == {"S1"}


def test_s5_port_range_covering_22():
    st = reference(); st["sgs"]["sg-app"]["ingress"].append(("tcp", 0, 1024, "cidr", "0.0.0.0/0"))
    assert {"S1", "S5"} <= violated(st)


# ---- F4: egress
def test_f4_nat_in_private_subnet():
    st = reference(); st["nats"]["nat-1"]["subnet"] = "prvA"
    assert "F4" in violated(st)


def test_f4_no_default_route():
    st = reference(); st["rts"]["rtb-prv"]["routes"] = []
    assert "F4" in violated(st)


def test_public_subnets_with_public_ip_have_egress_but_violate_s2():
    st = reference(); st["asgs"]["asg-1"]["subnets"] = ["pubA", "pubB"]
    v = violated(st)
    assert "F4" not in v and "S2" in v


# ---- F5: reference-task health check (paths {/, /health}, port 80)
def test_f5_wrong_path():
    st = reference(); st["tgs"]["tg-1"]["hc_path"] = "/index.php"
    assert "F5" in violated(st)


def test_f5_wrong_port():
    st = reference(); st["tgs"]["tg-1"]["port"] = 8080
    assert "F5" in violated(st)


# ---- F6 / S4: identity-policy evaluation (explicit deny overrides allow; wildcards)
def test_f6_bucket_arn_without_object_wildcard():
    st = reference()
    st["roles"]["app-role"]["inline"] = {"p": {"Statement": [{"Effect": "Allow", "Action": "s3:PutObject",
                                                              "Resource": "arn:aws:s3:::bkt"}]}}
    assert "F6" in violated(st)


def test_f6_explicit_deny_overrides_allow():
    st = reference()
    st["roles"]["app-role"]["inline"]["deny"] = {"Statement": [{"Effect": "Deny", "Action": "s3:*", "Resource": "*"}]}
    assert "F6" in violated(st)


def test_f6_no_instance_profile_makes_s4_not_applicable():
    st = reference(); st["lts"]["lt-1"]["profile"] = None
    v = ev(st)
    assert v["F6"] is True and v["S4"] is None


def test_s4_full_access_managed_policy():
    st = reference(); st["roles"]["app-role"]["attached"] = ["arn:aws:iam::aws:policy/AmazonS3FullAccess"]
    v = violated(st)
    assert "S4" in v and "F6" not in v


def test_s4_is_syntactic_scope_limit():
    # documented scope limit: S4 flags an over-broad ALLOW statement even when a Deny removes its effect
    st = reference()
    st["roles"]["app-role"]["inline"]["wide"] = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
    st["roles"]["app-role"]["inline"]["deny"] = {"Statement": [{"Effect": "Deny", "Action": "s3:*", "Resource": "*"}]}
    v = violated(st)
    assert "S4" in v and "F6" in v


# ---- F7: wiring
def test_f7_listener_forwards_to_unregistered_tg():
    st = reference(); st["tgs"]["tg-2"] = {"vpc": "vpc-1", "port": 80, "hc_path": "/"}
    st["listeners"]["li-1"]["tg"] = "tg-2"
    assert "F7" in violated(st)


def test_asg_invariants_not_applicable_before_asg_exists():
    st = reference(); st["asgs"] = {}
    v = ev(st)
    assert v["F7"] is None and v["F4"] is None and v["S3"] is None


# ---- S3: resilience
def test_s3_single_az():
    st = reference(); st["asgs"]["asg-1"]["subnets"] = ["prvA"]
    assert "S3" in violated(st)


def test_s3_min_size_one():
    st = reference(); st["asgs"]["asg-1"]["min"] = 1
    assert violated(st) == {"S3"}


# ---- applicability, stack location and episode semantics
def test_deleted_lb_makes_lb_invariants_not_applicable():
    st = reference(); st["lbs"] = {}; st["listeners"] = {}
    v = ev(st)
    assert v["F1"] is None and v["F2"] is None and v["F7"] is None and v["F3"] is None


def test_stack_locator_prefers_fully_wired_lb():
    st = reference()
    st["lbs"]["lb-2"] = {"scheme": "internal", "subnets": ["prvA"], "sgs": ["sg-app"]}   # stray, unwired LB
    assert ev(st)["F1"] is False


def test_stack_locator_tie_breaks_to_most_recent():
    st = reference()
    st["lbs"]["lb-2"] = {"scheme": "internal", "subnets": ["prvA"], "sgs": ["sg-alb"]}
    st["listeners"]["li-2"] = {"lb": "lb-2", "port": 80, "tg": "tg-1"}                  # equally wired
    assert inv.locate_stack(st)["lb"] == "lb-2" and ev(st)["F1"] is True


def test_episode_carries_status_through_absence():
    from cptrace import analytics as A
    t = [0, 10, 20, 30]
    v = [{"F2": True}, {"F2": None}, {"F2": None}, {"F2": False}]
    assert A.episodes(t, v, 100, codes=["F2"]) == [{"code": "F2", "t0": 0, "t1": 30, "censored": False}]


def test_episode_censored_at_deadline():
    from cptrace import analytics as A
    assert A.episodes([0, 10], [{"S1": False}, {"S1": True}], 100, codes=["S1"]) == \
        [{"code": "S1", "t0": 10, "t1": 100, "censored": True}]
