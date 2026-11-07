"""Emulated AWS control plane (moto) with a CloudTrail-shaped event recorder.

Every API call a simulated learner makes goes through World.call(), which
invokes the real boto3 client against moto and appends one record with the
CloudTrail management-event fields (eventTime, eventSource, eventName,
readOnly, requestParameters, responseElements, errorCode, errorMessage,
userIdentity, sourceIPAddress, userAgent, eventID, eventCategory).
Request/response payloads are kept in SDK shape; an adapter would map real
CloudTrail camelCase payloads onto the same normalised form.
"""
import datetime as dt
import json
import uuid

import boto3
from botocore.exceptions import ClientError, ParamValidationError

REGION = "us-east-1"
SOURCES = {"ec2": "ec2.amazonaws.com", "elbv2": "elasticloadbalancing.amazonaws.com",
           "autoscaling": "autoscaling.amazonaws.com", "iam": "iam.amazonaws.com",
           "s3": "s3.amazonaws.com"}
READ_PREFIXES = ("Describe", "Get", "List", "Head", "Lookup")


def _snake(op):
    out = []
    for i, ch in enumerate(op):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


class World:
    """One learner account inside an active moto mock_aws context."""

    def __init__(self, student_id, account_id, source_ip, user_agent, t0, rng):
        self.sid = student_id
        self.account = account_id
        self.ip = source_ip
        self.ua = user_agent
        self.t = t0                     # simulated wall clock (datetime, UTC)
        self.rng = rng
        self.events = []                # CloudTrail-shaped records
        self.dumps = []                 # (time, normalised true state) after each successful write
        kw = dict(region_name=REGION, aws_access_key_id="testing",
                  aws_secret_access_key="testing")
        self.c = {s: boto3.client(s, **kw) for s in SOURCES}

    # ------------------------------------------------------------------ calls
    def call(self, service, op, record=True, dump=True, **params):
        method = getattr(self.c[service], _snake(op))
        ro = op.startswith(READ_PREFIXES)
        err_code = err_msg = None
        resp = None
        try:
            resp = method(**params)
            resp = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
        except ClientError as e:
            err_code = e.response.get("Error", {}).get("Code", "ClientError")
            err_msg = e.response.get("Error", {}).get("Message", "")[:300]
        except ParamValidationError as e:
            err_code, err_msg = "ValidationException", str(e)[:300]
        if record:
            self.events.append({
                "eventVersion": "1.10",
                "eventID": str(uuid.UUID(int=(int(self.rng.integers(0, 2**62)) << 64) | len(self.events))),
                "eventTime": self.t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "eventSource": SOURCES[service],
                "eventName": op,
                "awsRegion": REGION,
                "sourceIPAddress": self.ip,
                "userAgent": self.ua,
                "userIdentity": {"type": "AssumedRole",
                                 "arn": f"arn:aws:sts::{self.account}:assumed-role/student/{self.sid}",
                                 "accountId": self.account},
                "eventType": "AwsApiCall",
                "eventCategory": "Management",
                "managementEvent": True,
                "readOnly": ro,
                "recipientAccountId": self.account,
                "requestParameters": _jsonable(params),
                # CloudTrail leaves responseElements null for most read calls
                "responseElements": None if (ro or err_code) else _jsonable(resp),
                "errorCode": err_code,
                "errorMessage": err_msg,
            })
        if dump and not ro and err_code is None:
            self.dumps.append((self.t, self.snapshot()))
        return resp, err_code

    def data_event(self, op, bucket, key, error_code):
        """S3 object-level (data-plane) event; only visible when data events are logged."""
        self.events.append({
            "eventVersion": "1.10", "eventID": str(uuid.UUID(int=int(self.rng.integers(0, 2**62)))),
            "eventTime": self.t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eventSource": "s3.amazonaws.com", "eventName": op, "awsRegion": REGION,
            "sourceIPAddress": "ec2.amazonaws.com", "userAgent": "aws-sdk-php",
            "userIdentity": {"type": "AssumedRole",
                             "arn": f"arn:aws:sts::{self.account}:assumed-role/app-role/i-instance",
                             "accountId": self.account},
            "eventType": "AwsApiCall", "eventCategory": "Data", "managementEvent": False,
            "readOnly": op.startswith(READ_PREFIXES), "recipientAccountId": self.account,
            "requestParameters": {"bucketName": bucket, "key": key},
            "responseElements": None, "errorCode": error_code,
            "errorMessage": "Access Denied" if error_code else None})

    def advance(self, seconds):
        self.t = self.t + dt.timedelta(seconds=float(seconds))

    # --------------------------------------------------------------- snapshot
    def snapshot(self):
        """Instructor-side describe-* snapshot, normalised (not recorded)."""
        ec2, elb, asg, iam, s3 = (self.c[k] for k in ("ec2", "elbv2", "autoscaling", "iam", "s3"))
        st = empty_state()
        for v in ec2.describe_vpcs()["Vpcs"]:
            if v.get("IsDefault"):
                continue
            st["vpcs"][v["VpcId"]] = {"cidr": v["CidrBlock"]}
        for s in ec2.describe_subnets()["Subnets"]:
            if s["VpcId"] in st["vpcs"]:
                st["subnets"][s["SubnetId"]] = {"vpc": s["VpcId"], "cidr": s["CidrBlock"],
                                                "az": s["AvailabilityZone"],
                                                "public_ip": bool(s.get("MapPublicIpOnLaunch"))}
        for g in ec2.describe_internet_gateways()["InternetGateways"]:
            att = [a["VpcId"] for a in g.get("Attachments", []) if a.get("VpcId") in st["vpcs"]]
            default_att = [a for a in g.get("Attachments", []) if a.get("VpcId") not in st["vpcs"]]
            if default_att:
                continue
            st["igws"][g["InternetGatewayId"]] = {"vpc": att[0] if att else None}
        for n in ec2.describe_nat_gateways()["NatGateways"]:
            if n.get("State") in ("deleted", "deleting"):
                continue
            st["nats"][n["NatGatewayId"]] = {"subnet": n["SubnetId"]}
        for rt in ec2.describe_route_tables()["RouteTables"]:
            if rt["VpcId"] not in st["vpcs"]:
                continue
            routes = []
            for r in rt.get("Routes", []):
                dest = r.get("DestinationCidrBlock")
                if not dest or r.get("GatewayId") == "local":
                    continue
                if r.get("GatewayId"):
                    routes.append((dest, "igw", r["GatewayId"]))
                elif r.get("NatGatewayId"):
                    routes.append((dest, "nat", r["NatGatewayId"]))
                else:
                    routes.append((dest, "other", None))
            assoc = {a["SubnetId"] for a in rt.get("Associations", []) if a.get("SubnetId")}
            main = any(a.get("Main") for a in rt.get("Associations", []))
            st["rts"][rt["RouteTableId"]] = {"vpc": rt["VpcId"], "routes": sorted(routes),
                                             "assoc": sorted(assoc), "main": main}
        for g in ec2.describe_security_groups()["SecurityGroups"]:
            if g["VpcId"] not in st["vpcs"] or g["GroupName"] == "default":
                continue
            st["sgs"][g["GroupId"]] = {"vpc": g["VpcId"], "name": g["GroupName"],
                                       "ingress": norm_perms(g.get("IpPermissions", []))}
        for r in iam.list_roles()["Roles"]:
            if r["Path"] != "/" or r["RoleName"].startswith("AWS"):
                continue
            name = r["RoleName"]
            doc = r.get("AssumeRolePolicyDocument", {})
            inline = {}
            for pn in iam.list_role_policies(RoleName=name)["PolicyNames"]:
                inline[pn] = as_doc(iam.get_role_policy(RoleName=name, PolicyName=pn)["PolicyDocument"])
            att = [p["PolicyArn"] for p in iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]]
            st["roles"][name] = {"trust_ec2": "ec2.amazonaws.com" in json.dumps(as_doc(doc)),
                                 "inline": inline, "attached": sorted(att)}
        for p in iam.list_policies(Scope="Local")["Policies"]:
            v = iam.get_policy_version(PolicyArn=p["Arn"], VersionId=p["DefaultVersionId"])
            st["policies"][p["Arn"]] = as_doc(v["PolicyVersion"]["Document"])
        for ip in iam.list_instance_profiles()["InstanceProfiles"]:
            st["profiles"][ip["InstanceProfileName"]] = {"roles": sorted(r["RoleName"] for r in ip["Roles"])}
        st["buckets"] = sorted(b["Name"] for b in s3.list_buckets().get("Buckets", []))
        for lt in ec2.describe_launch_templates()["LaunchTemplates"]:
            ver = ec2.describe_launch_template_versions(
                LaunchTemplateId=lt["LaunchTemplateId"], Versions=["$Default"])["LaunchTemplateVersions"][0]
            d = ver["LaunchTemplateData"]
            prof = d.get("IamInstanceProfile", {}) or {}
            st["lts"][lt["LaunchTemplateId"]] = {
                "name": lt["LaunchTemplateName"], "sgs": sorted(d.get("SecurityGroupIds", [])),
                "profile": prof.get("Name") or (prof.get("Arn", "").split("/")[-1] if prof.get("Arn") else None)}
        for tg in elb.describe_target_groups()["TargetGroups"]:
            st["tgs"][tg["TargetGroupArn"]] = {"vpc": tg.get("VpcId"), "port": tg.get("Port"),
                                               "hc_path": tg.get("HealthCheckPath", "/")}
        for lb in elb.describe_load_balancers()["LoadBalancers"]:
            st["lbs"][lb["LoadBalancerArn"]] = {
                "scheme": lb.get("Scheme"),
                "subnets": sorted(a["SubnetId"] for a in lb.get("AvailabilityZones", [])),
                "sgs": sorted(lb.get("SecurityGroups", []))}
            for li in elb.describe_listeners(LoadBalancerArn=lb["LoadBalancerArn"])["Listeners"]:
                fwd = [a.get("TargetGroupArn") for a in li.get("DefaultActions", []) if a.get("Type") == "forward"]
                st["listeners"][li["ListenerArn"]] = {"lb": lb["LoadBalancerArn"], "port": li.get("Port"),
                                                      "tg": fwd[0] if fwd else None}
        for g in asg.describe_auto_scaling_groups()["AutoScalingGroups"]:
            lt = g.get("LaunchTemplate", {}) or {}
            st["asgs"][g["AutoScalingGroupName"]] = {
                "lt": lt.get("LaunchTemplateId"),
                "subnets": sorted(x for x in g.get("VPCZoneIdentifier", "").split(",") if x),
                "min": g["MinSize"], "tgs": sorted(g.get("TargetGroupARNs", []))}
        return st


def empty_state():
    return {"vpcs": {}, "subnets": {}, "igws": {}, "nats": {}, "rts": {}, "sgs": {},
            "roles": {}, "policies": {}, "profiles": {}, "buckets": [], "lts": {},
            "tgs": {}, "lbs": {}, "listeners": {}, "asgs": {}}


def as_doc(d):
    if isinstance(d, str):
        from urllib.parse import unquote
        try:
            return json.loads(d)
        except json.JSONDecodeError:
            return json.loads(unquote(d))
    return d


def norm_perms(perms):
    """IpPermissions -> sorted tuples (proto, from, to, kind, source)."""
    out = []
    for p in perms:
        proto = str(p.get("IpProtocol"))
        fp, tp = p.get("FromPort", -1), p.get("ToPort", -1)
        for r in p.get("IpRanges", []):
            out.append((proto, fp, tp, "cidr", r.get("CidrIp")))
        for g in p.get("UserIdGroupPairs", []):
            out.append((proto, fp, tp, "sg", g.get("GroupId")))
    return sorted(set(out))
