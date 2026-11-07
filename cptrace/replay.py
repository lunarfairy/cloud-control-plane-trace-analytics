"""Reconstruct configuration state from CloudTrail-shaped management events.

Only successful write events are applied.  Read events (responseElements =
null) carry no state.  Implicit resources that never appear in a create
event (the VPC main route table) are materialised the first time an event
references their identifier, and their VPC is inferred from the gateway or
subnet they are wired to.
"""
import copy
import json

from .world import empty_state, norm_perms, as_doc


def _first(x, *keys, default=None):
    for k in keys:
        if isinstance(x, dict) and k in x:
            x = x[k]
        else:
            return default
    return x


class Replayer:
    def __init__(self):
        self.st = empty_state()
        self.assoc_ids = {}          # association id -> (rt, subnet)
        self.lt_versions = {}        # lt id -> {version: data}
        self.lt_default = {}
        self.lt_names = {}
        self.unknown_ops = set()
        self.failed_refs = 0             # events whose referenced resources are unknown (e.g. after event loss)

    # ----------------------------------------------------------- utilities
    def _rt(self, rid, hint_vpc=None):
        rts = self.st["rts"]
        if rid not in rts:                       # implicit main route table
            rts[rid] = {"vpc": hint_vpc, "routes": [], "assoc": [], "main": True}
        elif rts[rid]["vpc"] is None and hint_vpc:
            rts[rid]["vpc"] = hint_vpc
        return rts[rid]

    def _vpc_of_target(self, p):
        if p.get("GatewayId") in self.st["igws"]:
            return self.st["igws"][p["GatewayId"]]["vpc"]
        n = self.st["nats"].get(p.get("NatGatewayId"))
        if n:
            return self.st["subnets"].get(n["subnet"], {}).get("vpc")
        return None

    def _set_route(self, rt, dest, p):
        rt["routes"] = [r for r in rt["routes"] if r[0] != dest]
        if p.get("GatewayId"):
            rt["routes"].append((dest, "igw", p["GatewayId"]))
        elif p.get("NatGatewayId"):
            rt["routes"].append((dest, "nat", p["NatGatewayId"]))
        else:
            rt["routes"].append((dest, "other", None))
        rt["routes"].sort()

    def _lt_apply(self, ltid):
        d = self.lt_versions[ltid][self.lt_default[ltid]]
        prof = d.get("IamInstanceProfile") or {}
        self.st["lts"][ltid] = {
            "name": self.lt_names[ltid], "sgs": sorted(d.get("SecurityGroupIds", [])),
            "profile": prof.get("Name") or (prof.get("Arn", "").split("/")[-1] if prof.get("Arn") else None)}

    # --------------------------------------------------------------- apply
    def apply(self, ev):
        if ev.get("readOnly") or ev.get("errorCode") or ev.get("eventCategory") != "Management":
            return False
        op, p, r = ev["eventName"], ev.get("requestParameters") or {}, ev.get("responseElements") or {}
        st = self.st
        h = getattr(self, "op_" + op, None)
        if h is None:
            self.unknown_ops.add(op)
            return False
        try:
            h(p, r, st)
        except (KeyError, TypeError, IndexError, ValueError):
            self.failed_refs += 1        # degrade gracefully: count and skip, state left unchanged
            return False
        return True

    def resync(self, st):
        """Replace the graph by a polled snapshot and rebuild launch-template side tables from it."""
        self.st = st
        self.lt_versions, self.lt_default, self.lt_names = {}, {}, {}
        for i, lt in st["lts"].items():
            self.lt_names[i] = lt.get("name")
            self.lt_versions[i] = {0: {"SecurityGroupIds": list(lt.get("sgs", [])),
                                       "IamInstanceProfile": {"Name": lt["profile"]} if lt.get("profile") else {}}}
            self.lt_default[i] = 0

    # EC2 networking
    def op_CreateVpc(self, p, r, st):
        st["vpcs"][r["Vpc"]["VpcId"]] = {"cidr": p["CidrBlock"]}

    def op_DeleteVpc(self, p, r, st):
        v = p["VpcId"]
        st["vpcs"].pop(v, None)
        for rid in [k for k, x in st["rts"].items() if x["vpc"] == v]:
            st["rts"].pop(rid)

    def op_CreateSubnet(self, p, r, st):
        s = r["Subnet"]
        st["subnets"][s["SubnetId"]] = {"vpc": p["VpcId"], "cidr": p["CidrBlock"],
                                        "az": s.get("AvailabilityZone") or p.get("AvailabilityZone"),
                                        "public_ip": False}

    def op_DeleteSubnet(self, p, r, st):
        st["subnets"].pop(p["SubnetId"], None)
        for rt in st["rts"].values():
            rt["assoc"] = [x for x in rt["assoc"] if x != p["SubnetId"]]

    def op_ModifySubnetAttribute(self, p, r, st):
        if p["SubnetId"] in st["subnets"] and "MapPublicIpOnLaunch" in p:
            st["subnets"][p["SubnetId"]]["public_ip"] = bool(p["MapPublicIpOnLaunch"].get("Value"))

    def op_CreateInternetGateway(self, p, r, st):
        st["igws"][r["InternetGateway"]["InternetGatewayId"]] = {"vpc": None}

    def op_AttachInternetGateway(self, p, r, st):
        if p["InternetGatewayId"] in st["igws"]:
            st["igws"][p["InternetGatewayId"]]["vpc"] = p["VpcId"]

    def op_DetachInternetGateway(self, p, r, st):
        if p["InternetGatewayId"] in st["igws"]:
            st["igws"][p["InternetGatewayId"]]["vpc"] = None

    def op_DeleteInternetGateway(self, p, r, st):
        st["igws"].pop(p["InternetGatewayId"], None)

    def op_AllocateAddress(self, p, r, st):
        pass

    def op_ReleaseAddress(self, p, r, st):
        pass

    def op_CreateNatGateway(self, p, r, st):
        st["nats"][r["NatGateway"]["NatGatewayId"]] = {"subnet": p["SubnetId"]}

    def op_DeleteNatGateway(self, p, r, st):
        st["nats"].pop(p["NatGatewayId"], None)

    def op_CreateRouteTable(self, p, r, st):
        st["rts"][r["RouteTable"]["RouteTableId"]] = {"vpc": p["VpcId"], "routes": [], "assoc": [], "main": False}

    def op_DeleteRouteTable(self, p, r, st):
        st["rts"].pop(p["RouteTableId"], None)

    def op_CreateRoute(self, p, r, st):
        rt = self._rt(p["RouteTableId"], self._vpc_of_target(p))
        self._set_route(rt, p["DestinationCidrBlock"], p)

    op_ReplaceRoute = op_CreateRoute

    def op_DeleteRoute(self, p, r, st):
        rt = self._rt(p["RouteTableId"])
        rt["routes"] = [x for x in rt["routes"] if x[0] != p["DestinationCidrBlock"]]

    def op_AssociateRouteTable(self, p, r, st):
        vpc = st["subnets"].get(p["SubnetId"], {}).get("vpc")
        rt = self._rt(p["RouteTableId"], vpc)
        for other in st["rts"].values():
            other["assoc"] = [x for x in other["assoc"] if x != p["SubnetId"]]
        rt["assoc"] = sorted(set(rt["assoc"]) | {p["SubnetId"]})
        self.assoc_ids[r.get("AssociationId")] = (p["RouteTableId"], p["SubnetId"])

    def op_DisassociateRouteTable(self, p, r, st):
        rid, sub = self.assoc_ids.pop(p["AssociationId"], (None, None))
        if rid in st["rts"]:
            st["rts"][rid]["assoc"] = [x for x in st["rts"][rid]["assoc"] if x != sub]

    def op_CreateSecurityGroup(self, p, r, st):
        st["sgs"][r["GroupId"]] = {"vpc": p.get("VpcId"), "name": p["GroupName"], "ingress": []}

    def op_DeleteSecurityGroup(self, p, r, st):
        st["sgs"].pop(p.get("GroupId"), None)

    def op_AuthorizeSecurityGroupIngress(self, p, r, st):
        g = st["sgs"].get(p["GroupId"])
        if g is not None:
            g["ingress"] = sorted(set(g["ingress"]) | set(norm_perms(p.get("IpPermissions", []))))

    def op_RevokeSecurityGroupIngress(self, p, r, st):
        g = st["sgs"].get(p["GroupId"])
        if g is not None:
            g["ingress"] = sorted(set(g["ingress"]) - set(norm_perms(p.get("IpPermissions", []))))

    # IAM
    def op_CreateRole(self, p, r, st):
        st["roles"][p["RoleName"]] = {"trust_ec2": "ec2.amazonaws.com" in p.get("AssumeRolePolicyDocument", ""),
                                      "inline": {}, "attached": []}

    def op_DeleteRole(self, p, r, st):
        st["roles"].pop(p["RoleName"], None)

    def op_PutRolePolicy(self, p, r, st):
        if p["RoleName"] in st["roles"]:
            st["roles"][p["RoleName"]]["inline"][p["PolicyName"]] = as_doc(p["PolicyDocument"])

    def op_DeleteRolePolicy(self, p, r, st):
        if p["RoleName"] in st["roles"]:
            st["roles"][p["RoleName"]]["inline"].pop(p["PolicyName"], None)

    def op_AttachRolePolicy(self, p, r, st):
        if p["RoleName"] in st["roles"]:
            a = st["roles"][p["RoleName"]]["attached"]
            st["roles"][p["RoleName"]]["attached"] = sorted(set(a) | {p["PolicyArn"]})

    def op_DetachRolePolicy(self, p, r, st):
        if p["RoleName"] in st["roles"]:
            a = st["roles"][p["RoleName"]]["attached"]
            st["roles"][p["RoleName"]]["attached"] = sorted(set(a) - {p["PolicyArn"]})

    def op_CreatePolicy(self, p, r, st):
        st["policies"][r["Policy"]["Arn"]] = as_doc(p["PolicyDocument"])

    def op_CreateInstanceProfile(self, p, r, st):
        st["profiles"][p["InstanceProfileName"]] = {"roles": []}

    def op_DeleteInstanceProfile(self, p, r, st):
        st["profiles"].pop(p["InstanceProfileName"], None)

    def op_AddRoleToInstanceProfile(self, p, r, st):
        pr = st["profiles"].get(p["InstanceProfileName"])
        if pr is not None:
            pr["roles"] = sorted(set(pr["roles"]) | {p["RoleName"]})

    def op_RemoveRoleFromInstanceProfile(self, p, r, st):
        pr = st["profiles"].get(p["InstanceProfileName"])
        if pr is not None:
            pr["roles"] = sorted(set(pr["roles"]) - {p["RoleName"]})

    # S3 (bucket-level calls are management events)
    def op_CreateBucket(self, p, r, st):
        st["buckets"] = sorted(set(st["buckets"]) | {p["Bucket"]})

    def op_DeleteBucket(self, p, r, st):
        st["buckets"] = sorted(set(st["buckets"]) - {p["Bucket"]})

    # Launch templates
    def op_CreateLaunchTemplate(self, p, r, st):
        lt = r["LaunchTemplate"]
        i = lt["LaunchTemplateId"]
        self.lt_names[i] = p["LaunchTemplateName"]
        self.lt_versions[i] = {1: p.get("LaunchTemplateData", {})}
        self.lt_default[i] = 1
        self._lt_apply(i)

    def _lt_id(self, p):
        if p.get("LaunchTemplateId"):
            return p["LaunchTemplateId"]
        for i, n in self.lt_names.items():
            if n == p.get("LaunchTemplateName") and i in self.lt_versions:
                return i
        return None

    def op_CreateLaunchTemplateVersion(self, p, r, st):
        i = self._lt_id(p)
        if i is None:
            return
        v = int(r["LaunchTemplateVersion"]["VersionNumber"])
        base = {}
        if p.get("SourceVersion"):
            sv = p["SourceVersion"]
            base = copy.deepcopy(self.lt_versions[i].get(self.lt_default[i] if sv == "$Default" else int(sv),
                                                         self.lt_versions[i].get(self.lt_default.get(i), {})))
        base.update(p.get("LaunchTemplateData", {}))
        self.lt_versions[i][v] = base

    def op_ModifyLaunchTemplate(self, p, r, st):
        i = self._lt_id(p)
        if i is None or "DefaultVersion" not in p:
            return
        self.lt_default[i] = int(p["DefaultVersion"])
        self._lt_apply(i)

    def op_DeleteLaunchTemplate(self, p, r, st):
        i = self._lt_id(p)
        if i is not None:
            st["lts"].pop(i, None)
            self.lt_versions.pop(i, None)

    # ELBv2
    def op_CreateTargetGroup(self, p, r, st):
        tg = r["TargetGroups"][0]
        st["tgs"][tg["TargetGroupArn"]] = {"vpc": p.get("VpcId"), "port": p.get("Port"),
                                           "hc_path": p.get("HealthCheckPath", "/")}

    def op_ModifyTargetGroup(self, p, r, st):
        tg = st["tgs"].get(p["TargetGroupArn"])
        if tg is not None and "HealthCheckPath" in p:
            tg["hc_path"] = p["HealthCheckPath"]

    def op_DeleteTargetGroup(self, p, r, st):
        st["tgs"].pop(p["TargetGroupArn"], None)

    def op_CreateLoadBalancer(self, p, r, st):
        lb = r["LoadBalancers"][0]
        st["lbs"][lb["LoadBalancerArn"]] = {"scheme": p.get("Scheme", "internet-facing"),
                                            "subnets": sorted(p.get("Subnets", [])),
                                            "sgs": sorted(p.get("SecurityGroups", []))}

    def op_DeleteLoadBalancer(self, p, r, st):
        st["lbs"].pop(p["LoadBalancerArn"], None)
        for la in [k for k, li in st["listeners"].items() if li["lb"] == p["LoadBalancerArn"]]:
            st["listeners"].pop(la)

    def op_SetSubnets(self, p, r, st):
        if p["LoadBalancerArn"] in st["lbs"]:
            st["lbs"][p["LoadBalancerArn"]]["subnets"] = sorted(p["Subnets"])

    def op_SetSecurityGroups(self, p, r, st):
        if p["LoadBalancerArn"] in st["lbs"]:
            st["lbs"][p["LoadBalancerArn"]]["sgs"] = sorted(p["SecurityGroups"])

    def op_CreateListener(self, p, r, st):
        li = r["Listeners"][0]
        fwd = [a.get("TargetGroupArn") for a in p.get("DefaultActions", []) if a.get("Type") == "forward"]
        st["listeners"][li["ListenerArn"]] = {"lb": p["LoadBalancerArn"], "port": p.get("Port"),
                                              "tg": fwd[0] if fwd else None}

    def op_ModifyListener(self, p, r, st):
        li = st["listeners"].get(p["ListenerArn"])
        if li is not None and p.get("DefaultActions"):
            fwd = [a.get("TargetGroupArn") for a in p["DefaultActions"] if a.get("Type") == "forward"]
            li["tg"] = fwd[0] if fwd else None

    def op_DeleteListener(self, p, r, st):
        st["listeners"].pop(p["ListenerArn"], None)

    # Auto Scaling
    def _lt_ref(self, p):
        lt = p.get("LaunchTemplate") or {}
        return lt.get("LaunchTemplateId") or self._lt_id({"LaunchTemplateName": lt.get("LaunchTemplateName")})

    def op_CreateAutoScalingGroup(self, p, r, st):
        st["asgs"][p["AutoScalingGroupName"]] = {
            "lt": self._lt_ref(p),
            "subnets": sorted(x for x in p.get("VPCZoneIdentifier", "").split(",") if x),
            "min": p.get("MinSize", 0), "tgs": sorted(p.get("TargetGroupARNs", []))}

    def op_UpdateAutoScalingGroup(self, p, r, st):
        a = st["asgs"].get(p["AutoScalingGroupName"])
        if a is None:
            return
        if "LaunchTemplate" in p:
            a["lt"] = self._lt_ref(p)
        if "VPCZoneIdentifier" in p:
            a["subnets"] = sorted(x for x in p["VPCZoneIdentifier"].split(",") if x)
        if "MinSize" in p:
            a["min"] = p["MinSize"]

    def op_AttachLoadBalancerTargetGroups(self, p, r, st):
        a = st["asgs"].get(p["AutoScalingGroupName"])
        if a is not None:
            a["tgs"] = sorted(set(a["tgs"]) | set(p["TargetGroupARNs"]))

    def op_DetachLoadBalancerTargetGroups(self, p, r, st):
        a = st["asgs"].get(p["AutoScalingGroupName"])
        if a is not None:
            a["tgs"] = sorted(set(a["tgs"]) - set(p["TargetGroupARNs"]))

    def op_DeleteAutoScalingGroup(self, p, r, st):
        st["asgs"].pop(p["AutoScalingGroupName"], None)


def replay_timeline(events):
    """Yield (eventTime, state-copy) after each state-changing management event."""
    rp = Replayer()
    out = []
    for ev in events:
        if rp.apply(ev):
            out.append((ev["eventTime"], copy.deepcopy(rp.st)))
    return out, rp
