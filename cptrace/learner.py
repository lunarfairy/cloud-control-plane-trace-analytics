"""Generative model of a learner building the two-tier architecture.

The learner issues real boto3 calls (through World.call) to build, test and
debug.  Latent variables (skill theta, interface, strategy disposition,
misconception propensities, timetable) are drawn per learner; faults are
injected at build time with skill-dependent probabilities; debugging is
driven by the browser-visible symptom (invariants.symptom) and follows one
of three strategies:
  H  hypothesis-driven: targeted reads on the causal path, then a minimal fix
  T  trial-and-error: few reads, edits to plausible components, sometimes
     'fixing' by opening access (security debt)
  D  delete-and-recreate: tear down a component group and rebuild it
  help: after an attempt budget is exhausted the learner waits for a forum
     answer and applies the suggested fix.
Ground-truth labels (phase, strategy, debugging-attempt id) are written to a
separate ledger and never enter the analysed event stream.
"""
import datetime as dt
import json
import math

import numpy as np

from . import invariants as inv

AMI = None  # set by runner from moto's image catalogue


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# fault -> (base logit a_k, invariant it produces)
FAULTS = {
    "single_az":       (-2.0, "S3"), "main_rt_igw":    (-2.4, "S2"),
    "no_igw_route":    (-1.9, "F1"), "forget_assoc":   (-1.9, "F1"),
    "nat_private":     (-1.7, "F4"), "no_nat_route":   (-1.6, "F4"),
    "app_sg_wrong":    (-1.0, "F3"), "app_sg_world":   (-1.5, "S1"),
    "ssh_world":       (-0.8, "S5"), "alb_sg_closed":  (-2.3, "F2"),
    "iam_bucket_arn":  (-1.0, "F6"), "iam_overbroad":  (-1.4, "S4"),
    "no_profile":      (-1.7, "F6"), "hc_path":        (-1.6, "F5"),
    "alb_private":     (-2.0, "F1"), "no_listener":    (-2.8, "F7"),
    "asg_public":      (-2.1, "S2"), "asg_min1":       (-1.1, "S3"),
    "asg_no_tg":       (-1.9, "F7"),
}
SYMPTOM_CODES = {"unreachable": ["F1", "F2", "F7"], "unhealthy": ["F3", "F4", "F5"],
                 "upload_denied": ["F6"]}


class Learner:
    def __init__(self, sid, world, rng, params, t_deadline):
        self.sid, self.w, self.rng, self.P = sid, world, rng, params
        r = rng
        self.theta = float(r.normal())
        self.cli = bool(r.random() < params["p_cli"])
        u = r.normal(0, 0.6, 3)
        ss = params.get("strategy_skill_scale", 1.0)   # sensitivity: 0 decouples strategy from skill
        self.strat_logits = np.array([0.5 + ss * 0.9 * self.theta + u[0],
                                      0.4 - ss * 0.3 * self.theta + u[1],
                                      -0.5 - ss * 0.2 * self.theta + u[2]])
        self.mis = {k: float(r.normal(0, 0.5)) for k in FAULTS}
        self.lab_weekday = int(r.integers(0, 5))
        self.lab_hour = float(r.choice([9, 11, 13, 15, 17]))
        self.pref_hour = float(np.clip(r.normal(19.5, 2.0), 8, 23.5))
        self.deadline = t_deadline
        self.res = {}
        self.ledger = []          # hidden labels aligned with events
        self.attempts = []        # debugging attempts (ground truth)
        self.phase, self.strategy, self.attempt_id = "build", None, None
        self.session_end = None
        self.sessions = []
        self.helped = 0
        self.done = False
        self._new_session(first=True)

    # ----------------------------------------------------------- timing
    def _new_session(self, first=False):
        w, r = self.w, self.rng
        day0 = w.t.replace(hour=0, minute=0, second=0, microsecond=0)
        if first:
            day = day0 + dt.timedelta(days=int(r.integers(0, 4)))
        else:
            day = day0 + dt.timedelta(days=int(1 + r.poisson(self.P["gap_days"])))
        hour = self.lab_hour if day.weekday() == self.lab_weekday else self.pref_hour + r.normal(0, 0.7)
        start = day + dt.timedelta(hours=float(np.clip(hour, 7, 23.7)))
        if start < w.t:
            start = w.t + dt.timedelta(hours=10)
        w.t = start
        self.session_end = start + dt.timedelta(minutes=float(r.uniform(45, 150)))
        self.sessions.append([start.strftime("%Y-%m-%dT%H:%M:%SZ"), self.session_end.strftime("%Y-%m-%dT%H:%M:%SZ")])
        if w.t > self.deadline:
            self.done = True

    def think(self, median=None):
        m = (median or (25 if self.cli else 40)) * self.P.get("think_scale", 1.0)
        self.w.advance(self.rng.lognormal(math.log(m), 0.8))
        pi = self.P.get("p_idle", 0.0)                  # sensitivity: off-platform reading pauses
        if pi > 0 and self.rng.random() < pi:
            self.w.advance(60 * self.rng.lognormal(math.log(self.P.get("idle_median_min", 25.0)), 0.4))
        if self.w.t > self.session_end:
            self._new_session()

    def pause(self, minutes):
        self.w.advance(60 * self.rng.lognormal(math.log(minutes), 0.5))
        if self.w.t > self.session_end:
            self._new_session()

    # ------------------------------------------------------------- calls
    def call(self, svc, op, **kw):
        if self.done:
            return None, "Deadline"
        if not self.cli and not op.startswith(("Describe", "Get", "List")):
            for _ in range(self.rng.poisson(self.P["console_pre_reads"])):
                self._read(svc, background=True)
        n0 = len(self.w.events)
        resp, err = self.w.call(svc, op, **kw)
        self._label(n0)
        if not self.cli and not op.startswith(("Describe", "Get", "List")):
            for _ in range(self.rng.poisson(self.P["console_post_reads"])):
                self._read(svc, background=True)
        self.think()
        return resp, err

    def _label(self, n0, background=False):
        for i in range(n0, len(self.w.events)):
            self.ledger.append({"i": i, "phase": self.phase, "strategy": self.strategy,
                                "attempt": self.attempt_id, "background": background})

    READS = {"ec2": ["DescribeVpcs", "DescribeSubnets", "DescribeRouteTables", "DescribeSecurityGroups",
                     "DescribeNatGateways", "DescribeLaunchTemplates", "DescribeInternetGateways"],
             "elbv2": ["DescribeLoadBalancers", "DescribeTargetGroups"],
             "autoscaling": ["DescribeAutoScalingGroups"], "iam": ["ListRoles", "ListInstanceProfiles"],
             "s3": ["ListBuckets"]}

    def _read(self, svc, op=None, background=False, **kw):
        op = op or str(self.rng.choice(self.READS[svc]))
        if not kw:
            kw = self._read_kw(svc, op)
            if not kw and op in ("DescribeListeners", "DescribeTargetHealth", "DescribeLaunchTemplateVersions"):
                op = self.READS[svc][0]
        n0 = len(self.w.events)
        resp, err = self.w.call(svc, op, **kw)
        self._label(n0, background=background)
        self.w.advance(self.rng.exponential(3 if background else 12))
        return resp

    def p(self, fault, scale=1.0):
        a, code = FAULTS[fault]
        if code.startswith("S"):
            a = a + self.P.get("silent_shift", 0.0)     # sensitivity: silent-fault incidence
        return scale * sig(a - self.P["skill_slope"] * self.theta + self.mis[fault])

    def draw(self, fault, scale=1.0):
        return bool(self.rng.random() < self.p(fault, scale))

    # -------------------------------------------------------------- build
    def build(self):
        R, r = self.res, self.rng
        self.phase = "build"
        resp, _ = self.call("ec2", "CreateVpc", CidrBlock="10.0.0.0/16")
        R["vpc"] = resp["Vpc"]["VpcId"]
        self.pause(2)
        single = self.draw("single_az")
        plan = [("pubA", "10.0.1.0/24", "us-east-1a"), ("pubB", "10.0.2.0/24", "us-east-1b"),
                ("privA", "10.0.11.0/24", "us-east-1a"),
                ("privB", "10.0.12.0/24", "us-east-1a" if single else "us-east-1b")]
        for name, cidr, az in plan:
            if name.startswith("priv") and r.random() < self.P["p_cidr_conflict"]:
                self.call("ec2", "CreateSubnet", VpcId=R["vpc"], CidrBlock="10.0.1.0/24", AvailabilityZone=az)
            resp, err = self.call("ec2", "CreateSubnet", VpcId=R["vpc"], CidrBlock=cidr, AvailabilityZone=az)
            R[name] = resp["Subnet"]["SubnetId"]
        for name in ("pubA", "pubB"):
            if r.random() < 0.8:
                self.call("ec2", "ModifySubnetAttribute", SubnetId=R[name], MapPublicIpOnLaunch={"Value": True})
        self.pause(3)
        resp, _ = self.call("ec2", "CreateInternetGateway")
        R["igw"] = resp["InternetGateway"]["InternetGatewayId"]
        self.call("ec2", "AttachInternetGateway", InternetGatewayId=R["igw"], VpcId=R["vpc"])
        if self.draw("main_rt_igw"):
            rts = self._read("ec2", "DescribeRouteTables", Filters=[{"Name": "vpc-id", "Values": [R["vpc"]]}])
            R["main_rt"] = rts["RouteTables"][0]["RouteTableId"]
            self.call("ec2", "CreateRoute", RouteTableId=R["main_rt"], DestinationCidrBlock="0.0.0.0/0", GatewayId=R["igw"])
            R["pub_rt"] = R["main_rt"]
            R["pub_rt_is_main"] = True
        else:
            resp, _ = self.call("ec2", "CreateRouteTable", VpcId=R["vpc"])
            R["pub_rt"] = resp["RouteTable"]["RouteTableId"]
            if not self.draw("no_igw_route"):
                self.call("ec2", "CreateRoute", RouteTableId=R["pub_rt"], DestinationCidrBlock="0.0.0.0/0", GatewayId=R["igw"])
            self.call("ec2", "AssociateRouteTable", RouteTableId=R["pub_rt"], SubnetId=R["pubA"])
            if not self.draw("forget_assoc"):
                self.call("ec2", "AssociateRouteTable", RouteTableId=R["pub_rt"], SubnetId=R["pubB"])
        self.pause(4)
        resp, _ = self.call("ec2", "AllocateAddress", Domain="vpc")
        R["eip"] = resp["AllocationId"]
        nat_sub = R["privA"] if self.draw("nat_private") else R["pubA"]
        resp, _ = self.call("ec2", "CreateNatGateway", SubnetId=nat_sub, AllocationId=R["eip"])
        R["nat"] = resp["NatGateway"]["NatGatewayId"]
        resp, _ = self.call("ec2", "CreateRouteTable", VpcId=R["vpc"])
        R["priv_rt"] = resp["RouteTable"]["RouteTableId"]
        if not self.draw("no_nat_route"):
            self.call("ec2", "CreateRoute", RouteTableId=R["priv_rt"], DestinationCidrBlock="0.0.0.0/0", NatGatewayId=R["nat"])
        for name in ("privA", "privB"):
            self.call("ec2", "AssociateRouteTable", RouteTableId=R["priv_rt"], SubnetId=R[name])
        self.pause(5)
        self._make_sgs()
        self.pause(3)
        R["bucket"] = f"uploads-{self.sid.lower()}"
        self.call("s3", "CreateBucket", Bucket=R["bucket"])
        self._make_iam()
        self.pause(4)
        self._make_lt()
        self._make_tg()
        self._make_alb()
        self._make_asg()

    def _perm(self, port, cidr=None, sg=None, proto="tcp"):
        p = {"IpProtocol": proto, "FromPort": port, "ToPort": port}
        if proto == "icmp":
            p.update(FromPort=-1, ToPort=-1)
        if cidr:
            p["IpRanges"] = [{"CidrIp": cidr}]
        if sg:
            p["UserIdGroupPairs"] = [{"GroupId": sg}]
        return [p]

    def _make_sgs(self):
        R = self.res
        resp, err = self.call("ec2", "CreateSecurityGroup", GroupName="alb-sg", Description="alb", VpcId=R["vpc"])
        if err == "InvalidGroup.Duplicate":
            resp, err = self.call("ec2", "CreateSecurityGroup", GroupName=f"alb-sg-{len(self.w.events)}", Description="alb", VpcId=R["vpc"])
        R["alb_sg"] = resp["GroupId"]
        if not self.draw("alb_sg_closed"):
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["alb_sg"], IpPermissions=self._perm(80, cidr="0.0.0.0/0"))
        resp, err = self.call("ec2", "CreateSecurityGroup", GroupName="app-sg", Description="app", VpcId=R["vpc"])
        R["app_sg"] = resp["GroupId"]
        if self.draw("app_sg_world"):
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, cidr="0.0.0.0/0"))
        elif self.draw("app_sg_wrong"):
            # classic slip: reference the group itself / the wrong port
            if self.rng.random() < 0.5:
                self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, sg=R["app_sg"]))
            else:
                self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(443, sg=R["alb_sg"]))
        else:
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, sg=R["alb_sg"]))
        if self.draw("ssh_world"):
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(22, cidr="0.0.0.0/0"))

    def _policy(self, broken):
        res = f"arn:aws:s3:::{self.res['bucket']}" + ("" if broken else "/*")
        return json.dumps({"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": res}]})

    def _make_iam(self, scale=1.0):
        R = self.res
        trust = json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]})
        self.call("iam", "CreateRole", RoleName="app-role", AssumeRolePolicyDocument=trust)
        if self.draw("iam_overbroad", scale):
            self.call("iam", "AttachRolePolicy", RoleName="app-role", PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess")
        else:
            self.call("iam", "PutRolePolicy", RoleName="app-role", PolicyName="s3-uploads",
                      PolicyDocument=self._policy(self.draw("iam_bucket_arn", scale)))
        self.call("iam", "CreateInstanceProfile", InstanceProfileName="app-profile")
        self.call("iam", "AddRoleToInstanceProfile", InstanceProfileName="app-profile", RoleName="app-role")

    def _make_lt(self, scale=1.0):
        R = self.res
        data = {"ImageId": AMI, "InstanceType": "t3.micro", "SecurityGroupIds": [R["app_sg"]],
                "UserData": "IyEvYmluL2Jhc2gKZG5mIGluc3RhbGwgLXkgaHR0cGQgcGhwCg=="}
        if not self.draw("no_profile", scale):
            data["IamInstanceProfile"] = {"Name": "app-profile"}
        name = "app-lt" if "lt_gen" not in R else f"app-lt-{R['lt_gen']}"
        R["lt_gen"] = R.get("lt_gen", 0) + 1
        resp, err = self.call("ec2", "CreateLaunchTemplate", LaunchTemplateName=name, LaunchTemplateData=data)
        R["lt"] = resp["LaunchTemplate"]["LaunchTemplateId"] if resp else None
        R["lt_data"] = data

    def _make_tg(self, scale=1.0):
        R = self.res
        path = self.rng.choice(["/healthz", "/index.php", "/status"]) if self.draw("hc_path", scale) else "/health"
        name = "app-tg" if "tg_gen" not in R else f"app-tg-{R['tg_gen']}"
        R["tg_gen"] = R.get("tg_gen", 0) + 1
        resp, err = self.call("elbv2", "CreateTargetGroup", Name=name, Protocol="HTTP", Port=80,
                              VpcId=R["vpc"], HealthCheckPath=str(path), TargetType="instance")
        R["tg"] = resp["TargetGroups"][0]["TargetGroupArn"]

    def _make_alb(self, scale=1.0):
        R = self.res
        subs = [R["privA"], R["privB"]] if self.draw("alb_private", scale) else [R["pubA"], R["pubB"]]
        name = "app-alb" if "lb_gen" not in R else f"app-alb-{R['lb_gen']}"
        R["lb_gen"] = R.get("lb_gen", 0) + 1
        resp, err = self.call("elbv2", "CreateLoadBalancer", Name=name, Subnets=subs, SecurityGroups=[R["alb_sg"]],
                              Scheme="internet-facing", Type="application")
        R["lb"] = resp["LoadBalancers"][0]["LoadBalancerArn"]
        R["listener"] = None
        if not self.draw("no_listener", scale):
            self._make_listener()

    def _make_listener(self):
        R = self.res
        resp, err = self.call("elbv2", "CreateListener", LoadBalancerArn=R["lb"], Protocol="HTTP", Port=80,
                              DefaultActions=[{"Type": "forward", "TargetGroupArn": R["tg"]}])
        R["listener"] = resp["Listeners"][0]["ListenerArn"] if resp else None

    def _make_asg(self, scale=1.0):
        R = self.res
        subs = [R["pubA"], R["pubB"]] if self.draw("asg_public", scale) else [R["privA"], R["privB"]]
        mn = 1 if self.draw("asg_min1", scale) else 2
        name = "app-asg" if "asg_gen" not in R else f"app-asg-{R['asg_gen']}"
        R["asg_gen"] = R.get("asg_gen", 0) + 1
        kw = dict(AutoScalingGroupName=name, LaunchTemplate={"LaunchTemplateId": R["lt"], "Version": "$Default"},
                  MinSize=mn, MaxSize=4, DesiredCapacity=mn, VPCZoneIdentifier=",".join(subs))
        if not self.draw("asg_no_tg", scale):
            kw["TargetGroupARNs"] = [R["tg"]]
        self.call("autoscaling", "CreateAutoScalingGroup", **kw)
        R["asg"] = name

    # ------------------------------------------------------- truth access
    def truth(self):
        st = self.w.dumps[-1][1] if self.w.dumps else self.w.snapshot()
        return inv.evaluate(st, bucket=self.res.get("bucket")), st

    # --------------------------------------------------------------- test
    def test(self):
        """Browser test: no control-plane trace except S3 data events for uploads."""
        v, _ = self.truth()
        s = inv.symptom(v)
        if s in ("upload_denied", "ok") and v.get("F6") is not None:
            self.w.data_event("PutObject", self.res["bucket"], "uploads/test.jpg",
                              "AccessDenied" if v.get("F6") else None)
        self.w.advance(self.rng.lognormal(math.log(90), 0.5))
        return s, v

    # ---------------------------------------------------------- debugging
    def pick_strategy(self):
        z = self.strat_logits - self.strat_logits.max()
        pr = np.exp(z) / np.exp(z).sum()
        return ["H", "T", "D"][int(self.rng.choice(3, p=pr))]

    def debug_loop(self):
        budget = int(2 + self.rng.poisson(self.P["attempt_budget"]))
        tries = 0
        while not self.done:
            s, v = self.test()
            if s == "ok":
                return True
            if tries >= budget:
                self._help(s, v)
                tries, budget = 0, int(2 + self.rng.poisson(self.P["attempt_budget"]))
                continue
            self.phase = "debug"
            self.strategy = self.pick_strategy()
            self.attempt_id = f"{self.sid}-a{len(self.attempts)}"
            t0 = self.w.t
            if self.rng.random() < sig(-1.6 - 0.8 * self.theta) and s != "upload_denied":
                self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=self.res["alb_sg"],
                          IpPermissions=self._perm(-1, cidr="0.0.0.0/0", proto="icmp"))   # ping-for-HTTP marker
            getattr(self, "_debug_" + self.strategy)(s, v)
            self.attempts.append({"attempt": self.attempt_id, "strategy": self.strategy, "symptom": s,
                                  "t0": t0.isoformat(), "t1": self.w.t.isoformat(),
                                  "causes": [c for c in SYMPTOM_CODES[s] if v.get(c)]})
            self.strategy, self.attempt_id = None, None
            self.phase = "build"
            tries += 1
        return False

    def _causes(self, s, v):
        return [c for c in SYMPTOM_CODES[s] if v.get(c)]

    def _debug_H(self, s, v):
        causes = self._causes(s, v)
        cands = list(SYMPTOM_CODES[s])
        q = sig(0.4 + 1.0 * self.theta)
        order = causes + [c for c in cands if c not in causes] if self.rng.random() < q else list(self.rng.permutation(cands))
        for c in order[:2]:
            hp = self.P.get("h_read_p", 1.0)            # sensitivity: weaker H read signature
            for op_svc in self._probe_ops(c):
                if hp >= 1.0 or self.rng.random() < hp:
                    self._read(*op_svc)
            self.pause(2)
            if c in causes:
                if self.rng.random() < 0.85:
                    self.fix(c)
                return
        self.pause(3)

    def _debug_T(self, s, v):
        causes = self._causes(s, v)
        if self.rng.random() < self.P.get("t_read_p", 0.3):
            self._read(*self._probe_ops(self.rng.choice(SYMPTOM_CODES[s]))[0])
            for _ in range(int(self.P.get("t_extra_reads", 0))):   # sensitivity: T learners who read
                cc = str(self.rng.choice(SYMPTOM_CODES[s]))
                self._read(*self._probe_ops(cc)[int(self.rng.integers(len(self._probe_ops(cc))))])
        c = str(self.rng.choice(SYMPTOM_CODES[s]))
        if c in causes and self.rng.random() < 0.55:
            self.fix(c, sloppy=True)
        else:
            self.tinker(c)

    def _debug_D(self, s, v):
        R = self.res
        if s == "unreachable":
            if R.get("listener"):
                self.call("elbv2", "DeleteListener", ListenerArn=R["listener"])
            self.call("elbv2", "DeleteLoadBalancer", LoadBalancerArn=R["lb"])
            self.pause(3)
            self._make_alb(scale=0.5)
        elif s == "unhealthy":
            self.call("autoscaling", "DeleteAutoScalingGroup", AutoScalingGroupName=R["asg"], ForceDelete=True)
            _, err = self.call("elbv2", "DeleteTargetGroup", TargetGroupArn=R["tg"])
            if err and R.get("listener"):
                self.call("elbv2", "DeleteListener", ListenerArn=R["listener"])
                self.call("elbv2", "DeleteTargetGroup", TargetGroupArn=R["tg"])
                R["listener"] = None
            self.call("ec2", "DeleteLaunchTemplate", LaunchTemplateId=R["lt"])
            self.pause(3)
            self._make_lt(scale=0.5)
            self._make_tg(scale=0.5)
            if R.get("listener"):
                self.call("elbv2", "ModifyListener", ListenerArn=R["listener"],
                          DefaultActions=[{"Type": "forward", "TargetGroupArn": R["tg"]}])
            else:
                self._make_listener()
            self._make_asg(scale=0.5)
        else:
            self.call("iam", "RemoveRoleFromInstanceProfile", InstanceProfileName="app-profile", RoleName="app-role")
            self.call("iam", "DeleteInstanceProfile", InstanceProfileName="app-profile")
            pols = self.w.c["iam"].list_role_policies(RoleName="app-role")["PolicyNames"]
            for pn in pols:
                self.call("iam", "DeleteRolePolicy", RoleName="app-role", PolicyName=pn)
            for a in self.w.c["iam"].list_attached_role_policies(RoleName="app-role")["AttachedPolicies"]:
                self.call("iam", "DetachRolePolicy", RoleName="app-role", PolicyArn=a["PolicyArn"])
            self.call("iam", "DeleteRole", RoleName="app-role")
            self.pause(2)
            self._make_iam(scale=0.5)

    def _help(self, s, v):
        """Forum help: long wall-clock wait, then a correct, targeted fix."""
        self.phase, self.strategy = "debug", "help"
        self.attempt_id = f"{self.sid}-a{len(self.attempts)}"
        t0 = self.w.t
        self.w.advance(3600 * self.rng.uniform(4, 30))
        self._new_session()
        causes = self._causes(s, v)
        for c in causes[:1]:
            for op_svc in self._probe_ops(c)[:1]:
                self._read(*op_svc)
            self.fix(c)
        self.helped += 1
        self.attempts.append({"attempt": self.attempt_id, "strategy": "help", "symptom": s,
                              "t0": t0.isoformat(), "t1": self.w.t.isoformat(), "causes": causes})
        self.strategy, self.attempt_id, self.phase = None, None, "build"

    def _probe_ops(self, code):
        m = {"F1": [("ec2", "DescribeRouteTables"), ("elbv2", "DescribeLoadBalancers")],
             "F2": [("ec2", "DescribeSecurityGroups")],
             "F7": [("elbv2", "DescribeListeners"), ("autoscaling", "DescribeAutoScalingGroups")],
             "F3": [("ec2", "DescribeSecurityGroups"), ("elbv2", "DescribeTargetHealth")],
             "F4": [("ec2", "DescribeRouteTables"), ("ec2", "DescribeNatGateways")],
             "F5": [("elbv2", "DescribeTargetGroups"), ("elbv2", "DescribeTargetHealth")],
             "F6": [("iam", "ListRolePolicies"), ("ec2", "DescribeLaunchTemplateVersions")]}
        return m[code]

    def _read_kw(self, svc, op):
        R = self.res
        need = {"DescribeListeners": ("LoadBalancerArn", "lb"), "DescribeTargetHealth": ("TargetGroupArn", "tg"),
                "ListRolePolicies": ("RoleName", None),
                "DescribeLaunchTemplateVersions": ("LaunchTemplateId", "lt")}
        if op not in need:
            return {}
        key, res = need[op]
        if res is None:
            return {"RoleName": "app-role"}
        return {key: R[res]} if R.get(res) else {}

    # ------------------------------------------------------------ repairs
    def fix(self, code, sloppy=False):
        R = self.res
        v, st = self.truth()
        if not v.get(code):
            return
        if code == "F1":
            lb = st["lbs"].get(R["lb"])
            if lb and not all(inv.igw_routed(st, s) for s in lb["subnets"]) and set(lb["subnets"]) & {R["privA"], R["privB"]}:
                self.call("elbv2", "SetSubnets", LoadBalancerArn=R["lb"], Subnets=[R["pubA"], R["pubB"]])
                v, st = self.truth()
                if not v.get("F1"):
                    return
            rt = st["rts"].get(R["pub_rt"], {"routes": []})
            if not any(x[0] == "0.0.0.0/0" for x in rt["routes"]):
                self.call("ec2", "CreateRoute", RouteTableId=R["pub_rt"], DestinationCidrBlock="0.0.0.0/0", GatewayId=R["igw"])
            for sname in ("pubA", "pubB"):
                if not inv.igw_routed(st, R[sname]) and not R.get("pub_rt_is_main"):
                    self.call("ec2", "AssociateRouteTable", RouteTableId=R["pub_rt"], SubnetId=R[sname])
        elif code == "F2":
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["alb_sg"], IpPermissions=self._perm(80, cidr="0.0.0.0/0"))
        elif code == "F3":
            if sloppy and self.rng.random() < 0.6:
                self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, cidr="0.0.0.0/0"))
            else:
                self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, sg=R["alb_sg"]))
        elif code == "F4":
            nat = st["nats"].get(R["nat"])
            if nat and not inv.igw_routed(st, nat["subnet"]):
                self.call("ec2", "DeleteNatGateway", NatGatewayId=R["nat"])
                resp, _ = self.call("ec2", "AllocateAddress", Domain="vpc")
                resp, _ = self.call("ec2", "CreateNatGateway", SubnetId=R["pubA"], AllocationId=resp["AllocationId"])
                R["nat"] = resp["NatGateway"]["NatGatewayId"]
                self.call("ec2", "ReplaceRoute" if any(x[0] == "0.0.0.0/0" for x in st["rts"][R["priv_rt"]]["routes"]) else "CreateRoute",
                          RouteTableId=R["priv_rt"], DestinationCidrBlock="0.0.0.0/0", NatGatewayId=R["nat"])
            elif not any(x[0] == "0.0.0.0/0" for x in st["rts"][R["priv_rt"]]["routes"]):
                self.call("ec2", "CreateRoute", RouteTableId=R["priv_rt"], DestinationCidrBlock="0.0.0.0/0", NatGatewayId=R["nat"])
            v, st = self.truth()
            if v.get("F4"):
                self.call("autoscaling", "UpdateAutoScalingGroup", AutoScalingGroupName=R["asg"],
                          VPCZoneIdentifier=",".join([R["privA"], R["privB"]]))
        elif code == "F5":
            self.call("elbv2", "ModifyTargetGroup", TargetGroupArn=R["tg"], HealthCheckPath="/health")
        elif code == "F6":
            lt = st["lts"].get(R["lt"])
            if lt and not lt["profile"]:
                data = dict(R["lt_data"], IamInstanceProfile={"Name": "app-profile"})
                resp, _ = self.call("ec2", "CreateLaunchTemplateVersion", LaunchTemplateId=R["lt"], LaunchTemplateData=data)
                if resp:
                    self.call("ec2", "ModifyLaunchTemplate", LaunchTemplateId=R["lt"],
                              DefaultVersion=str(resp["LaunchTemplateVersion"]["VersionNumber"]))
                    R["lt_data"] = data
                v, st = self.truth()
                if not v.get("F6"):
                    return
            if "app-profile" not in st["profiles"]:
                self.call("iam", "CreateInstanceProfile", InstanceProfileName="app-profile")
                self.call("iam", "AddRoleToInstanceProfile", InstanceProfileName="app-profile", RoleName="app-role")
            if sloppy and self.rng.random() < 0.6:
                self.call("iam", "AttachRolePolicy", RoleName="app-role", PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess")
            else:
                self.call("iam", "PutRolePolicy", RoleName="app-role", PolicyName="s3-uploads", PolicyDocument=self._policy(False))
        elif code == "F7":
            if not R.get("listener"):
                self._make_listener()
            asg = st["asgs"].get(R["asg"])
            if asg is not None and R["tg"] not in asg["tgs"]:
                self.call("autoscaling", "AttachLoadBalancerTargetGroups", AutoScalingGroupName=R["asg"], TargetGroupARNs=[R["tg"]])

    def tinker(self, code):
        """Plausible but wrong edit (trial-and-error)."""
        R, r = self.res, self.rng
        if code in ("F2", "F3"):
            g = R["app_sg"] if code == "F3" else R["alb_sg"]
            port = int(r.choice([8080, 443, 80]))
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=g,
                      IpPermissions=self._perm(port, cidr="0.0.0.0/0"))
        elif code == "F5":
            self.call("elbv2", "ModifyTargetGroup", TargetGroupArn=R["tg"], HealthCheckPath=str(r.choice(["/", "/index.php", "/healthz", "/health"])))
        elif code == "F1":
            self.call("elbv2", "SetSecurityGroups", LoadBalancerArn=R["lb"], SecurityGroups=[R["alb_sg"]])
        elif code == "F4":
            self.call("autoscaling", "UpdateAutoScalingGroup", AutoScalingGroupName=R["asg"], MinSize=2, DesiredCapacity=2)
        elif code == "F6":
            self.call("iam", "AttachRolePolicy", RoleName="app-role", PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess")
        elif code == "F7":
            self.call("autoscaling", "UpdateAutoScalingGroup", AutoScalingGroupName=R["asg"], MaxSize=4)

    # ------------------------------------------------------------- review
    def review(self):
        R = self.res
        if self.done or self.rng.random() > sig(-0.3 + self.P.get("review_shift", 0.0) + 0.8 * self.theta):
            return
        self.phase = "review"
        v, st = self.truth()
        if v.get("S1") and self.rng.random() < 0.5:
            self.call("ec2", "AuthorizeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, sg=R["alb_sg"]))
            self.call("ec2", "RevokeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(80, cidr="0.0.0.0/0"))
        if v.get("S5") and self.rng.random() < 0.5:
            self.call("ec2", "RevokeSecurityGroupIngress", GroupId=R["app_sg"], IpPermissions=self._perm(22, cidr="0.0.0.0/0"))
        if v.get("S3") and self.rng.random() < 0.5:
            self.call("autoscaling", "UpdateAutoScalingGroup", AutoScalingGroupName=R["asg"], MinSize=2, DesiredCapacity=2)
        if v.get("S4") and self.rng.random() < 0.5:
            for arn in ("arn:aws:iam::aws:policy/AmazonS3FullAccess",):
                self.call("iam", "DetachRolePolicy", RoleName="app-role", PolicyArn=arn)
            self.call("iam", "PutRolePolicy", RoleName="app-role", PolicyName="s3-uploads", PolicyDocument=self._policy(False))
        self.phase = "build"

    # ---------------------------------------------------------------- run
    def run(self):
        self.build()
        ok = self.debug_loop()
        if ok:
            self.review()
        return ok
