import json, os, subprocess, sys, time
ROOT=os.path.dirname(os.path.abspath(__file__))
env=dict(os.environ, PYTHONPATH=ROOT, MOTO_IAM_LOAD_MANAGED_POLICIES="true")
V=json.load(open(os.path.join(ROOT,"revision","variants.json")))
KEEP=["think2","idle","review_hi","console3","weaksig","flatskill"]
CH=25; NV=300
jobs=[("dumps",s,CH,"{}",400) for s in range(0,400,CH)]
for name in KEEP:
    for s in range(0, (300 if name in ("think2", "idle") else 200), CH):  # cohort sizes unchanged
        jobs.append((name,s,CH,json.dumps(V[name]),0))
W=int(sys.argv[1]) if len(sys.argv)>1 else 17
def outp(name,s): return os.path.join(ROOT,"sim","v2" if name!="dumps" else "",name,f"c{s:04d}.jsonl.gz")
orphans=[outp(n,s) for n,s,_,_,_ in jobs if os.path.exists(outp(n,s)+".log") and not os.path.exists(outp(n,s))]
jobs=[j for j in jobs if not os.path.exists(outp(j[0],j[1])) and not os.path.exists(outp(j[0],j[1])+".log")]
run=[]; t0=time.time(); done=0
print("orphans",len(orphans),"todo",len(jobs),flush=True)
while jobs or run or orphans:
    orphans=[o for o in orphans if not os.path.exists(o)]
    while jobs and len(run)+len(orphans)<W:
        name,s,n,ov,keep=jobs.pop(0)
        out=outp(name,s); os.makedirs(os.path.dirname(out),exist_ok=True)
        log=open(out+".log","w")
        p=subprocess.Popen([sys.executable,"-m","cptrace.runner",str(s),str(n),out,str(keep),ov],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        run.append((p,out,log))
    time.sleep(5)
    for x in list(run):
        if x[0].poll() is not None:
            x[2].close(); run.remove(x); done+=1
            print(done, os.path.basename(os.path.dirname(x[1])), os.path.basename(x[1]), x[0].returncode, round(time.time()-t0), flush=True)
print("ALLDONE", round(time.time()-t0), flush=True)
