
"""rq4_run.py - regenerates every RQ4 re-analysis number.
Usage:  python rq4_run.py <npz> all        (calibrate -> reproduction -> 5 Monte Carlo studies -> analysis)
        python rq4_run.py <npz> analyze    (merge existing worker outputs and write rq4_results.json / summary)
Workers: rq4_worker.py (one OS process per shard; multiprocessing.Pool is not usable on this host)."""
import sys, os, json, time, subprocess, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NW = 16
STUDIES = ["rq3B", "rq3A", "h2", "var", "pred"]

def env1():
    e = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): e[k] = "1"
    return e

def launch(npz):
    import rq4_lib as L, rq4_calib
    D = L.load(npz)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cal = rq4_calib.calibrate(D); json.dump(cal, open(os.path.join(HERE, "calib.json"), "w"), indent=1)
    timing = {}
    t = time.time()
    rp = [subprocess.Popen([sys.executable, os.path.join(HERE, "rq4_repro.py"), npz, w, os.path.join(HERE, f"repro_{w}.json")], env=env1())
          for w in ("FS_old", "FS_corrected")]
    for st in STUDIES:
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, os.path.join(HERE, "rq4_worker.py"), npz, os.path.join(HERE, "calib.json"), st,
                                str(w), str(NW), os.path.join(HERE, "out", f"{st}_{w:02d}.json")], env=env1(),
                               stdout=open(os.path.join(HERE, "out", f"{st}_{w:02d}.log"), "w"), stderr=subprocess.STDOUT)
              for w in range(NW)]
        codes = [p.wait() for p in ps]
        timing[st] = {"wall_s": time.time() - t0, "exit_codes": codes}
        json.dump(timing, open(os.path.join(HERE, "timing.json"), "w"), indent=1)
    [p.wait() for p in rp]
    timing["reproduction_wall_s_concurrent"] = time.time() - t
    json.dump(timing, open(os.path.join(HERE, "timing.json"), "w"), indent=1)

if __name__ == "__main__":
    npz, stage = sys.argv[1], sys.argv[2]
    if stage in ("all", "launch"): launch(npz)
    if stage in ("all", "analyze"):
        import rq4_analyze; rq4_analyze.main(npz, HERE)
