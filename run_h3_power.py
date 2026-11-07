"""Planning simulation for field hypothesis H3 -> results/h3_power.json.
An online, label-free activity + state HMM is trained on half of the simulated learners; its test-half scores are
rank-normalised and degraded with Gaussian noise (sd 0.95) to a population AUROC of about .80 (the planning value).
For n think-aloud learners (10, 15, 20) we draw 200 samples of n learners and compute the learner-clustered
bootstrap 95% interval (200 resamples); power = share of samples whose lower bound exceeds .70."""
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from hmmlearn.hmm import GaussianHMM  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cptrace import study2 as S2  # noqa: E402

recs = []
for fn in sorted(glob.glob(os.path.join(ROOT, "sim", "main", "shard*.jsonl.gz"))):
    with gzip.open(fn, "rt", encoding="utf8") as f:
        recs += [json.loads(line) for line in f]
rng = np.random.default_rng(11)
seqs = [s for s in (S2.bin_sequences2(r) for r in recs) if s is not None and len(s["y_debug"]) >= 3]
idx = rng.permutation(len(seqs))
tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
Xtr = np.concatenate([seqs[i]["X"] for i in tr])
ltr = [len(seqs[i]["y_debug"]) for i in tr]
m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=11, min_covar=1e-3).fit(Xtr, ltr)
rank = np.argsort(np.argsort(m.means_[:, 4])) / 2
sc = {i: S2.forward_filter(m, seqs[i]["X"]) @ rank for i in te}
yy = {i: seqs[i]["y_debug"] for i in te}
allp = np.concatenate([sc[i] for i in te])
ally = np.concatenate([yy[i] for i in te])
z = norm.ppf((rankdata(allp) - 0.5) / len(allp))
rg = np.random.default_rng(3)
zn = z + rg.normal(0, 0.95, len(z))
pos, scn = 0, {}
for i in te:
    n = len(yy[i])
    scn[i] = zn[pos:pos + n]
    pos += n
te_l = list(te)
H3 = {"population_auroc": float(roc_auc_score(ally, zn)), "raw_auroc": float(roc_auc_score(ally, allp))}
for n in (10, 15, 20):
    lcb, wid, est = [], [], []
    for s in range(200):
        pick = rg.choice(te_l, n, replace=False)
        yv = np.concatenate([yy[i] for i in pick]); pv = np.concatenate([scn[i] for i in pick])
        if yv.min() == yv.max():
            continue
        est.append(roc_auc_score(yv, pv))
        bs = []
        for b in range(200):
            pb = rg.choice(len(pick), n, replace=True)
            yb = np.concatenate([yy[pick[k]] for k in pb]); sb = np.concatenate([scn[pick[k]] for k in pb])
            if yb.min() != yb.max():
                bs.append(roc_auc_score(yb, sb))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lcb.append(lo); wid.append(hi - lo)
    H3[str(n)] = {"power_lcb_gt_070": float(np.mean(np.array(lcb) > 0.70)), "median_ci_width": float(np.median(wid)), "median_auc": float(np.median(est))}
    print(n, H3[str(n)], flush=True)
H3["samples"], H3["boot"] = 200, 200
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "h3_power.json")
json.dump(H3, open(out, "w"), indent=1)
