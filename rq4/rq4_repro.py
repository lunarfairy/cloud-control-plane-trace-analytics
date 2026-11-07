
import numpy as np, json, sys, time
from scipy.stats import chi2
NPZ=sys.argv[1]; which=sys.argv[2]; out=sys.argv[3]
d=np.load(NPZ,allow_pickle=True)
def _wald_hc1(y, X, Z, test_cols):
    n, k = X.shape
    PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    Xh = PZ @ X
    A_ = np.linalg.pinv(Xh.T @ Xh)
    b = A_ @ Xh.T @ y
    u = y - X @ b
    meat = (Xh * u[:, None] ** 2).T @ Xh
    V = A_ @ meat @ A_ * n / (n - k)
    bb = b[test_cols]
    W = float(bb @ np.linalg.pinv(V[np.ix_(test_cols, test_cols)]) @ bb)
    return float(chi2.sf(W, len(test_cols)))
def make_outcomes2(th, gain, lam, seed):
    rng = np.random.default_rng(seed)
    zg = (gain - gain.mean()) / gain.std()
    G = 0.6 * th + 0.8 * rng.normal(size=len(th))
    lp = 0.2 + 0.9 * th
    P1 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    P2 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    C = rng.binomial(30, 1 / (1 + np.exp(-(0.4 + 0.9 * th + lam * zg)))).astype(float)
    return th, G, P1, P2, C, zg
t0=time.time()
F1=d["F1"]; FS=d[which]; th0=d["theta"]; gain=d["gain_H_help"]
lams=(0.0,0.3,0.5); Ns=(100,200,400,700); reps=1000
outd={}
rng=np.random.default_rng(3)
for lam in lams:
    th, G, P1, P2, C, zg = make_outcomes2(th0, gain, lam, seed=101)
    P=P1+P2
    res={"corr_theta_G": float(np.corrcoef(th, G)[0, 1]), "corr_theta_P": float(np.corrcoef(th, P)[0, 1]),
         "split_half_r": float(np.corrcoef(P1,P2)[0,1]), "corr_gain_theta": float(np.corrcoef(zg, th)[0, 1])}
    for chname, X1 in (("O1", F1), ("snapshot60", FS)):
        q = X1.shape[1]
        for design in ("prior_grade_OLS", "pretest_OLS", "pretest_IV"):
            pw = {}
            for N in Ns:
                rej = 0
                for _ in range(reps):
                    s = rng.choice(len(C), size=N, replace=False)
                    one = np.ones((N, 1))
                    if design == "prior_grade_OLS":
                        X = np.column_stack([one, G[s], X1[s]]); Z = X
                    elif design == "pretest_OLS":
                        X = np.column_stack([one, P[s], X1[s]]); Z = X
                    else:
                        X = np.column_stack([one, P1[s], X1[s]]); Z = np.column_stack([one, P2[s], X1[s]])
                    pv = _wald_hc1(C[s], X, Z, list(range(2, 2 + q)))
                    rej += pv < 0.05
                pw[f"N{N}"] = rej / reps
            res[f"{design}|{chname}"] = pw
    outd[f"lam{lam}"] = res
outd["runtime_s"]=time.time()-t0
json.dump(outd, open(out,"w"), indent=1)
