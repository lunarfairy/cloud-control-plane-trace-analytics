
"""rq4_worker.py <npz> <calib.json> <study> <wid> <nworkers> <out.json> [reps]
Runs replicates rep = wid, wid+nworkers, ... of one study and writes raw per-replicate results."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq4_lib as L

npz, calib_f, study, wid, nw, outf = sys.argv[1:7]
wid = int(wid); nw = int(nw)
D = L.load(npz); CAL = json.load(open(calib_f))
TH = D["theta"]; NPOP = len(TH)
BLOCKS = {"O1": D["F1"], "snapshot_corrected": D["FS_corrected"]}
STUDY_CODE = {"rq3A": 1, "rq3B": 2, "var": 3, "pred": 4, "h2": 5}[study]
DEF_REPS = {"rq3A": 1000, "rq3B": 1000, "var": 1000, "pred": 500, "h2": 1000}[study]
reps = int(sys.argv[7]) if len(sys.argv) > 7 else DEF_REPS
R = {}
def put(key, row): R.setdefault(key, []).append(row)
fl = lambda a: [float(np.round(x, 7)) for x in np.atleast_1d(a)]
t0 = time.time()

def run_battery(prefix, rep, th, G, P1, P2, C, s, lam):
    for ch, B in BLOCKS.items():
        o = L.battery(th, G, P1, P2, C, B[s])
        for dz in L.DESIGNS:
            v = o[dz]
            row = [rep, v[0]]
            if lam == 0.0: row += fl(v[1])          # block coefficients (standardised block)
            if dz == "oracle_glm": row += [float(v[2]), float(v[3])]
            if dz == "pretest_IV": row += [v[2], v[3]]
            if dz == "latent_MML": row += [float(v[2]), float(v[3]), v[4]]
            put(f"{prefix}|lam{lam}|{ch}|{dz}", row)
        put(f"{prefix}|q|{ch}", [rep, o["q"]])

if study in ("rq3A", "rq3B"):
    LAMS = (0.0, 0.3, 0.5); NS = (100, 200, 400, 700)
    if study == "rq3A":
        fixed = {lam: L.make_outcomes2(TH, D["zg_M1"], lam, seed=101) for lam in LAMS}
    for rep in range(wid, reps, nw):
        for N in NS:
            rc = L.rng_for(STUDY_CODE, N, rep, 0)
            s = rc.choice(NPOP, size=N, replace=False)
            th = TH[s]
            if study == "rq3B":
                G, P1, P2 = L.gen_pre(L.rng_for(STUDY_CODE, N, rep, 1), th)
            for lam in LAMS:
                if study == "rq3B":
                    C = L.gen_post(L.rng_for(STUDY_CODE, N, rep, 2), th, D["zg_M1"][s], lam)
                else:
                    Gf, P1f, P2f, Cf = fixed[lam]; G, P1, P2, C = Gf[s], P1f[s], P2f[s], Cf[s]
                run_battery(f"N{N}", rep, th, G, P1, P2, C, s, lam)

elif study == "var":
    VARS = {"binom": None, "gauss": CAL["gauss"], "binomP_linC": CAL["gauss"], "gaussP_binomC": CAL["gauss"],
            "binom_corr": CAL["binom_corr"], "gauss_corr": CAL["gauss_corr"]}
    for rep in range(wid, reps, nw):
        for N in (200, 400, 700):
            s = L.rng_for(STUDY_CODE, N, rep, 0).choice(NPOP, size=N, replace=False)
            th = TH[s]
            for vn, par in VARS.items():
                G, P1, P2 = L.gen_pre(L.rng_for(STUDY_CODE, N, rep, 1), th, vn, par)
                C = L.gen_post(L.rng_for(STUDY_CODE, N, rep, 2), th, D["zg_M1"][s], 0.0, vn, par)
                for ch, B in BLOCKS.items():
                    o = L.battery(th, G, P1, P2, C, B[s], designs=["pretest_IV", "oracle_linear"])
                    v = o["pretest_IV"]
                    put(f"N{N}|{vn}|{ch}|pretest_IV", [rep, v[0], v[2], v[3]] + fl(v[1]))
                    put(f"N{N}|{vn}|{ch}|oracle_linear", [rep, o["oracle_linear"][0]])

elif study == "pred":
    def cv_pred(y, X, folds):
        yh = np.empty_like(y)
        for f in range(10):
            te = folds == f; tr = ~te
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            yh[te] = X[te] @ b
        sse = ((y - yh) ** 2).sum()
        return 1 - sse / ((y - y.mean()) ** 2).sum(), np.sqrt(sse / len(y))
    for rep in range(wid, reps, nw):
        for N in (200, 400, 700):
            rc = L.rng_for(STUDY_CODE, N, rep, 0)
            s = rc.choice(NPOP, size=N, replace=False)
            folds = rc.permutation(np.arange(N) % 10)
            th = TH[s]
            G, P1, P2 = L.gen_pre(L.rng_for(STUDY_CODE, N, rep, 1), th)
            P = P1 + P2; one = np.ones(N)
            for lam in (0.0, 0.3):
                C = L.gen_post(L.rng_for(STUDY_CODE, N, rep, 2), th, D["zg_M1"][s], lam)
                for bn, bv in (("G", G), ("P", P), ("theta_oracle", th)):
                    r0, e0 = cv_pred(C, np.column_stack([one, bv]), folds)
                    for ch, B in BLOCKS.items():
                        r1, e1 = cv_pred(C, np.column_stack([one, bv, B[s]]), folds)
                        put(f"N{N}|lam{lam}|{bn}|{ch}", [rep, r0, e0, r1, e1])

elif study == "h2":
    H2 = D["H2"]; NRES = H2[:, 0]
    PRED = {"share_rule_H": 1, "share_clfH_prob": 3, "share_true_H": 4}
    LM = [(0.0, "M1")] + [(l, m) for m in ("M1", "M2") for l in (0.15, 0.3, 0.5)]
    for rep in range(wid, reps, nw):
        for N in (200, 400, 700):
            s = L.rng_for(STUDY_CODE, N, rep, 0).choice(NPOP, size=N, replace=False)
            th = TH[s]
            G, P1, P2 = L.gen_pre(L.rng_for(STUDY_CODE, N, rep, 1), th)
            obs = NRES[s] > 0
            put(f"N{N}|neff", [rep, int(obs.sum())])
            Cs = {(lam, m): L.gen_post(L.rng_for(STUDY_CODE, N, rep, 2), th, D["zg_" + m][s], lam) for lam, m in LM}
            for pn, j in PRED.items():
                x = H2[s, j]
                for hd in ("exclude", "impute_indicator"):
                    if hd == "exclude":
                        idx = np.flatnonzero(obs); Xraw = x[idx][:, None]
                    else:
                        idx = np.arange(N); xi = np.where(obs, x, np.nanmean(x))
                        Xraw = np.column_stack([xi, (~obs).astype(float)])
                    Xs, keep = L.standardize(Xraw)
                    if 0 not in keep:
                        continue
                    n = len(idx); one = np.ones(n)
                    t1, t2, t3, pp = th[idx], P1[idx], P2[idx], (P1 + P2)[idx]
                    for (lam, m), Cfull in Cs.items():
                        C = Cfull[idx]
                        pv, b, _ = L.wald_hc1(C, np.column_stack([one, t2, Xs]), np.column_stack([one, t3, Xs]), [2])
                        put(f"N{N}|{pn}|{hd}|lam{lam}|{m}|pretest_IV", [rep, pv, float(b[2])])
                        pv, bg, okg = L.glm_lr(C, t1, Xs, [0])
                        put(f"N{N}|{pn}|{hd}|lam{lam}|{m}|oracle_glm", [rep, pv, float(bg[2]), float(okg)])
                        pv, xl, okl, tr = L.latent_lr(pp, C, Xs, Xs, [0])
                        put(f"N{N}|{pn}|{hd}|lam{lam}|{m}|latent_MML", [rep, pv, float(xl[4 + Xs.shape[1]]), float(okl), tr])

json.dump({"study": study, "wid": wid, "nw": nw, "reps": reps, "runtime_s": time.time() - t0, "R": R}, open(outf, "w"))
