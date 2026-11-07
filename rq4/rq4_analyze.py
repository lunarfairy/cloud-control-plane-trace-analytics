
"""rq4_analyze.py - merges worker outputs, computes all reported quantities, writes rq4_results.json and rq4_summary.md."""
import os, json, glob, time
import numpy as np
from scipy.stats import chi2, ncx2
import rq4_lib as L

def mcse(r, n): return float(np.sqrt(max(r * (1 - r), 0) / n)) if n else None

def merge(here, study):
    R = {}; rt = []; reps = None
    for f in sorted(glob.glob(os.path.join(here, "out", f"{study}_*.json"))):
        j = json.load(open(f)); rt.append(j["runtime_s"]); reps = j["reps"]
        for k, v in j["R"].items(): R.setdefault(k, []).extend(v)
    for k in R: R[k] = sorted(R[k], key=lambda r: r[0])
    return R, {"n_shards": len(rt), "reps": reps, "shard_runtime_s_sum": float(np.sum(rt)), "shard_runtime_s_max": float(np.max(rt))}

def rate(p, a=0.05):
    p = np.asarray(p); n = len(p); r = float(np.mean(p < a)); return r, mcse(r, n), n

def adj_rate(p_alt, p_null):
    crit = float(np.quantile(p_null, 0.05)); r = float(np.mean(np.asarray(p_alt) < crit))
    return r, mcse(r, len(p_alt)), crit

# ---------------------------------------------------------------- population & pseudo-true
def r2(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    return 1 - u @ u / ((y - y.mean()) @ (y - y.mean()))

def population(D, K=200):
    th = D["theta"]; n = len(th); one = np.ones(n)
    blocks = {"O1": D["F1"], "snapshot_corrected": D["FS_corrected"], "snapshot_old": D["FS_old"]}
    out = {}
    G, P1, P2, C = L.make_outcomes2(th, D["zg_M1"], 0.0, 101)
    out["seed101_draw"] = {"corr_theta_G": float(np.corrcoef(th, G)[0, 1]), "corr_theta_P": float(np.corrcoef(th, P1 + P2)[0, 1]),
                           "split_half_r": float(np.corrcoef(P1, P2)[0, 1]), "corr_theta_C_lam0": float(np.corrcoef(th, C)[0, 1])}
    out["corr_zg_theta"] = {"M1_gain_H_help": float(np.corrcoef(D["zg_M1"], th)[0, 1]), "M2_gain_H": float(np.corrcoef(D["zg_M2"], th)[0, 1])}
    out["R2_theta_on_block"] = {bn: float(r2(th, np.column_stack([one, B]))) for bn, B in blocks.items()}
    acc = {}
    stats = {"corr_theta_G": [], "corr_theta_P": [], "split_half_r": []}
    for k in range(K):
        rg = L.rng_for(7, k, 1); G, P1, P2 = L.gen_pre(rg, th); P = P1 + P2
        stats["corr_theta_G"].append(np.corrcoef(th, G)[0, 1]); stats["corr_theta_P"].append(np.corrcoef(th, P)[0, 1])
        stats["split_half_r"].append(np.corrcoef(P1, P2)[0, 1])
        for lam in (0.0, 0.3, 0.5):
            C = L.gen_post(L.rng_for(7, k, 2), th, D["zg_M1"], lam)
            for basen, bv in (("theta", th), ("G", G), ("P", P)):
                X0 = np.column_stack([one, bv]); r0 = r2(C, X0)
                for bn, B in blocks.items():
                    acc.setdefault((lam, basen, bn), []).append((r0, r2(C, np.column_stack([X0, B]))))
    out["mean_over_regenerations"] = {"K": K, **{k: float(np.mean(v)) for k, v in stats.items()},
                                      **{k + "_sd": float(np.std(v)) for k, v in stats.items()}}
    dr = {}
    for (lam, basen, bn), v in acc.items():
        v = np.array(v)
        dr.setdefault(f"lam{lam}", {}).setdefault(basen, {})[bn] = {"R2_baseline": float(v[:, 0].mean()), "R2_baseline_plus_block": float(v[:, 1].mean()),
                                                                     "delta_R2": float((v[:, 1] - v[:, 0]).mean()), "delta_R2_sd_over_regen": float((v[:, 1] - v[:, 0]).std())}
    out["delta_R2_population_N2000_insample"] = dr
    return out

def pseudo_true(D, CAL, K=50):
    """Large-sample (stacked K x 2,000) estimates under lam=0: block coefficients the estimator converges to,
    and the implied asymptotic rejection rate at N via the noncentral chi-square with ncp = N b' (n V)^-1 b."""
    th0 = D["theta"]; out = {}
    T = np.tile(th0, K); n = len(T); one = np.ones(n)
    for vn, par in (("binom", None), ("gauss", CAL["gauss"]), ("binom_corr", CAL["binom_corr"]), ("gauss_corr", CAL["gauss_corr"]),
                    ("binomP_linC", CAL["gauss"]), ("gaussP_binomC", CAL["gauss"])):
        G, P1, P2 = L.gen_pre(L.rng_for(8, 1, K), T, vn, par)
        C = L.gen_post(L.rng_for(8, 2, K), T, np.zeros(n), 0.0, vn, par)
        for bn, B0 in (("O1", D["F1"]), ("snapshot_corrected", D["FS_corrected"])):
            Xs, _ = L.standardize(np.tile(B0, (K, 1))); q = Xs.shape[1]; tc = list(range(2, 2 + q))
            res = {}
            designs = [("pretest_IV", np.column_stack([one, P1, Xs]), np.column_stack([one, P2, Xs]))]
            if vn == "binom":
                designs += [("oracle_linear", np.column_stack([one, T, Xs]), None), ("prior_grade_OLS", np.column_stack([one, G, Xs]), None),
                            ("pretest_OLS", np.column_stack([one, P1 + P2, Xs]), None)]
            for dz, X, Z in designs:
                p, b, V = L.wald_hc1(C, X, X if Z is None else Z, tc)
                bb = b[tc]; Vn = V[np.ix_(tc, tc)] * n
                ncp1 = float(bb @ np.linalg.solve(Vn, bb))
                crit = chi2.ppf(0.95, q)
                res[dz] = {"block_coef_pseudo_true": [float(x) for x in bb], "ncp_per_obs": ncp1,
                           "implied_asymptotic_rejection": {f"N{N}": float(ncx2.sf(crit, q, N * ncp1)) for N in (100, 200, 400, 700)}}
            if vn == "binom":
                xr, fr, okr, _ = L.latent_fit(P1 + P2, C, Xs, Xs[:, []])
                xf, ff, ok, _ = L.latent_fit(P1 + P2, C, Xs, Xs, start=np.r_[xr, np.zeros(q)])
                LRn = 2 * (fr - ff) / n   # per-observation LR ~ ncp per observation
                res["latent_MML"] = {"gamma_pseudo_true": [float(x) for x in xf[4 + q:]], "b_over_d": float(xf[3] / xf[1]), "converged": bool(ok and okr),
                                     "LR_per_obs": float(LRn), "implied_asymptotic_rejection": {f"N{N}": float(ncx2.sf(chi2.ppf(.95, q), q, N * max(LRn, 0))) for N in (100, 200, 400, 700)}}
                bg, llf, okg = L.glm_binom(C, np.column_stack([one, T, Xs]))
                res["oracle_glm"] = {"block_coef_pseudo_true": [float(x) for x in bg[2:]], "b_theta": float(bg[1])}
            out.setdefault(vn, {})[bn] = res
    out["K"] = K; out["n_stacked"] = int(n)
    return out

# ---------------------------------------------------------------- RQ3 summaries
FEAT = None
def rq3_summary(R, info, blocks_names):
    res = {"monte_carlo": info}
    for ch in ("O1", "snapshot_corrected"):
        for dz in L.DESIGNS:
            for N in (100, 200, 400, 700):
                rows0 = R[f"N{N}|lam0.0|{ch}|{dz}"]
                ok0 = np.ones(len(rows0), bool)
                if dz == "latent_MML":
                    okf = {lam: np.array([r[-2] for r in R[f"N{N}|lam{lam}|{ch}|{dz}"]], bool) for lam in (0.0, 0.3, 0.5)}
                elif dz == "oracle_glm":
                    okf = {lam: np.array([r[-1] for r in R[f"N{N}|lam{lam}|{ch}|{dz}"]], bool) for lam in (0.0, 0.3, 0.5)}
                else:
                    okf = {lam: np.ones(len(R[f"N{N}|lam{lam}|{ch}|{dz}"]), bool) for lam in (0.0, 0.3, 0.5)}
                p = {lam: np.array([r[1] for r in R[f"N{N}|lam{lam}|{ch}|{dz}"]])[okf[lam]] for lam in (0.0, 0.3, 0.5)}
                e = {}
                r_, s_, n_ = rate(p[0.0]); e["size"] = r_; e["size_mcse"] = s_; e["n_reps_used"] = n_
                e["n_convergence_failures"] = {f"lam{lam}": int((~okf[lam]).sum()) for lam in okf}
                for lam in (0.3, 0.5):
                    r_, s_, _ = rate(p[lam]); a_, as_, crit = adj_rate(p[lam], p[0.0])
                    e[f"power_raw_lam{lam}"] = r_; e[f"power_raw_lam{lam}_mcse"] = s_
                    e[f"power_size_adj_lam{lam}"] = a_; e[f"power_size_adj_lam{lam}_mcse"] = as_
                    e["size_adj_critical_p"] = crit
                q = len(blocks_names[ch])
                qd_ = {r[0]: r[1] for r in R[f"N{N}|q|{ch}"]}          # q is logged once per lambda; identical across lambda
                qe0 = np.array([qd_[r[0]] for r in rows0])
                full = (qe0 == q) & okf[0.0]
                Bc = np.array([r[2:2 + q] for r, k in zip(rows0, full) if k])
                e["lam0_block_coef_mean"] = dict(zip(blocks_names[ch], [float(x) for x in Bc.mean(0)]))
                e["lam0_block_coef_mcse"] = dict(zip(blocks_names[ch], [float(x) for x in Bc.std(0) / np.sqrt(len(Bc))]))
                e["lam0_block_coef_units"] = "per in-sample SD of feature; outcome points (linear designs) or logit (GLM, latent)"
                okr = [r for r, k in zip(rows0, okf[0.0]) if k]
                if dz == "oracle_glm":
                    bt = np.array([r[-2] for r in okr]); e["lam0_b_theta_mean"] = float(bt.mean()); e["lam0_b_theta_true"] = 0.9; e["lam0_b_theta_mcse"] = float(bt.std() / np.sqrt(len(bt)))
                if dz == "pretest_IV":
                    F = np.array([r[-2] for r in okr]); Fr = np.array([r[-1] for r in okr])
                    e["first_stage_F_conventional_mean"] = float(F.mean()); e["first_stage_F_conventional_p05"] = float(np.quantile(F, .05))
                    e["first_stage_F_HC1_mean"] = float(Fr.mean()); e["first_stage_F_HC1_p05"] = float(np.quantile(Fr, .05))
                if dz == "latent_MML":
                    bd = np.array([r[-3] for r in okr]); e["lam0_b_over_d_mean"] = float(bd.mean()); e["lam0_b_over_d_true"] = 1.0; e["lam0_b_over_d_mcse"] = float(bd.std() / np.sqrt(len(bd)))
                    e["mean_optimizer_runs_lam0"] = float(np.mean([r[-1] for r in okr]))
                e["reps_with_dropped_constant_block_columns"] = int(np.sum(qe0 < q))
                res.setdefault(ch, {}).setdefault(dz, {})[f"N{N}"] = e
    return res

def var_summary(R, info):
    out = {"monte_carlo": info}
    for k, rows in R.items():
        N, vn, ch, dz = k.split("|")
        p = [r[1] for r in rows]; r_, s_, n_ = rate(p)
        e = {"size": r_, "size_mcse": s_, "n_reps": n_}
        if dz == "pretest_IV":
            F = np.array([r[2] for r in rows]); e["first_stage_F_conventional_mean"] = float(F.mean()); e["first_stage_F_conventional_p05"] = float(np.quantile(F, .05))
            ql = max(len(r) for r in rows); B = np.array([r[4:] for r in rows if len(r) == ql]); e["block_coef_mean"] = [float(x) for x in B.mean(0)]
        out.setdefault(vn, {}).setdefault(ch, {}).setdefault(dz, {})[N] = e
    return out

def pred_summary(R, info):
    out = {"monte_carlo": info}
    for k, rows in R.items():
        N, lam, bn, ch = k.split("|")
        A = np.array([r[1:] for r in rows]); n = len(A)
        d = A[:, 2] - A[:, 0]; drm = A[:, 3] - A[:, 1]
        out.setdefault(lam, {}).setdefault(bn, {}).setdefault(ch, {})[N] = {
            "R2cv_baseline_mean": float(A[:, 0].mean()), "R2cv_full_mean": float(A[:, 2].mean()),
            "delta_R2cv_mean": float(d.mean()), "delta_R2cv_mcse": float(d.std(ddof=1) / np.sqrt(n)),
            "share_delta_R2cv_gt0": float(np.mean(d > 0)), "share_mcse": mcse(float(np.mean(d > 0)), n),
            "RMSEcv_baseline_mean": float(A[:, 1].mean()), "RMSEcv_full_mean": float(A[:, 3].mean()),
            "delta_RMSEcv_mean": float(drm.mean()), "delta_RMSEcv_mcse": float(drm.std(ddof=1) / np.sqrt(n)), "n_reps": n}
    return out

def h2_summary(R, info):
    out = {"monte_carlo": info, "effective_N": {}}
    for N in (200, 400, 700):
        ne = np.array([r[1] for r in R[f"N{N}|neff"]])
        out["effective_N"][f"N{N}"] = {"mean": float(ne.mean()), "sd": float(ne.std()), "min": int(ne.min()), "max": int(ne.max())}
    LMS = [(0.15, "M1"), (0.3, "M1"), (0.5, "M1"), (0.15, "M2"), (0.3, "M2"), (0.5, "M2")]
    for pn in ("share_rule_H", "share_clfH_prob", "share_true_H"):
        for hd in ("exclude", "impute_indicator"):
            for meth in ("pretest_IV", "latent_MML", "oracle_glm"):
                blk = {}
                for N in (200, 400, 700):
                    k0 = f"N{N}|{pn}|{hd}|lam0.0|M1|{meth}"
                    if k0 not in R: continue
                    rows0 = R[k0]
                    ok = lambda rows: np.array([r[3] for r in rows], bool) if meth != "pretest_IV" else np.ones(len(rows), bool)
                    ok0 = ok(rows0); p0 = np.array([r[1] for r in rows0])[ok0]
                    r_, s_, n_ = rate(p0)
                    e = {"size": r_, "size_mcse": s_, "n_reps_used": n_, "n_convergence_failures_lam0": int((~ok0).sum()),
                         "lam0_coef_mean": float(np.mean([r[2] for r in rows0])), "lam0_coef_mcse": float(np.std([r[2] for r in rows0]) / np.sqrt(len(rows0)))}
                    for lam, m in LMS:
                        rows = R[f"N{N}|{pn}|{hd}|lam{lam}|{m}|{meth}"]; okk = ok(rows)
                        pp = np.array([r[1] for r in rows])[okk]
                        a, sa, _ = rate(pp); b, sb, crit = adj_rate(pp, p0)
                        e[f"{m}|lam{lam}"] = {"power_raw": a, "power_raw_mcse": sa, "power_size_adj": b, "power_size_adj_mcse": sb,
                                             "coef_mean": float(np.mean([r[2] for r in rows])), "n_convergence_failures": int((~okk).sum())}
                    e["size_adj_critical_p"] = crit
                    blk[f"N{N}"] = e
                # smallest N reaching 0.80 size-adjusted power
                sm = {}
                for lam, m in LMS:
                    hit = [N for N in (200, 400, 700) if f"N{N}" in blk and blk[f"N{N}"][f"{m}|lam{lam}"]["power_size_adj"] >= 0.80]
                    sm[f"{m}|lam{lam}"] = f"N{min(hit)}" if hit else "not reached (N<=700)"
                blk["smallest_N_size_adj_power_ge_0.80"] = sm
                out.setdefault(pn, {}).setdefault(hd, {})[meth] = blk
    return out

def repro_summary(here):
    out = {}
    for w in ("FS_old", "FS_corrected"):
        f = os.path.join(here, f"repro_{w}.json")
        if os.path.exists(f): out["snapshot_block=" + w] = json.load(open(f))
    tgt = {"O1_pretest_IV_lam0": [0.140, 0.105, 0.113, 0.140], "snapshot_pretest_IV_lam0_N700": 0.215,
           "O1_pretest_IV_lam0.3_approx": [0.28, 0.58, 0.94, 1.0], "prior_grade_OLS_lam0": "~0.99-1.0"}
    o = out["snapshot_block=FS_old"]
    got = {"O1_pretest_IV_lam0": list(o["lam0.0"]["pretest_IV|O1"].values()), "snapshot_pretest_IV_lam0_N700": o["lam0.0"]["pretest_IV|snapshot60"]["N700"],
           "O1_pretest_IV_lam0.3_approx": list(o["lam0.3"]["pretest_IV|O1"].values()), "prior_grade_OLS_lam0": list(o["lam0.0"]["prior_grade_OLS|O1"].values())}
    out["check"] = {"targets": tgt, "reproduced": got,
                    "exact_match_O1_IV_lam0": [round(x, 3) for x in got["O1_pretest_IV_lam0"]] == tgt["O1_pretest_IV_lam0"],
                    "exact_match_snapshot_IV_N700": round(got["snapshot_pretest_IV_lam0_N700"], 3) == 0.215}
    return out

def main(npz, here):
    t0 = time.time()
    D = L.load(npz); CAL = json.load(open(os.path.join(here, "calib.json")))
    names = {"O1": [str(x) for x in D["F1_names"]], "snapshot_corrected": [str(x) for x in D["FS_names"]]}
    res = {"meta": {"inputs": os.path.basename(npz), "n_learners": int(len(D["theta"])), "block_features": names, "alpha": 0.05,
                    "base_seed": L.BASE_SEED, "gauss_hermite_nodes": L.NQ}}
    res["reproduction_original_designA"] = repro_summary(here)
    res["calibration_section4"] = CAL
    res["population"] = population(D)
    res["pseudo_true_lam0"] = pseudo_true(D, CAL)
    for st, key in (("rq3B", "explanatory_designB_primary"), ("rq3A", "explanatory_designA_fixed_corpus")):
        R, info = merge(here, st); res[key] = rq3_summary(R, info, names)
    R, info = merge(here, "var"); res["iv_inflation_sources"] = var_summary(R, info)
    R, info = merge(here, "pred"); res["predictive_designB"] = pred_summary(R, info)
    R, info = merge(here, "h2"); res["h2_exact_field_hypothesis"] = h2_summary(R, info)
    tf = os.path.join(here, "timing.json")
    res["runtime"] = {"launch": json.load(open(tf)) if os.path.exists(tf) else None, "analysis_s": time.time() - t0}
    json.dump(res, open(os.path.join(here, "rq4_results.json"), "w"), indent=1)
    import rq4_report; rq4_report.write_summary(res, os.path.join(here, "rq4_summary.md")); rq4_report.write_methods(res, os.path.join(here, "rq4_methods.md"))
    return res
