
"""rq4_report.py - compact markdown summary tables from rq4_results.json."""
def f3(x, s=None):
    if x is None: return "-"
    return f"{x:.3f}" if s is None else f"{x:.3f} ({s:.3f})"

DN = {"oracle_linear": "ORACLE-linear (OLS, theta)", "oracle_glm": "ORACLE-GLM (logit LR, theta)", "prior_grade_OLS": "prior_grade_OLS",
      "pretest_OLS": "pretest_OLS", "pretest_IV": "pretest_IV (2SLS)", "latent_MML": "LATENT MIMIC (MML LR)"}

def write_summary(res, path):
    L = []
    w = L.append
    w("# RQ4 re-analysis: compact summary\n")
    w("Entries are rejection rates at alpha = .05 with Monte Carlo SE in parentheses. Size-adjusted power uses the empirical 5% quantile of the same design's lambda = 0 p-values at the same N as the critical value. All values are read from rq4_results.json.\n")
    rc = res["reproduction_original_designA"]
    w("## 1. Reproduction of the original A7 analysis (Design A, outcome seed 101, cohort RNG seed 3, 1,000 reps)\n")
    w("| Quantity | Original paper | Reproduced (FS_old) | Rerun with FS_corrected |\n|---|---|---|---|")
    o, c = rc["snapshot_block=FS_old"], rc["snapshot_block=FS_corrected"]
    j = lambda d: "/".join(f"{v:.3f}" for v in d.values())
    w(f"| O1 pretest_IV, lam=0, N=100/200/400/700 | .140/.105/.113/.140 | {j(o['lam0.0']['pretest_IV|O1'])} | {j(c['lam0.0']['pretest_IV|O1'])} |")
    w(f"| Snapshot pretest_IV, lam=0, N=100/200/400/700 | N700 = .215 | {j(o['lam0.0']['pretest_IV|snapshot60'])} | {j(c['lam0.0']['pretest_IV|snapshot60'])} |")
    w(f"| O1 pretest_IV, lam=0.3 | ~.28/.58/.94/1.0 | {j(o['lam0.3']['pretest_IV|O1'])} | {j(c['lam0.3']['pretest_IV|O1'])} |")
    w(f"| O1 prior_grade_OLS, lam=0 | ~.99-1.0 | {j(o['lam0.0']['prior_grade_OLS|O1'])} | {j(c['lam0.0']['prior_grade_OLS|O1'])} |")
    w(f"| Snapshot prior_grade_OLS, lam=0 | - | {j(o['lam0.0']['prior_grade_OLS|snapshot60'])} | {j(c['lam0.0']['prior_grade_OLS|snapshot60'])} |")
    w(f"| Snapshot pretest_OLS, lam=0 | - | {j(o['lam0.0']['pretest_OLS|snapshot60'])} | {j(c['lam0.0']['pretest_OLS|snapshot60'])} |")
    w("")
    for key, title in (("explanatory_designB_primary", "Design B (primary: fresh cohort and fresh G, P1, P2, C per replicate)"),
                       ("explanatory_designA_fixed_corpus", "Design A (outcomes generated once, cohorts subsampled; new seeds)")):
        E = res[key]
        w(f"## 2. Explanatory null (block adds nothing given theta): {title}\n")
        for ch in ("O1", "snapshot_corrected"):
            w(f"### Block: {ch}\n")
            w("| Design | Size N=100 | N=200 | N=400 | N=700 | Size-adj. power lam=0.3 N=100/200/400/700 | Raw power lam=0.3 | Size-adj. power lam=0.5 |\n|---|---|---|---|---|---|---|---|")
            for dz in DN:
                e = E[ch][dz]
                sz = " | ".join(f3(e[f"N{N}"]["size"], e[f"N{N}"]["size_mcse"]) for N in (100, 200, 400, 700))
                pa = "/".join(f"{e[f'N{N}']['power_size_adj_lam0.3']:.3f}" for N in (100, 200, 400, 700))
                pr = "/".join(f"{e[f'N{N}']['power_raw_lam0.3']:.3f}" for N in (100, 200, 400, 700))
                p5 = "/".join(f"{e[f'N{N}']['power_size_adj_lam0.5']:.3f}" for N in (100, 200, 400, 700))
                w(f"| {DN[dz]} | {sz} | {pa} | {pr} | {p5} |")
            iv = E[ch]["pretest_IV"]
            w("\nFirst-stage F (pretest_IV, lam=0), mean [5th percentile]: " + "; ".join(
                f"N={N}: {iv[f'N{N}']['first_stage_F_conventional_mean']:.1f} [{iv[f'N{N}']['first_stage_F_conventional_p05']:.1f}]" for N in (100, 200, 400, 700)))
            lat = E[ch]["latent_MML"]
            w("\nLatent-model convergence failures (lam=0/0.3/0.5, summed over N): " + "/".join(
                str(sum(lat[f"N{N}"]["n_convergence_failures"][f"lam{l}"] for N in (100, 200, 400, 700))) for l in (0.0, 0.3, 0.5)) + "\n")
    V = res["iv_inflation_sources"]
    w("## 3. Source of pretest_IV size inflation (Design B, lam = 0, 1,000 reps)\n")
    w("| Measurement variant | Block | IV size N=200 | N=400 | N=700 | ORACLE-linear size N=700 |\n|---|---|---|---|---|---|")
    lab = {"binom": "Binomial P and C (main DGP)", "gauss": "(i) Linear-Gaussian P and C, reliability matched",
           "binomP_linC": "Binomial P, linear-Gaussian C", "gaussP_binomC": "Gaussian P, binomial C",
           "binom_corr": "(ii) Binomial, correlated pre-test errors rho=0.2", "gauss_corr": "Linear-Gaussian, correlated pre-test errors rho=0.2"}
    for vn in lab:
        for ch in ("O1", "snapshot_corrected"):
            e = V[vn][ch]["pretest_IV"]
            w(f"| {lab[vn]} | {ch} | " + " | ".join(f3(e[f"N{N}"]["size"], e[f"N{N}"]["size_mcse"]) for N in (200, 400, 700)) +
              f" | {f3(V[vn][ch]['oracle_linear']['N700']['size'])} |")
    pt = res["pseudo_true_lam0"]
    w("\nLarge-sample implied IV rejection rate at lam=0 (noncentral chi-square from the pseudo-true block coefficients, stacked n = "
      f"{pt['n_stacked']}): " + "; ".join(f"{vn}/{ch}: " + "/".join(f"{pt[vn][ch]['pretest_IV']['implied_asymptotic_rejection'][f'N{N}']:.3f}" for N in (200, 400, 700))
                                          for vn in ("binom", "gauss", "binom_corr") for ch in ("O1", "snapshot_corrected")) + " (N=200/400/700).\n")
    P = res["predictive_designB"]
    w("## 4. Predictive increment (Design B, 10-fold CV within cohort, 500 reps)\n")
    w("| lam | Baseline | Block | N | mean R2cv baseline | mean delta-R2cv (MCSE) | share delta>0 | mean delta-RMSEcv |\n|---|---|---|---|---|---|---|---|")
    for lam in ("lam0.0", "lam0.3"):
        for bn in ("G", "P", "theta_oracle"):
            for ch in ("O1",):
                for N in ("N200", "N400", "N700"):
                    e = P[lam][bn][ch][N]
                    w(f"| {lam[3:]} | {bn} | {ch} | {N[1:]} | {e['R2cv_baseline_mean']:.3f} | {e['delta_R2cv_mean']:.4f} ({e['delta_R2cv_mcse']:.4f}) | {e['share_delta_R2cv_gt0']:.3f} | {e['delta_RMSEcv_mean']:.4f} |")
    H = res["h2_exact_field_hypothesis"]
    w("\n## 5. Exact field hypothesis H2 (1-df test of the week-1 hypothesis-driven share; Design B, 1,000 reps)\n")
    w("Effective N after excluding learners with no resolved week-1 episode: " + "; ".join(
        f"N={N[1:]}: mean {v['mean']:.1f} (min {v['min']}, max {v['max']})" for N, v in H["effective_N"].items()) + "\n")
    for hd in ("exclude", "impute_indicator"):
        w(f"### Handling: {hd}\n")
        w("| Predictor | Method | Size N=200/400/700 | Mech | Size-adj. power lam=0.15 | lam=0.3 | lam=0.5 | Smallest N with adj. power >= .80 (lam .15/.3/.5) |\n|---|---|---|---|---|---|---|---|")
        for pn in ("share_rule_H", "share_clfH_prob", "share_true_H"):
            for meth in ("pretest_IV", "latent_MML", "oracle_glm"):
                b = H[pn][hd][meth]
                sz = "/".join(f"{b[f'N{N}']['size']:.3f}" for N in (200, 400, 700))
                for m in ("M1", "M2"):
                    cols = ["/".join(f"{b[f'N{N}'][f'{m}|lam{l}']['power_size_adj']:.3f}" for N in (200, 400, 700)) for l in (0.15, 0.3, 0.5)]
                    sm = "/".join(b["smallest_N_size_adj_power_ge_0.80"][f"{m}|lam{l}"].replace(" (N<=700)", "") for l in (0.15, 0.3, 0.5))
                    w(f"| {pn} | {meth} | {sz if m == 'M1' else ''} | {m} | " + " | ".join(cols) + f" | {sm} |")
        w("")
    pop = res["population"]
    w("## 6. Population quantities (all 2,000 learners)\n")
    mg = pop["mean_over_regenerations"]
    w(f"corr(theta,G) = {mg['corr_theta_G']:.3f}; corr(theta,P) = {mg['corr_theta_P']:.3f}; split-half r = {mg['split_half_r']:.3f} (means over {mg['K']} regenerations); "
      f"corr(zg,theta) = {pop['corr_zg_theta']['M1_gain_H_help']:.3f} (M1), {pop['corr_zg_theta']['M2_gain_H']:.3f} (M2). R2(theta | block): " +
      ", ".join(f"{k} {v:.3f}" for k, v in pop["R2_theta_on_block"].items()) + ".\n")
    w("| lam | Baseline | delta-R2 O1 | delta-R2 snapshot_corrected | delta-R2 snapshot_old |\n|---|---|---|---|---|")
    for lam in ("lam0.0", "lam0.3", "lam0.5"):
        for bn in ("theta", "G", "P"):
            e = pop["delta_R2_population_N2000_insample"][lam][bn]
            w(f"| {lam[3:]} | {bn} | {e['O1']['delta_R2']:.4f} | {e['snapshot_corrected']['delta_R2']:.4f} | {e['snapshot_old']['delta_R2']:.4f} |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_methods(res, path):
    cal = res["calibration_section4"]; rt = res["runtime"]["launch"] or {}
    def fails(key):
        tot = 0
        for ch in ("O1", "snapshot_corrected"):
            for N in (100, 200, 400, 700):
                tot += sum(res[key][ch]["latent_MML"][f"N{N}"]["n_convergence_failures"].values())
        return tot
    def h2fails():
        tot = 0
        H = res["h2_exact_field_hypothesis"]
        for pn in ("share_rule_H", "share_clfH_prob", "share_true_H"):
            for hd in ("exclude", "impute_indicator"):
                for meth in ("latent_MML", "oracle_glm"):
                    for N in (200, 400, 700):
                        e = H[pn][hd][meth][f"N{N}"]
                        tot += e["n_convergence_failures_lam0"] + sum(v["n_convergence_failures"] for k, v in e.items() if isinstance(v, dict))
        return tot
    wall = {k: v["wall_s"] for k, v in rt.items() if isinstance(v, dict) and "wall_s" in v}
    T = f"""# RQ4 re-analysis: methods

All numbers quoted in the companion files come from `rq4_run.py` (orchestrator), `rq4_worker.py` (Monte Carlo shards), `rq4_lib.py` (estimators and data-generating mechanism), `rq4_calib.py`, `rq4_repro.py` (verbatim original A7 logic), `rq4_analyze.py` and `rq4_report.py`. The command is `python rq4_run.py <rq4_inputs.npz> all`.

## 1. Estimands (stated separately)

1. **Explanatory null (the RQ4 hypothesis as re-specified).** $H_0^{{expl}}$: the trace block $X$ carries no information about the post-test beyond latent skill, $C \\perp X \\mid \\theta$. In the data-generating mechanism (DGP), $H_0^{{expl}}$ holds exactly when $\\lambda = 0$. Under $\\lambda>0$ the outcome depends on $z_g$, which is not itself in $X$. A test of the block coefficients $\\gamma$ then targets the projection of $\\lambda z_g$ on $X$ given $\\theta$, and "power" means detecting that the block carries outcome information beyond $\\theta$. It does not mean recovering $\\lambda$.
2. **Predictive increment.** $\\Delta R^2_{{cv}} = R^2_{{cv}}(\\text{{baseline}} + X) - R^2_{{cv}}(\\text{{baseline}})$, with baseline G (prior grade) or P (pre-test total). This quantity can be positive under $\\lambda=0$, because $X$ predicts $\\theta$ and the baseline measures $\\theta$ with error. A positive $\\Delta R^2_{{cv}}$ is therefore not evidence against $H_0^{{expl}}$.
3. **Causal effect** of hypothesis-driven repair behaviour on the post-test: **not identified** in this design. $\\lambda$ is a coefficient in a simulated outcome model applied to an observational behavioural count ($z_g$, corr. with $\\theta$ = {res['population']['corr_zg_theta']['M1_gain_H_help']:.3f}). Identifying a causal effect from real traces would need an assignment mechanism, for example randomised scaffolding, or a defensible no-unmeasured-confounding argument beyond $\\theta$. Neither is available. None of the analyses below estimates a causal effect.

## 2. Data-generating mechanism (unchanged from the original paper)

For learner $i$ (2,000 simulated learners; $\\theta_i$, trace features and gain counts read from `rq4_inputs.npz`):
$z_{{g,i}} = (g_i - \\bar g)/\\mathrm{{sd}}(g)$ with $g$ = `gain_H_help` (mechanism M1) or `gain_H` (M2, H2 only); standardised over all 2,000 learners.

$G_i = 0.6\\,\\theta_i + 0.8\\,\\varepsilon_i$, $\\varepsilon_i \\sim N(0,1)$;
$P_{{1i}}, P_{{2i}} \\overset{{iid}}{{\\sim}} \\mathrm{{Bin}}(15, \\mathrm{{logit}}^{{-1}}(0.2 + 0.9\\theta_i))$, $P_i = P_{{1i}} + P_{{2i}}$;
$C_i \\sim \\mathrm{{Bin}}(30, \\mathrm{{logit}}^{{-1}}(0.4 + 0.9\\theta_i + \\lambda z_{{g,i}}))$.

Blocks: O1 = 6 event-log features (`F1`); snapshot = 3 features computed from episodes detected by hourly polling (`FS_corrected`). `FS_old` (computed from true episodes; identical to O1 columns 1, 2, 6) is used only in the reproduction check. Each block is centred and scaled within the analysed sample. All tests are invariant to this, and constant columns are dropped with the df reduced. This occurred in {sum(res['explanatory_designB_primary']['snapshot_corrected'][dz][f'N{N}']['reps_with_dropped_constant_block_columns'] for dz in ['oracle_glm'] for N in (100,200,400,700))} Design B snapshot cohorts.

## 3. Replication structures

* **Design A (fixed finite corpus; original).** G, P1, P2, C are generated once for all 2,000 learners with `numpy.random.default_rng(101)` (in the call order G, P1, P2, C; same seed for each $\\lambda$). Each replicate draws a cohort of N learners without replacement. In the reproduction check, cohorts come from the single `default_rng(3)` stream, consumed in the original loop order. In the new Design A runs, cohorts come from `SeedSequence([{res['meta']['base_seed']}, 1, N, rep, 0])`.
* **Design B (primary).** Each replicate draws a cohort of N learners without replacement (`SeedSequence([{res['meta']['base_seed']}, study, N, rep, 0])`). It then regenerates G, P1, P2 for that cohort (`...,1`) and C (`...,2`). The C stream is reused across $\\lambda$ (and across M1/M2), so outcomes under different $\\lambda$ are coupled by common random numbers. All designs are applied to the same simulated cohort. Study codes: 2 = explanatory, 3 = IV-inflation variants, 4 = predictive, 5 = H2.
* Replicates: 1,000 per cell for every explanatory design, **including the latent MML model** (no reduction to 500 was needed), the IV-variant study and H2; 500 for the predictive study (as specified).
* Monte Carlo SE of a rejection rate $\\hat r$ from $R$ replicates: $\\sqrt{{\\hat r(1-\\hat r)/R}}$. For size-adjusted power this binomial SE ignores the sampling error of the estimated critical value, so it is slightly optimistic.
* **Size-adjusted power:** $c_N$ = empirical 5% quantile of the design's $\\lambda=0$ p-values at the same N (same block, same handling); power = $\\Pr(p < c_N)$ under $\\lambda>0$.

## 4. Tests of the trace block (joint, df = block size q: 6 for O1, 3 for snapshot)

a. **ORACLE-linear:** OLS $C = \\beta_0 + \\beta_1\\theta + X\\gamma + u$; Wald $W = \\hat\\gamma' \\hat V_{{\\gamma}}^{{-1}} \\hat\\gamma \\sim \\chi^2_q$ with HC1 $\\hat V = (X'X)^{{-1}} (\\sum \\hat u_i^2 x_i x_i') (X'X)^{{-1}} \\cdot n/(n-k)$.
b. **ORACLE-GLM:** binomial-logit GLM $\\mathrm{{logit}}\\,\\pi_i = \\beta_0 + \\beta_1 \\theta_i + x_i'\\gamma$, $C_i \\sim \\mathrm{{Bin}}(30, \\pi_i)$, fitted by Newton/IRLS. LR $= 2(\\ell_1 - \\ell_0) \\sim \\chi^2_q$. This model is correctly specified under $\\lambda=0$.
c. **prior_grade_OLS:** as (a) with G in place of $\\theta$.
d. **pretest_OLS:** as (a) with P in place of $\\theta$.
e. **pretest_IV:** 2SLS of $C = \\beta_0 + \\beta_1 P_1 + X\\gamma + u$ with instruments $Z = (1, P_2, X)$ ($X$ exogenous), $\\hat X = Z(Z'Z)^{{-1}}Z'X$. HC1 sandwich $(\\hat X'\\hat X)^{{-1}}(\\sum \\hat u_i^2 \\hat x_i \\hat x_i')(\\hat X'\\hat X)^{{-1}} n/(n-k)$ with $\\hat u = C - X\\hat b$. Wald $\\chi^2_q$. This is the original estimator: identical algebra, without forming the $N\\times N$ projection. First stage: $P_1$ on $(1, P_2, X)$; F for $P_2$ reported both conventional ($t^2$) and HC1-robust.
f. **LATENT (MIMIC) marginal ML:**
$\\theta_i \\mid x_i \\sim N(x_i'\\delta, 1)$; $P_{{1i}}, P_{{2i}} \\mid \\theta_i \\overset{{iid}}{{\\sim}} \\mathrm{{Bin}}(15, \\mathrm{{logit}}^{{-1}}(c + d\\theta_i))$; $C_i \\mid \\theta_i, x_i \\sim \\mathrm{{Bin}}(30, \\mathrm{{logit}}^{{-1}}(a + b\\theta_i + x_i'\\gamma))$. Measurements are conditionally independent given $\\theta$.
Identification: because $x$ is centred, the intercept of $\\theta \\mid x$ is fixed at 0 and the residual variance at 1; $c, d, a, b$ are free. Because the two halves share item parameters, the pre-test kernel is $P_i\\eta^P - 30\\log(1+e^{{\\eta^P}})$ with $P = P_1+P_2$; this is exactly the product of the two Bin(15) likelihoods up to a constant.
Marginal likelihood: $\\ell = \\sum_i \\log \\sum_{{k=1}}^{{{res['meta']['gauss_hermite_nodes']}}} w_k\\, f(P_i, C_i \\mid \\theta = x_i'\\delta + t_k)$, with probabilists' Gauss–Hermite nodes $t_k$ and normalised weights $w_k$ ({res['meta']['gauss_hermite_nodes']} nodes). The quadrature was checked against 61 nodes in a pilot cohort: LR p-values agreed to about $10^{{-5}}$ (computed with 21 nodes; 31 used).
LR test of $\\gamma = 0$ (df = q). $\\delta$ is free in both models, so the block may predict skill.
Optimisation: BFGS with the analytic gradient (checked against central differences, max abs diff $1.8\\times10^{{-6}}$). Start values: $c$, $a$ = empirical logits of the mean scores, $d = b = 0.9$, $\\delta = \\gamma = 0$. The full model starts from the restricted MLE with $\\gamma = 0$, so LR $\\ge 0$ up to tolerance.
Convergence: a fit is accepted when the max abs gradient is below $10^{{-3}}$. Otherwise it is retried with L-BFGS-B from the current point, then BFGS from $d = b = 0.5$. A replicate is flagged as failed if either fit stays unconverged or LR $< -10^{{-4}}$; failed replicates are excluded from the rate denominators and counted.
Observed failures: Design B {fails('explanatory_designB_primary')}, Design A {fails('explanatory_designA_fixed_corpus')} (out of 24,000 replicate-cells each); H2 latent/GLM {h2fails()}.

Coefficient bias at $\\lambda=0$ is reported as the MC mean of the standardised block coefficients (per in-sample SD; outcome points for linear designs, logit units for GLM and latent). The true value of $\\gamma$ is 0 for the GLM and latent models. Scale-free checks: GLM $\\hat\\beta_1$ vs 0.9, and latent $\\hat b/\\hat d$ vs 1. For OLS/2SLS designs the block coefficient has no causal or structural target; its mean is a bias relative to the explanatory null of 0. Large-sample pseudo-true values come from a stacked sample of {res['pseudo_true_lam0']['K']} regenerations of all 2,000 learners (n = {res['pseudo_true_lam0']['n_stacked']}). The implied asymptotic rejection rate at N uses a noncentral $\\chi^2_q$ with ncp $= N\\,\\hat\\gamma'(n\\hat V)^{{-1}}\\hat\\gamma$, or per-observation LR $\\times N$ for the latent model.

## 5. Isolating the residual IV inflation (Design B, $\\lambda=0$, N = 200, 400, 700, 1,000 reps)

Calibration ({cal['K']} pooled regenerations, seed {cal['seed']}): binomial half-test reliability $r = {cal['binom_half_reliability']:.4f}$; post-test $R^2(C\\mid\\theta) = {cal['binom_post_R2_theta_lam0']:.4f}$; var($\\theta$) = {cal['var_theta']:.4f}; var(C) = {cal['var_C_binom_lam0']:.3f}.

* (i) **Linear-Gaussian:** $P_j = \\theta + e_j$, $e_j \\sim N(0, \\sigma_e^2)$, $\\sigma_e = \\sqrt{{\\mathrm{{var}}(\\theta)(1-r)/r}} = {cal['gauss']['se_half']:.4f}$ (simulated half r = {cal['gauss']['check_half_r']:.4f}). $C = s(0.9\\theta + \\lambda z_g) + e_C$ with $s = {cal['gauss']['s']:.4f}$ and $\\mathrm{{sd}}(e_C) = {cal['gauss']['se_post']:.4f}$, matching both $R^2(C\\mid\\theta)$ and var(C) of the binomial post-test. Two hybrids separate the parts: binomial P with linear-Gaussian C, and Gaussian P with binomial C.
* (ii) **Correlated pre-test errors:** $P_j \\sim \\mathrm{{Bin}}(15, \\mathrm{{logit}}^{{-1}}(0.2 + 0.9\\theta + \\eta))$ with a shared $\\eta \\sim N(0, \\tau^2)$, $\\tau = {cal['binom_corr']['tau']:.4f}$. $\\tau$ is chosen by quadrature so that corr$(P_1 - E[P_1\\mid\\theta], P_2 - E[P_2\\mid\\theta]) = 0.2$ (simulated {cal['binom_corr']['sim_err_corr']:.4f}). A Gaussian analogue uses $e_j = \\sigma_e(\\sqrt{{\\rho}}\\,e_0 + \\sqrt{{1-\\rho}}\\,e_j')$ with $\\rho=0.2$.

## 6. Predictive study

Design B, N in {{200, 400, 700}}, 500 reps, $\\lambda \\in \\{{0, 0.3\\}}$. Within each cohort a random 10-fold partition is drawn from the cohort seed. OLS models: baseline (1, G), (1, P), and the reference (1, $\\theta$), each with and without the raw O1 block (snapshot_corrected also reported). $R^2_{{cv}} = 1 - \\sum (C_i - \\hat C_{{-f(i)}})^2 / \\sum (C_i - \\bar C)^2$ ($\\bar C$ = cohort mean); $\\mathrm{{RMSE}}_{{cv}} = \\sqrt{{\\sum(C_i - \\hat C_{{-f(i)}})^2/N}}$. Reported: mean $\\Delta R^2_{{cv}}$, its MC SE (sd/$\\sqrt R$), and the share of cohorts with $\\Delta R^2_{{cv}} > 0$.

## 7. Exact field hypothesis H2

Predictor: the share of week-1 resolved functional episodes coded hypothesis-driven. Three operationalisations are used: `share_rule_H`, `share_clfH_prob` and `share_true_H` (oracle coding). Each enters as a 1-df test after in-sample standardisation.
* **Primary handling:** exclude learners with `n_resolved_week1` = 0 (18.25% of the population). N refers to consenting learners before exclusion, and the effective N is reported.
* **Sensitivity handling:** keep all N, set the missing share to the mean of the observed shares in the cohort, and add a missingness indicator. The indicator enters every model (exogenous regressor in 2SLS; in both $\\theta\\mid x$ and $C$ equations of the latent model), but only the share coefficient is tested.
* Methods: pretest_IV (HC1 Wald, 1 df), LATENT MML (LR, 1 df; $x$ = share [, indicator]), ORACLE-GLM (LR, 1 df).
* Mechanisms: M1 ($z_g$ from `gain_H_help`), M2 ($z_g$ from `gain_H`). $\\lambda \\in \\{{0, 0.15, 0.3, 0.5\\}}$; $\\lambda=0$ is common to M1 and M2. N in {{200, 400, 700}}.
* Reported: size, raw and size-adjusted power, and the smallest N in the grid with size-adjusted power $\\ge 0.80$ ("not reached" if none).

## 8. Population quantities

These are computed on all 2,000 learners. Two versions are given: the seed-101 draw (as in the original) and the mean over 200 regenerations (`SeedSequence([{res['meta']['base_seed']}, 7, k, ·])`). Quantities: corr($\\theta$, G), corr($\\theta$, P), split-half r, corr($z_g$, $\\theta$), $R^2(\\theta\\mid X)$, and the in-sample $\\Delta R^2$ of each block over baselines $\\theta$, G and P for $\\lambda \\in \\{{0, 0.3, 0.5\\}}$.

## 9. Runtime

Wall time of each 16-process study (Windows host, one BLAS thread per process): {', '.join(f'{k} {v/60:.1f} min' for k, v in wall.items())}. Analysis/merge (including the stacked pseudo-true fits): {res['runtime']['analysis_s']/60:.1f} min.

## 10. Reproduction check

`rq4_repro.py` is the original `make_outcomes2`/`_wald_hc1`/`a7_designs` loop verbatim, run with F1 and FS_old. It reproduces the published values exactly; see `rq4_summary.md`, section 1. Rerunning it with FS_corrected changes only the snapshot rows, because the cohort stream is identical.
"""
    open(path, "w", encoding="utf-8").write(T)
