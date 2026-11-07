
import numpy as np, json
from scipy.special import expit
from scipy.optimize import brentq
def calibrate(D, K=200, seed=777):
    """Moment calibration of the Section-4 measurement variants (pooled over K regenerations of the 2,000 learners)."""
    th = D["theta"]; rng = np.random.default_rng(seed)
    T = np.tile(th, K)
    lp = 0.2 + 0.9 * T
    P1 = rng.binomial(15, expit(lp)); P2 = rng.binomial(15, expit(lp))
    C = rng.binomial(30, expit(0.4 + 0.9 * T))
    r_half = float(np.corrcoef(P1, P2)[0, 1])
    R2_post = float(np.corrcoef(C, T)[0, 1] ** 2)
    varC = float(C.var()); vth = float(th.var())
    se_half = float(np.sqrt(vth * (1 - r_half) / r_half))
    s = float(np.sqrt(R2_post * varC / (0.81 * vth)))
    se_post = float(np.sqrt((1 - R2_post) * varC))
    # binomial halves with a shared logit nuisance eta ~ N(0, tau^2): error corr = E Var_eta(15p) / (E 15 E_eta p(1-p) + E Var_eta(15p))
    xg, wg = np.polynomial.hermite_e.hermegauss(41); wg = wg / wg.sum()
    def err_corr(tau):
        p = expit(0.2 + 0.9 * th[:, None] + tau * xg[None, :])
        m = (p * wg).sum(1)
        v_between = 225 * ((p ** 2 * wg).sum(1) - m ** 2)
        v_within = 15 * ((p * (1 - p)) * wg).sum(1)
        return float(v_between.mean() / (v_within.mean() + v_between.mean()))
    tau = float(brentq(lambda t: err_corr(t) - 0.2, 1e-3, 5))
    # check by simulation
    eta = tau * rng.normal(size=len(T))
    Q1 = rng.binomial(15, expit(lp + eta)); Q2 = rng.binomial(15, expit(lp + eta))
    pm = (expit(0.2 + 0.9 * th[:, None] + tau * xg[None, :]) * wg).sum(1)
    mT = 15 * np.tile(pm, K)
    sim_err_corr = float(np.corrcoef(Q1 - mT, Q2 - mT)[0, 1])
    # implied Gaussian reliability check
    G1 = T + se_half * rng.normal(size=len(T)); G2 = T + se_half * rng.normal(size=len(T))
    return {"K": K, "seed": seed, "binom_half_reliability": r_half, "binom_post_R2_theta_lam0": R2_post,
            "var_theta": vth, "var_C_binom_lam0": varC,
            "gauss": {"se_half": se_half, "s": s, "se_post": se_post,
                      "check_half_r": float(np.corrcoef(G1, G2)[0, 1])},
            "gauss_corr": {"se_half": se_half, "rho": 0.2, "s": s, "se_post": se_post},
            "binom_corr": {"tau": tau, "target_err_corr": 0.2, "sim_err_corr": sim_err_corr,
                           "half_reliability": float(np.corrcoef(Q1, Q2)[0, 1])}}
