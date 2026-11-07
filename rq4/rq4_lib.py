
"""rq4_lib.py - shared estimators and data-generating mechanisms for the RQ4 re-analysis."""
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import expit

BASE_SEED = 20261105
NQ = 31  # Gauss-Hermite nodes for the latent model

def load(npz):
    d = np.load(npz, allow_pickle=True)
    D = {k: d[k] for k in d.files}
    g = D["gain_H_help"]; D["zg_M1"] = (g - g.mean()) / g.std()
    g = D["gain_H"]; D["zg_M2"] = (g - g.mean()) / g.std()
    return D

def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED] + [int(k) for k in key]))

# ----------------------------------------------------------------- DGPs
def gen_pre(rng, th, variant="binom", par=None):
    """Prior grade G and pre-test halves P1,P2 for the learners in th."""
    n = len(th)
    G = 0.6 * th + 0.8 * rng.normal(size=n)
    if variant in ("binom", "binomP_linC"):
        lp = 0.2 + 0.9 * th
        P1 = rng.binomial(15, expit(lp)).astype(float); P2 = rng.binomial(15, expit(lp)).astype(float)
    elif variant == "binom_corr":
        eta = par["tau"] * rng.normal(size=n)          # shared nuisance on both halves' logits
        lp = 0.2 + 0.9 * th + eta
        P1 = rng.binomial(15, expit(lp)).astype(float); P2 = rng.binomial(15, expit(lp)).astype(float)
    elif variant in ("gauss", "gaussP_binomC"):
        se = par["se_half"]
        P1 = th + se * rng.normal(size=n); P2 = th + se * rng.normal(size=n)
    elif variant == "gauss_corr":
        se = par["se_half"]; r = par["rho"]
        e0 = rng.normal(size=n)
        P1 = th + se * (np.sqrt(r) * e0 + np.sqrt(1 - r) * rng.normal(size=n))
        P2 = th + se * (np.sqrt(r) * e0 + np.sqrt(1 - r) * rng.normal(size=n))
    else:
        raise ValueError(variant)
    return G, P1, P2

def gen_post(rng, th, zg, lam, variant="binom", par=None):
    n = len(th)
    if variant in ("binom", "binom_corr", "gaussP_binomC"):
        return rng.binomial(30, expit(0.4 + 0.9 * th + lam * zg)).astype(float)
    s = par["s"]; sc = par["se_post"]
    return s * (0.9 * th + lam * zg) + sc * rng.normal(size=n)

def make_outcomes2(th, zg, lam, seed=101):
    """Original fixed-corpus generator (Design A); identical call order to study_v1.make_outcomes2."""
    rng = np.random.default_rng(seed)
    G = 0.6 * th + 0.8 * rng.normal(size=len(th))
    lp = 0.2 + 0.9 * th
    P1 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    P2 = rng.binomial(15, 1 / (1 + np.exp(-lp))).astype(float)
    C = rng.binomial(30, 1 / (1 + np.exp(-(0.4 + 0.9 * th + lam * zg)))).astype(float)
    return G, P1, P2, C

# ----------------------------------------------------------------- helpers
def standardize(X):
    """Centre/scale columns in-sample; drop constant columns. Returns Xs, kept-index."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    sd = X.std(0)
    keep = np.flatnonzero(sd > 1e-12)
    Xs = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    return Xs, keep

def wald_hc1(y, X, Z, test_cols):
    """2SLS (OLS when Z is X) with HC1 sandwich; identical estimator to study_v1._wald_hc1
    (projection computed without forming the N x N matrix). Returns p, b, V."""
    n, k = X.shape
    if Z is X:
        Xh = X
    else:
        Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A_ = np.linalg.pinv(Xh.T @ Xh)
    b = A_ @ (Xh.T @ y)
    u = y - X @ b
    meat = (Xh * (u ** 2)[:, None]).T @ Xh
    V = A_ @ meat @ A_ * n / (n - k)
    tc = list(test_cols)
    bb = b[tc]
    W = float(bb @ np.linalg.pinv(V[np.ix_(tc, tc)]) @ bb)
    return float(chi2.sf(W, len(tc))), b, V

def first_stage_F(P1, P2, Xs):
    """First stage P1 ~ 1 + P2 + block: conventional F (=t^2) and HC1-robust F for the excluded instrument."""
    n = len(P1)
    Z = np.column_stack([np.ones(n), P2, Xs])
    b = np.linalg.lstsq(Z, P1, rcond=None)[0]
    u = P1 - Z @ b
    A_ = np.linalg.pinv(Z.T @ Z)
    k = Z.shape[1]
    s2 = u @ u / (n - k)
    F_conv = b[1] ** 2 / (s2 * A_[1, 1])
    meat = (Z * (u ** 2)[:, None]).T @ Z
    V = A_ @ meat @ A_ * n / (n - k)
    return float(F_conv), float(b[1] ** 2 / V[1, 1])

def glm_binom(y, X, m=30, beta0=None, maxit=50, tol=1e-10):
    """Binomial-logit GLM by IRLS (Newton). Returns beta, loglik (without the constant), converged."""
    n, k = X.shape
    if beta0 is None:
        pbar = np.clip(y.mean() / m, 1e-3, 1 - 1e-3)
        beta = np.zeros(k); beta[0] = np.log(pbar / (1 - pbar))
    else:
        beta = beta0.copy()
    ll_old = -np.inf
    for it in range(maxit):
        eta = X @ beta
        mu = expit(eta)
        W = m * mu * (1 - mu)
        g = X.T @ (y - m * mu)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, g)
        beta = beta + step
        eta = X @ beta
        ll = float(y @ eta - m * np.logaddexp(0, eta).sum())
        if abs(ll - ll_old) < tol and np.max(np.abs(step)) < 1e-7:
            return beta, ll, True
        ll_old = ll
    return beta, ll, False

def glm_lr(y, theta, Xs, test_cols_in_X, m=30):
    """LR test of block columns (indices into Xs) in logit GLM of y on 1+theta+Xs."""
    n = len(y)
    Xf = np.column_stack([np.ones(n), theta, Xs])
    keep_r = [j for j in range(Xs.shape[1]) if j not in set(test_cols_in_X)]
    Xr = np.column_stack([np.ones(n), theta, Xs[:, keep_r]])
    br, llr, cr = glm_binom(y, Xr, m)
    b0 = np.zeros(Xf.shape[1]); b0[:2] = br[:2]
    for i, j in enumerate(keep_r): b0[2 + j] = br[2 + i]
    bf, llf, cf = glm_binom(y, Xf, m, beta0=b0)
    LR = max(2 * (llf - llr), 0.0)
    return float(chi2.sf(LR, len(test_cols_in_X))), bf, (cr and cf)

# ----------------------------------------------------------------- latent MIMIC model
_x, _w = np.polynomial.hermite_e.hermegauss(NQ)
GH_X = _x; GH_LW = np.log(_w / np.sqrt(2 * np.pi))

def set_nodes(nq):
    global GH_X, GH_LW
    x, w = np.polynomial.hermite_e.hermegauss(nq)
    GH_X = x; GH_LW = np.log(w / np.sqrt(2 * np.pi))

def _nll(par, P, C, Xd, Xg, mP, mC):
    """Negative marginal log-likelihood (up to binomial constants) and analytic gradient."""
    qd = Xd.shape[1]
    c, dd, a, b = par[:4]
    delta = par[4:4 + qd]; gam = par[4 + qd:]
    mu = Xd @ delta if qd else np.zeros(len(P))
    th = mu[:, None] + GH_X[None, :]
    ep = c + dd * th
    ec = a + b * th
    if len(gam): ec = ec + (Xg @ gam)[:, None]
    sp_ = expit(-ep); sc_ = expit(-ec)            # 1 - sigmoid
    # log-lik kernel: y*eta - m*softplus(eta), softplus(eta) = -log(expit(-eta))
    l = P[:, None] * ep + mP * np.log(sp_) + C[:, None] * ec + mC * np.log(sc_) + GH_LW[None, :]
    lm = l.max(1, keepdims=True)
    e = np.exp(l - lm)
    s = e.sum(1, keepdims=True)
    L = (lm + np.log(s))[:, 0]
    r = e / s
    rgp = r * (P[:, None] - mP * (1.0 - sp_))
    rgc = r * (C[:, None] - mC * (1.0 - sc_))
    g = np.empty_like(par)
    g[0] = rgp.sum(); g[1] = (rgp * th).sum(); g[2] = rgc.sum(); g[3] = (rgc * th).sum()
    if qd: g[4:4 + qd] = Xd.T @ (dd * rgp.sum(1) + b * rgc.sum(1))
    if len(gam): g[4 + qd:] = Xg.T @ rgc.sum(1)
    return -L.sum(), -g

def latent_fit(P, C, Xd, Xg, start=None, mP=30, mC=30):
    """Marginal ML of the MIMIC model: theta|X ~ N(Xd delta, 1); P|theta ~ Bin(30, expit(c+d theta))
    (= P1,P2 iid Bin(15,.) with common item parameters); C|theta,X ~ Bin(30, expit(a+b theta+Xg gamma)).
    Location/scale of theta fixed by intercept 0 (Xd centred) and residual variance 1."""
    qd = Xd.shape[1]; qg = Xg.shape[1]
    if start is None:
        lp = np.log((P.mean() + .5) / (mP - P.mean() + .5)); lc = np.log((C.mean() + .5) / (mC - C.mean() + .5))
        start = np.r_[lp, 0.9, lc, 0.9, np.zeros(qd + qg)]
    f = lambda x: _nll(x, P, C, Xd, Xg, mP, mC)
    res = minimize(f, start, jac=True, method="BFGS", options={"gtol": 1e-4, "maxiter": 2000})
    gmax = float(np.max(np.abs(f(res.x)[1])))
    ok = gmax < 1e-3
    tries = 1
    if not ok:  # retry 1: L-BFGS-B from current point
        tries += 1
        res2 = minimize(f, res.x, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8})
        if res2.fun <= res.fun: res = res2
        gmax = float(np.max(np.abs(f(res.x)[1]))); ok = gmax < 1e-3
    if not ok:  # retry 2: different start
        tries += 1
        st2 = start.copy(); st2[1] = 0.5; st2[3] = 0.5
        res3 = minimize(f, st2, jac=True, method="BFGS", options={"gtol": 1e-4, "maxiter": 4000})
        if res3.fun < res.fun: res = res3
        gmax = float(np.max(np.abs(f(res.x)[1]))); ok = gmax < 1e-3
    return res.x, float(res.fun), ok, tries

def latent_lr(P, C, Xd, Xg, test_cols_in_Xg):
    """LR test of gamma[test_cols]=0. Restricted model keeps the other Xg columns."""
    qd = Xd.shape[1]
    keep = [j for j in range(Xg.shape[1]) if j not in set(test_cols_in_Xg)]
    xr, fr, okr, tr = latent_fit(P, C, Xd, Xg[:, keep])
    st = np.r_[xr[:4 + qd], np.zeros(Xg.shape[1])]
    for i, j in enumerate(keep): st[4 + qd + j] = xr[4 + qd + i]
    xf, ff, okf, tf = latent_fit(P, C, Xd, Xg, start=st)
    LR = 2 * (fr - ff)
    ok = okr and okf and LR > -1e-4
    return float(chi2.sf(max(LR, 0.0), len(test_cols_in_Xg))), xf, ok, (tr + tf)

# ----------------------------------------------------------------- RQ3 battery
DESIGNS = ["oracle_linear", "oracle_glm", "prior_grade_OLS", "pretest_OLS", "pretest_IV", "latent_MML"]

def battery(th, G, P1, P2, C, Xraw, designs=DESIGNS):
    """Joint tests of the trace block. Block standardised in-sample (tests are invariant)."""
    Xs, keep = standardize(Xraw)
    n, q = Xs.shape
    P = P1 + P2
    one = np.ones(n)
    tc = list(range(2, 2 + q))
    out = {"q": q}
    if "oracle_linear" in designs:
        X = np.column_stack([one, th, Xs]); pv, b, _ = wald_hc1(C, X, X, tc); out["oracle_linear"] = (pv, b[tc])
    if "oracle_glm" in designs:
        pv, b, ok = glm_lr(C, th, Xs, list(range(q))); out["oracle_glm"] = (pv, b[tc], b[1], ok)
    if "prior_grade_OLS" in designs:
        X = np.column_stack([one, G, Xs]); pv, b, _ = wald_hc1(C, X, X, tc); out["prior_grade_OLS"] = (pv, b[tc])
    if "pretest_OLS" in designs:
        X = np.column_stack([one, P, Xs]); pv, b, _ = wald_hc1(C, X, X, tc); out["pretest_OLS"] = (pv, b[tc])
    if "pretest_IV" in designs:
        X = np.column_stack([one, P1, Xs]); Z = np.column_stack([one, P2, Xs])
        pv, b, _ = wald_hc1(C, X, Z, tc); F, Frob = first_stage_F(P1, P2, Xs)
        out["pretest_IV"] = (pv, b[tc], F, Frob, b[1])
    if "latent_MML" in designs:
        pv, x, ok, tries = latent_lr(P, C, Xs, Xs, list(range(q)))
        out["latent_MML"] = (pv, x[4 + q:], x[3] / x[1], ok, tries)
    return out
