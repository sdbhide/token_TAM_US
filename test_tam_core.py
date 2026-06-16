"""
test_tam_core.py
================
Validation of the Phase 1 analytical core against independent numerical
ground truth. Each closed form is checked against either scipy.integrate
quadrature or large-sample brute-force simulation.

Run:  python3 test_tam_core.py
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

import tam_core as tc

RNG = np.random.default_rng(20260612)
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []


def check(name, got, want, rtol=1e-3, atol=0.0):
    ok = np.allclose(got, want, rtol=rtol, atol=atol)
    _results.append(ok)
    rel = abs(got - want) / (abs(want) + 1e-30)
    print(f"  [{PASS if ok else FAIL}] {name:<46} "
          f"got={got:.6g} want={want:.6g} rel={rel:.2e}")
    return ok


# ---------------------------------------------------------------------------
# 1. Lognormal moments vs. quadrature
# ---------------------------------------------------------------------------
def test_lognormal_moments():
    print("\n[1] Lognormal moments vs. numerical integration")
    mu, sigma = 0.7, 0.9
    mean, var, mu3 = tc.lognormal_moments(mu, sigma)

    def pdf(x):
        return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))

    m1, _ = integrate.quad(lambda x: x * pdf(x), 0, np.inf, limit=200)
    m2, _ = integrate.quad(lambda x: x**2 * pdf(x), 0, np.inf, limit=200)
    m3, _ = integrate.quad(lambda x: x**3 * pdf(x), 0, np.inf, limit=200)
    q_var = m2 - m1**2
    q_mu3 = m3 - 3 * m1 * m2 + 2 * m1**3

    check("E[X]", mean, m1)
    check("Var(X)", var, q_var)
    check("mu3(X)", mu3, q_mu3, rtol=5e-3)


# ---------------------------------------------------------------------------
# 2. Truncated lognormal vs. quadrature
# ---------------------------------------------------------------------------
def test_truncated_lognormal():
    print("\n[2] Truncated lognormal moments vs. numerical integration")
    mu, sigma, cap = 0.5, 1.1, 12.0
    mean, var, mu3 = tc.truncated_lognormal_moments(mu, sigma, cap)

    def pdf(x):
        return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))

    Z, _ = integrate.quad(pdf, 0, cap, limit=200)
    m1, _ = integrate.quad(lambda x: x * pdf(x), 0, cap, limit=200)
    m2, _ = integrate.quad(lambda x: x**2 * pdf(x), 0, cap, limit=200)
    m3, _ = integrate.quad(lambda x: x**3 * pdf(x), 0, cap, limit=200)
    m1, m2, m3 = m1 / Z, m2 / Z, m3 / Z
    q_var = m2 - m1**2
    q_mu3 = m3 - 3 * m1 * m2 + 2 * m1**3

    check("E[X|X<=cap]", mean, m1)
    check("Var(X|X<=cap)", var, q_var)
    check("mu3(X|X<=cap)", mu3, q_mu3, rtol=5e-3)


# ---------------------------------------------------------------------------
# 3. Product-of-lognormals stays lognormal; rho correlation
# ---------------------------------------------------------------------------
def test_segment_unit_moments_rho():
    print("\n[3] Segment unit X=K*I with correlation, vs. brute sampling")
    seg = tc.SegmentParams(mu_I=2.0, sigma_I=0.8, mu_K=-5.0, sigma_K=0.6, rho=-0.4)
    mean, var, mu3 = tc.segment_unit_moments(seg)

    # brute: sample correlated (lnI, lnK)
    n = 4_000_000
    cov = [[seg.sigma_I**2, seg.rho * seg.sigma_I * seg.sigma_K],
           [seg.rho * seg.sigma_I * seg.sigma_K, seg.sigma_K**2]]
    draws = RNG.multivariate_normal([seg.mu_I, seg.mu_K], cov, size=n)
    X = np.exp(draws[:, 0] + draws[:, 1])
    check("E[X]", mean, X.mean(), rtol=5e-3)
    check("Var(X)", var, X.var(), rtol=2e-2)


# ---------------------------------------------------------------------------
# 4. Compound-sum moments vs. brute simulation
# ---------------------------------------------------------------------------
def test_compound_moments():
    print("\n[4] Compound-sum moments vs. brute simulation (Poisson N)")
    seg = tc.SegmentParams(mu_I=1.0, sigma_I=0.5, mu_K=-2.0, sigma_K=0.4)
    lam = 300.0
    # Poisson: Var = mean, mu3 = mean
    counts = tc.CountParams(mean=lam, var=lam, mu3=lam)
    e_s, var_s, mu3_s = tc.compound_moments(seg, counts)

    reps = 200_000
    Ns = RNG.poisson(lam, size=reps)
    ex, _, _ = tc.segment_unit_moments(seg)
    totals = np.empty(reps)
    # vectorized-ish: draw all spends in one block
    total_draws = Ns.sum()
    spends = np.exp(RNG.normal(seg.mu_X, seg.sigma_X, size=total_draws))
    idx = np.concatenate([[0], np.cumsum(Ns)])
    csum = np.concatenate([[0.0], np.cumsum(spends)])
    totals = csum[idx[1:]] - csum[idx[:-1]]

    check("E[S]", e_s, totals.mean(), rtol=5e-3)
    check("Var(S)", var_s, totals.var(), rtol=3e-2)
    s_mean = totals.mean()
    emp_mu3 = np.mean((totals - s_mean) ** 3)
    check("mu3(S)", mu3_s, emp_mu3, rtol=8e-2)


# ---------------------------------------------------------------------------
# 5. TAM aggregates and skewness vs. brute two-segment simulation
# ---------------------------------------------------------------------------
def test_tam_aggregate():
    print("\n[5] Two-segment TAM mean/var/skew vs. brute simulation")
    cons = tc.SegmentParams(mu_I=1.0, sigma_I=0.5, mu_K=-2.0, sigma_K=0.4)
    ent = tc.SegmentParams(mu_I=3.0, sigma_I=1.0, mu_K=-1.0, sigma_K=0.7)
    cons_n = tc.CountParams(mean=500.0, var=500.0, mu3=500.0)   # Poisson-like
    ent_n = tc.CountParams(mean=80.0, var=80.0, mu3=80.0)

    e_tam = tc.tam_expected(cons, cons_n, ent, ent_n)
    v_tam = tc.tam_variance(cons, cons_n, ent, ent_n)
    g_tam = tc.tam_skewness(cons, cons_n, ent, ent_n)

    reps = 300_000

    def sim_segment(seg, cp):
        Ns = RNG.poisson(cp.mean, size=reps)
        tot = Ns.sum()
        spends = np.exp(RNG.normal(seg.mu_X, seg.sigma_X, size=tot))
        idx = np.concatenate([[0], np.cumsum(Ns)])
        csum = np.concatenate([[0.0], np.cumsum(spends)])
        return csum[idx[1:]] - csum[idx[:-1]]

    tam = sim_segment(cons, cons_n) + sim_segment(ent, ent_n)
    check("E[TAM]", e_tam, tam.mean(), rtol=5e-3)
    check("Var(TAM)", v_tam, tam.var(), rtol=4e-2)
    m = tam.mean()
    emp_g = np.mean((tam - m) ** 3) / tam.var() ** 1.5
    check("skew(TAM)", g_tam, emp_g, rtol=1.2e-1)


# ---------------------------------------------------------------------------
# 6. sigma_crit: numeric root-find actually hits gamma == epsilon
# ---------------------------------------------------------------------------
def test_sigma_crit_numeric():
    print("\n[6] sigma_crit_numeric: verify gamma(sigma_crit) == epsilon")
    cons = tc.SegmentParams(mu_I=1.0, sigma_I=0.5, mu_K=-2.0, sigma_K=0.4)
    ent_template = tc.SegmentParams(mu_I=3.0, sigma_I=0.6, mu_K=-1.0, sigma_K=0.6)
    cons_n = tc.CountParams(mean=1e6, var=1e6, mu3=1e6)
    ent_n = tc.CountParams(mean=5e4, var=5e4, mu3=5e4)
    eps = 0.05

    sig = tc.sigma_crit_numeric(cons, cons_n, ent_template, ent_n, eps)
    print(f"      -> sigma_crit = {sig:.6g}")

    # rebuild enterprise at that sigma and confirm gamma == eps
    a, b, c = 1.0, 2 * ent_template.rho * ent_template.sigma_I, \
        ent_template.sigma_I**2 - sig**2
    disc = b**2 - 4 * a * c
    sigK = (-b + np.sqrt(disc)) / 2.0
    ent = tc.SegmentParams(mu_I=ent_template.mu_I, sigma_I=ent_template.sigma_I,
                           mu_K=ent_template.mu_K, sigma_K=sigK)
    g = tc.tam_skewness(cons, cons_n, ent, ent_n)
    check("gamma(sigma_crit)", g, eps, rtol=1e-4)


# ---------------------------------------------------------------------------
# 7. sigma_crit closed forms: correct vs. the document's broken form
# ---------------------------------------------------------------------------
def test_sigma_crit_closed_forms():
    print("\n[7] sigma_crit closed forms (sanity + corrected-vs-document)")
    # enterprise-dominated form: check it satisfies its own approximation
    e_n_e, eps = 5e4, 0.05
    s = tc.sigma_crit_enterprise_dominated(e_n_e, eps)
    # approx gamma ~ e^{1.5 s^2} / sqrt(E[N_e]) should equal eps
    approx_gamma = np.exp(1.5 * s**2) / np.sqrt(e_n_e)
    check("ent-dominated self-consistency", approx_gamma, eps, rtol=1e-6)

    # corrected consumer-anchored form returns sigma (not sigma^2). Choose a
    # genuinely consumer-anchored regime (small theta, huge consumer base) so
    # the log-argument exceeds 1 and a real root exists.
    e_n_i, e_n_e, sig_i, theta_med, eps2 = 5e7, 1e3, 1.2, 3.0, 0.05
    s2 = tc.sigma_crit_consumer_anchored(
        e_n_i=e_n_i, e_n_e=e_n_e, sigma_i=sig_i,
        theta_med=theta_med, epsilon=eps2)
    print(f"      -> corrected consumer-anchored sigma_crit = {s2:.6g}")

    # the document's formula (1/2)*ln(arg) with no outer root, same inputs:
    num = eps2**2 * e_n_i**3 * (np.exp(sig_i**2) - 1.0) ** 3
    den = e_n_e**2 * theta_med**6
    doc_val = 0.5 * np.log(num / den)
    print(f"      -> document's (1/2, no-root) value        = {doc_val:.6g}")
    print(f"      -> ratio doc/corrected                    = {doc_val/s2:.4g} "
          f"(document overstates sigma by ~3x here)")
    ok = s2 > 0 and np.isfinite(s2)
    _results.append(ok)
    print(f"  [{PASS if ok else FAIL}] corrected form returns positive finite sigma")


def main():
    test_lognormal_moments()
    test_truncated_lognormal()
    test_segment_unit_moments_rho()
    test_compound_moments()
    test_tam_aggregate()
    test_sigma_crit_numeric()
    test_sigma_crit_closed_forms()

    n = len(_results)
    p = sum(_results)
    print(f"\n{'='*60}\n  {p}/{n} checks passed\n{'='*60}")
    return 0 if p == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
