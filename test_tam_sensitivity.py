"""
test_tam_sensitivity.py
=======================
Phase 3 validation. The critical test: the hand-rolled Saltelli/Jansen Sobol
estimator must reproduce the ANALYTICAL indices of the Ishigami function
(the standard benchmark, with exact closed-form S1/ST). Plus cross-checks of
the vectorized evaluators against tam_core, and structural sanity of the
tornado, scenario grid, and constraint weights.

Run:  python3 test_tam_sensitivity.py
"""

from __future__ import annotations

import numpy as np

import tam_core as tc
import tam_sensitivity as ts
from tam_mc import PHASE0_PRIORS, Prior

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name:<54} {detail}")


# ---------------------------------------------------------------------------
# 1. Sobol estimator vs Ishigami analytical indices
# ---------------------------------------------------------------------------
def test_sobol_ishigami():
    print("\n[1] Sobol estimator vs Ishigami analytical indices")
    a, b = 7.0, 0.1
    pi = np.pi

    # Analytical values (e.g. Saltelli et al. 2008):
    V1 = 0.5 * (1 + b * pi**4 / 5) ** 2
    V2 = a**2 / 8
    V13 = b**2 * pi**8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    S1_true = np.array([V1 / V, V2 / V, 0.0])
    ST_true = np.array([(V1 + V13) / V, V2 / V, V13 / V])

    # Build a 3-param uniform(-pi, pi) prior set and an Ishigami QoI that reads
    # only the first three canonical columns.
    pri = {k: Prior(-pi, pi, kind="uniform") for k in ts.PARAM_NAMES}

    def ishigami(P):
        P = np.atleast_2d(P)
        x1, x2, x3 = P[:, 0], P[:, 1], P[:, 2]
        return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)

    res = ts.sobol_indices(ishigami, n_base=2**13, priors=pri, seed=42,
                           n_boot=200)
    for j, name in enumerate(["x1", "x2", "x3"]):
        check(f"S1[{name}] = {res.S1[j]:.3f} (true {S1_true[j]:.3f})",
              abs(res.S1[j] - S1_true[j]) < 0.02)
        check(f"ST[{name}] = {res.ST[j]:.3f} (true {ST_true[j]:.3f})",
              abs(res.ST[j] - ST_true[j]) < 0.02)
    # remaining 7 dummy parameters must register ~zero total effect
    check("dummy params have |ST| < 0.01",
          np.all(np.abs(res.ST[3:]) < 0.01),
          f"max |ST_dummy| = {np.abs(res.ST[3:]).max():.4f}")


# ---------------------------------------------------------------------------
# 2. Vectorized evaluators vs tam_core
# ---------------------------------------------------------------------------
def test_evaluators_vs_core():
    print("\n[2] Vectorized evaluators vs tam_core closed forms")
    rng = np.random.default_rng(5)
    ok_e, ok_g = True, True
    for _ in range(50):
        v = ts.central_vector().copy()
        # jitter all params within prior ranges
        for j, name in enumerate(ts.PARAM_NAMES):
            p = PHASE0_PRIORS[name]
            v[j] = rng.uniform(p.low, p.high)
        cons = tc.SegmentParams(mu_I=v[1], sigma_I=v[2], mu_K=v[3], sigma_K=v[4])
        ent = tc.SegmentParams(mu_I=v[6], sigma_I=v[7], mu_K=v[8], sigma_K=v[9])
        cn = tc.CountParams(mean=v[0], var=0.0, mu3=0.0)
        en = tc.CountParams(mean=v[5], var=0.0, mu3=0.0)
        e_core = tc.tam_expected(cons, cn, ent, en)
        g_core = tc.tam_skewness(cons, cn, ent, en)
        e_vec = float(ts.e_tam_vectorized(v[None, :])[0])
        g_vec = float(ts.gamma_tam_vectorized(v[None, :])[0])
        ok_e &= np.isclose(e_vec, e_core, rtol=1e-10)
        ok_g &= np.isclose(g_vec, g_core, rtol=1e-10)
    check("E[TAM] vectorized == tam_core (50 random params)", ok_e)
    check("gamma vectorized == tam_core (50 random params)", ok_g)


# ---------------------------------------------------------------------------
# 3. Tornado structure
# ---------------------------------------------------------------------------
def test_tornado():
    print("\n[3] Tornado structure")
    rows = ts.tornado()
    check("returns all 10 params, sorted by swing",
          len(rows) == 10 and
          all(rows[i]["swing"] >= rows[i+1]["swing"] for i in range(9)))
    # Jensen: E[TAM] increases in every sigma and mu -> y_high > y_low for all
    monotone = all(r["y_high"] > r["y_low"] for r in rows)
    check("E[TAM] monotone increasing in every parameter", monotone)
    top = rows[0]["param"]
    check(f"top driver is an enterprise dispersion/location param ({top})",
          top in ("sigma_I_e", "sigma_K_e", "mu_I_e", "mu_K_e"))


# ---------------------------------------------------------------------------
# 4. Scenario grid: limits and monotonicity
# ---------------------------------------------------------------------------
def test_scenario_grid():
    print("\n[4] Scenario grid sanity")
    thetas = np.geomspace(5, 5000, 12)
    sigmas = np.linspace(1.0, 3.0, 9)
    grid = ts.scenario_grid(thetas, sigmas, n_es=[1e6, 5e6])

    # E[TAM] must be exactly N_i E[X_i] + N_e theta E[X_i] (mean-ratio pin)
    c = grid["consumer"]
    expect = c["n_i"] * c["e_xi"] + 5e6 * thetas * c["e_xi"]
    got = grid["E_TAM"][1, 0, :]   # any sigma row — mean is sigma-free
    check("E[TAM] independent of sigma_X_e at fixed theta (mean pinned)",
          np.allclose(grid["E_TAM"][1, 0, :], grid["E_TAM"][1, -1, :]))
    check("E[TAM] matches N_i E[X_i] + N_e theta E[X_i]",
          np.allclose(got, expect, rtol=1e-12))

    g = grid["gamma"]
    check("gamma increases in sigma_X_e (concentration -> skew)",
          np.all(np.diff(g[1], axis=0) > 0))
    check("gamma higher at smaller E[N_e] (fewer firms -> slower CLT)",
          np.all(g[0, :, -1] > g[1, :, -1]))


# ---------------------------------------------------------------------------
# 5. Macro-constraint weights
# ---------------------------------------------------------------------------
def test_weights():
    print("\n[5] Macro-constraint weights")
    x = np.array([10e9, 14e9, 18e9, 22e9, 30e9])
    w = ts.macro_constraint_weights(x)
    check("inside window -> weight 1", np.all(w[1:4] == 1.0))
    check("outside window -> weight < 1, symmetric decay",
          w[0] < 1.0 and w[4] < 1.0 and
          np.isclose(w[0], np.exp(-0.5 * (4 / 2) ** 2)))


def main():
    test_sobol_ishigami()
    test_evaluators_vs_core()
    test_tornado()
    test_scenario_grid()
    test_weights()
    n, p = len(_results), sum(_results)
    print(f"\n{'='*64}\n  {p}/{n} checks passed\n{'='*64}")
    return 0 if p == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
