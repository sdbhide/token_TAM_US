"""
test_tam_viz.py
===============
Validation of the Phase 4 numeric helpers (the figure builders themselves are
visual and checked by eye, but their quantitative kernels are tested here).

Run:  python3 test_tam_viz.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

import tam_core as tc
import tam_viz as tv

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name:<50} {detail}")


def test_gini_lognormal():
    print("\n[1] Gini of lognormal vs analytical G = erf(sigma/2)")
    rng = np.random.default_rng(0)
    for sigma in (0.5, 1.0, 1.8):
        x = rng.lognormal(0.0, sigma, size=2_000_000)
        g_emp = tv.gini(x)
        g_true = 2 * norm.cdf(sigma / np.sqrt(2)) - 1   # = erf(sigma/2)
        check(f"sigma={sigma}: emp={g_emp:.4f} true={g_true:.4f}",
              abs(g_emp - g_true) < 0.01)


def test_gini_edge():
    print("\n[2] Gini edge cases")
    check("equal values -> Gini 0", abs(tv.gini(np.ones(1000))) < 1e-9)
    check("empty -> 0", tv.gini(np.array([])) == 0.0)
    # one unit holds everything -> Gini -> (n-1)/n
    x = np.zeros(1000); x[0] = 1.0
    check("maximal concentration -> ~(n-1)/n",
          abs(tv.gini(x) - 999 / 1000) < 1e-6)


def test_lorenz():
    print("\n[3] Lorenz curve endpoints and monotonicity")
    x = np.abs(np.random.default_rng(1).normal(size=5000)) + 0.1
    pop, val = tv.lorenz_points(x)
    check("starts at (0,0)", pop[0] == 0 and val[0] == 0)
    check("ends at (1,1)", np.isclose(pop[-1], 1) and np.isclose(val[-1], 1))
    check("value fraction is monotone non-decreasing",
          np.all(np.diff(val) >= -1e-12))
    check("Lorenz lies weakly below equality", np.all(val <= pop + 1e-9))


def test_skew_grid_monotone():
    print("\n[4] Empirical skew grid increases with sigma_X_e")
    cons = tc.SegmentParams(mu_I=10.6, sigma_I=0.88, mu_K=-6.52, sigma_K=0.70)
    ent_t = tc.SegmentParams(mu_I=12.0, sigma_I=1.8, mu_K=-6.74, sigma_K=1.2)
    n_es = np.array([1e6, 5e6])
    sigmas = np.linspace(1.85, 2.6, 4)   # all >= sigma_I so feasible
    grid = tv.empirical_skew_grid(n_es, sigmas, cons, int(1e8), ent_t,
                                  reps=3000, seed=5)
    # skewness should rise with sigma at each N_e (allow MC noise via trend)
    rising = np.all(grid[-1] > grid[0])
    check("skew(sigma_max) > skew(sigma_min) at both E[N_e]", rising,
          f"low={grid[0]} high={grid[-1]}")


def main():
    test_gini_lognormal()
    test_gini_edge()
    test_lorenz()
    test_skew_grid_monotone()
    n, p = len(_results), sum(_results)
    print(f"\n{'='*60}\n  {p}/{n} checks passed\n{'='*60}")
    return 0 if p == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
