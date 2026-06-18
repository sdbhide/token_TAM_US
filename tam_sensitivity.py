"""
tam_sensitivity.py
==================
Phase 3: Sensitivity and uncertainty attribution for the dual-segment TAM model.

Components
----------
1. VECTORIZED CLOSED-FORM EVALUATORS - E[TAM | params] and gamma_TAM as pure
   numpy functions of an (n, 10) parameter matrix. Because Phase 2 showed
   population noise contributes ~0.04% of variance, the deterministic map
   params -> E[TAM | params] IS the model for attribution purposes; this makes
   Sobol analysis essentially free (no inner simulation needed).

2. TORNADO - one-at-a-time perturbation of each parameter to its prior low/high
   with all others at the Phase 0 central anchors.

3. SOBOL INDICES - first-order (S1, Saltelli 2010 estimator) and total-order
   (ST, Jansen 1999 estimator) on a Saltelli design built from scipy's quasi-
   random Sobol sequence, with bootstrap CIs. SALib is not available in this
   environment; the implementation below is validated against the Ishigami
   function's analytical indices in test_tam_sensitivity.py.

   QoIs: Y = E[TAM | params]  (drivers of the level)
         Y = 1{E[TAM | params] > x}  (drivers of tail exceedance P(TAM > x))

4. SCENARIO GRID over (theta, sigma_X_e, E[N_e]) - E[TAM] and exact gamma_TAM
   on each grid point, with the gamma = epsilon contour (the sigma_crit
   frontier of theory doc §7) overlaid by the Phase 3 driver.

5. MACRO-CONSTRAINT CONDITIONING - importance weights that softly condition the
   Phase 0 prior on the observed current-revenue window ($14-22B), yielding a
   constrained posterior over parameters and TAM.

Parameter order (canonical, used everywhere):
    0 N_i   1 mu_I_i  2 sigma_I_i  3 mu_K_i  4 sigma_K_i
    5 N_e   6 mu_I_e  7 sigma_I_e  8 mu_K_e  9 sigma_K_e
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import qmc, triang

import tam_core as tc
from tam_mc import PHASE0_PRIORS, Prior

__all__ = [
    "PARAM_NAMES",
    "e_tam_vectorized",
    "gamma_tam_vectorized",
    "central_vector",
    "tornado",
    "saltelli_sample",
    "sobol_indices",
    "SobolResult",
    "scenario_grid",
    "macro_constraint_weights",
]

PARAM_NAMES: list[str] = [
    "N_i", "mu_I_i", "sigma_I_i", "mu_K_i", "sigma_K_i",
    "N_e", "mu_I_e", "sigma_I_e", "mu_K_e", "sigma_K_e",
]


# ---------------------------------------------------------------------------
# 1. Vectorized closed-form evaluators
# ---------------------------------------------------------------------------

def _split(P: np.ndarray):
    """Unpack an (n, 10) matrix into named columns (canonical order)."""
    P = np.atleast_2d(P)
    return (P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4],
            P[:, 5], P[:, 6], P[:, 7], P[:, 8], P[:, 9])


def e_tam_vectorized(P: np.ndarray) -> np.ndarray:
    """E[TAM | params] for each row of P. Annual USD. rho=0, no caps (Phase 0).

    E[TAM] = N_i exp(mu_Xi + s_Xi^2/2) + N_e exp(mu_Xe + s_Xe^2/2)
    """
    n_i, mu_Ii, s_Ii, mu_Ki, s_Ki, n_e, mu_Ie, s_Ie, mu_Ke, s_Ke = _split(P)
    e_xi = np.exp(mu_Ii + mu_Ki + (s_Ii**2 + s_Ki**2) / 2.0)
    e_xe = np.exp(mu_Ie + mu_Ke + (s_Ie**2 + s_Ke**2) / 2.0)
    return n_i * e_xi + n_e * e_xe


def gamma_tam_vectorized(P: np.ndarray) -> np.ndarray:
    """Exact skewness of TAM for each row, deterministic counts (Var(N)=0).

    Var_s = E[N_s] Var(X_s);  mu3_s = E[N_s] mu3(X_s);
    gamma = (mu3_i + mu3_e) / (Var_i + Var_e)^{3/2}.
    Matches tam_core.tam_skewness with CountParams(var=0, mu3=0); vectorized.
    """
    n_i, mu_Ii, s_Ii, mu_Ki, s_Ki, n_e, mu_Ie, s_Ie, mu_Ke, s_Ke = _split(P)

    def seg(n, mu, s2):
        w = np.exp(s2)
        e = np.exp(mu + s2 / 2.0)
        var = (w - 1.0) * np.exp(2.0 * mu + s2)
        mu3 = (w + 2.0) * np.sqrt(w - 1.0) * var**1.5
        return n * var, n * mu3

    v_i, m3_i = seg(n_i, mu_Ii + mu_Ki, s_Ii**2 + s_Ki**2)
    v_e, m3_e = seg(n_e, mu_Ie + mu_Ke, s_Ie**2 + s_Ke**2)
    return (m3_i + m3_e) / (v_i + v_e) ** 1.5


def central_vector(priors: dict[str, Prior] = PHASE0_PRIORS) -> np.ndarray:
    return np.array([priors[k].central for k in PARAM_NAMES])


# ---------------------------------------------------------------------------
# 2. Tornado (one-at-a-time)
# ---------------------------------------------------------------------------

def tornado(
    qoi: Callable[[np.ndarray], np.ndarray] = e_tam_vectorized,
    priors: dict[str, Prior] = PHASE0_PRIORS,
) -> list[dict]:
    """OAT perturbation of each parameter to its prior (low, high), others
    central. Returns rows sorted by swing = |y_high - y_low|, descending."""
    c = central_vector(priors)
    y0 = float(qoi(c[None, :])[0])
    rows = []
    for j, name in enumerate(PARAM_NAMES):
        lo_vec, hi_vec = c.copy(), c.copy()
        lo_vec[j] = priors[name].low
        hi_vec[j] = priors[name].high
        y_lo = float(qoi(lo_vec[None, :])[0])
        y_hi = float(qoi(hi_vec[None, :])[0])
        rows.append({
            "param": name, "y_low": y_lo, "y_high": y_hi, "y_central": y0,
            "swing": abs(y_hi - y_lo),
        })
    rows.sort(key=lambda r: -r["swing"])
    return rows


# ---------------------------------------------------------------------------
# 3. Sobol indices (Saltelli design, Jansen/Saltelli-2010 estimators)
# ---------------------------------------------------------------------------

def _prior_ppf(priors: dict[str, Prior]) -> Callable[[np.ndarray], np.ndarray]:
    """Map a (n, d) matrix of U(0,1) to parameter space via prior inverse CDFs."""
    specs = []
    for name in PARAM_NAMES:
        p = priors[name]
        if p.kind == "triangular":
            cshape = (p.central - p.low) / (p.high - p.low)
            specs.append(("tri", cshape, p.low, p.high - p.low))
        elif p.kind == "fixed":
            specs.append(("fix", p.central, 0.0, 0.0))
        else:
            specs.append(("uni", 0.0, p.low, p.high - p.low))

    def transform(U: np.ndarray) -> np.ndarray:
        X = np.empty_like(U)
        for j, (kind, a, loc, scale) in enumerate(specs):
            if kind == "tri":
                X[:, j] = triang.ppf(U[:, j], a, loc=loc, scale=scale)
            elif kind == "fix":
                X[:, j] = a
            else:
                X[:, j] = loc + scale * U[:, j]
        return X

    return transform


def saltelli_sample(
    n_base: int,
    priors: dict[str, Prior] = PHASE0_PRIORS,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saltelli design: returns (A, B, AB) with A,B of shape (n, d) and AB of
    shape (d, n, d), where AB[i] equals A with column i replaced by B's.

    Uses scipy's scrambled Sobol sequence in 2d dimensions, split into A | B -
    the standard quasi-random construction. n_base should be a power of 2.
    """
    d = len(PARAM_NAMES)
    eng = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    U = eng.random(n_base)
    # clip away exact 0/1 to keep ppf finite
    U = np.clip(U, 1e-12, 1 - 1e-12)
    transform = _prior_ppf(priors)
    A = transform(U[:, :d])
    B = transform(U[:, d:])
    AB = np.empty((d, n_base, d))
    for i in range(d):
        AB[i] = A
        AB[i][:, i] = B[:, i]
    return A, B, AB


@dataclass
class SobolResult:
    names: list[str]
    S1: np.ndarray
    ST: np.ndarray
    S1_ci: np.ndarray        # (d, 2) bootstrap 95% CI
    ST_ci: np.ndarray
    var_y: float

    def table(self) -> str:
        order = np.argsort(-self.ST)
        lines = [f"{'param':<10} {'S1':>8} {'S1 95% CI':>20} "
                 f"{'ST':>8} {'ST 95% CI':>20}"]
        for j in order:
            lines.append(
                f"{self.names[j]:<10} {self.S1[j]:8.3f} "
                f"[{self.S1_ci[j,0]:8.3f},{self.S1_ci[j,1]:8.3f}] "
                f"{self.ST[j]:8.3f} "
                f"[{self.ST_ci[j,0]:8.3f},{self.ST_ci[j,1]:8.3f}]")
        return "\n".join(lines)


def sobol_indices(
    qoi: Callable[[np.ndarray], np.ndarray],
    n_base: int = 2**13,
    priors: dict[str, Prior] = PHASE0_PRIORS,
    seed: int = 13,
    n_boot: int = 500,
) -> SobolResult:
    """First-order (Saltelli 2010) and total-order (Jansen 1999) Sobol indices.

        S1_i = mean( Y_B * (Y_ABi - Y_A) ) / Var(Y)
        ST_i = 0.5 * mean( (Y_A - Y_ABi)^2 ) / Var(Y)

    Bootstrap CIs resample design rows jointly across A/B/AB columns.
    Cost: (d + 2) * n_base model evaluations.
    """
    A, B, AB = saltelli_sample(n_base, priors, seed)
    d, n = AB.shape[0], A.shape[0]
    yA = qoi(A)
    yB = qoi(B)
    yAB = np.stack([qoi(AB[i]) for i in range(d)])  # (d, n)

    def estimate(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, ab = yA[idx], yB[idx], yAB[:, idx]
        y_all = np.concatenate([a, b])
        var = y_all.var()
        if var <= 0:
            return np.zeros(d), np.zeros(d)
        s1 = np.mean(b * (ab - a), axis=1) / var
        st = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
        return s1, st

    full = np.arange(n)
    S1, ST = estimate(full)

    rng = np.random.default_rng(seed + 1)
    s1_b = np.empty((n_boot, d))
    st_b = np.empty((n_boot, d))
    for bidx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s1_b[bidx], st_b[bidx] = estimate(idx)
    S1_ci = np.quantile(s1_b, [0.025, 0.975], axis=0).T
    ST_ci = np.quantile(st_b, [0.025, 0.975], axis=0).T

    y_all = np.concatenate([yA, yB])
    return SobolResult(names=list(PARAM_NAMES), S1=S1, ST=ST,
                       S1_ci=S1_ci, ST_ci=ST_ci, var_y=float(y_all.var()))


# ---------------------------------------------------------------------------
# 4. Scenario grid over (theta, sigma_X_e, E[N_e])
# ---------------------------------------------------------------------------

def scenario_grid(
    thetas: np.ndarray,
    sigma_xes: np.ndarray,
    n_es: Sequence[float],
    priors: dict[str, Prior] = PHASE0_PRIORS,
) -> dict:
    """E[TAM] and exact gamma_TAM over a (theta, sigma_X_e) grid at several
    E[N_e] levels. Consumer segment fixed at the Phase 0 central calibration.

    Parameterization: theta is the MEAN-RATIO scale factor E[X_e]/E[X_i]
    (theory doc §6.2). Given (theta, sigma_X_e), the enterprise location is
        mu_X_e = ln(theta * E[X_i]) - sigma_X_e^2 / 2,
    so each grid point holds the enterprise *mean* fixed at theta*E[X_i] while
    sigma_X_e reallocates that mean between body and tail - exactly the
    concentration experiment of theory doc §7.

    Returns dict with E_TAM and gamma arrays of shape (len(n_es),
    len(sigma_xes), len(thetas)).
    """
    c = central_vector(priors)
    n_i, mu_Ii, s_Ii, mu_Ki, s_Ki = c[0], c[1], c[2], c[3], c[4]
    mu_Xi = mu_Ii + mu_Ki
    s2_Xi = s_Ii**2 + s_Ki**2
    e_xi = np.exp(mu_Xi + s2_Xi / 2.0)

    # consumer aggregate moments (deterministic count)
    w_i = np.exp(s2_Xi)
    var_xi = (w_i - 1.0) * np.exp(2 * mu_Xi + s2_Xi)
    mu3_xi = (w_i + 2.0) * np.sqrt(w_i - 1.0) * var_xi**1.5
    V_i, M3_i = n_i * var_xi, n_i * mu3_xi

    TH, SG = np.meshgrid(thetas, sigma_xes)            # (n_sig, n_th)
    s2_e = SG**2
    mu_Xe = np.log(TH * e_xi) - s2_e / 2.0
    w_e = np.exp(s2_e)
    var_xe = (w_e - 1.0) * np.exp(2 * mu_Xe + s2_e)
    mu3_xe = (w_e + 2.0) * np.sqrt(w_e - 1.0) * var_xe**1.5
    e_xe = TH * e_xi                                    # by construction

    E_TAM = np.empty((len(n_es),) + TH.shape)
    GAMMA = np.empty_like(E_TAM)
    for k, n_e in enumerate(n_es):
        E_TAM[k] = n_i * e_xi + n_e * e_xe
        V = V_i + n_e * var_xe
        M3 = M3_i + n_e * mu3_xe
        GAMMA[k] = M3 / V**1.5

    return {"thetas": thetas, "sigma_xes": sigma_xes, "n_es": list(n_es),
            "E_TAM": E_TAM, "gamma": GAMMA,
            "consumer": {"n_i": n_i, "e_xi": e_xi, "V_i": V_i, "M3_i": M3_i}}


# ---------------------------------------------------------------------------
# 5. Macro-constraint conditioning (importance weights)
# ---------------------------------------------------------------------------

def macro_constraint_weights(
    e_tam_values: np.ndarray,
    low: float = 14e9,
    high: float = 22e9,
    soft_scale: float = 2e9,
) -> np.ndarray:
    """Soft-window importance weights conditioning on the Phase 0 macro
    constraint (current US inference revenue in [low, high]).

    w = 1 inside the window; Gaussian decay with sd = soft_scale outside.
    Interpretation: the model as calibrated estimates the CURRENT market, so
    parameter draws inconsistent with observed revenue are down-weighted.
    Returns unnormalized weights; use w / w.sum() for expectations and report
    effective sample size ESS = (sum w)^2 / sum w^2.
    """
    w = np.ones_like(e_tam_values, dtype=float)
    below = e_tam_values < low
    above = e_tam_values > high
    w[below] = np.exp(-0.5 * ((e_tam_values[below] - low) / soft_scale) ** 2)
    w[above] = np.exp(-0.5 * ((e_tam_values[above] - high) / soft_scale) ** 2)
    return w
