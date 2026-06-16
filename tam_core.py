"""
tam_core.py
===========
Phase 1: Analytical core for the dual-segment AI-inference-token TAM model.

This module implements the closed-form machinery from the framework document,
with the corrections discussed in review:

  * Section 6.2 second-case sigma_crit fixed (1/6 under the root, not 1/2; and
    it returns sigma, not sigma^2).
  * theta is treated as carrying sigma_{X,e} dependence, so sigma_crit is found
    by a numerical root-find on the *exact* skewness expression rather than the
    transcendental "closed form".
  * Optional rho-correlation between ln(K) and ln(I) inside a segment.
  * Optional upper truncation of the per-unit spend X (hard cap), with
    closed-form truncated-lognormal moments.

Every closed form here is cross-checked against numerical integration / brute
sampling in test_tam_core.py. This module is the ground truth that the Monte
Carlo engine (Phase 2) will be validated against.

Units convention: X is *monthly USD spend per unit*. Keep mu_K consistent with
that (a monthly share of monthly income, or fold the /12 into mu_K explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "SegmentParams",
    "CountParams",
    "lognormal_moments",
    "truncated_lognormal_moments",
    "segment_unit_moments",
    "compound_moments",
    "theta",
    "tam_expected",
    "tam_variance",
    "tam_skewness",
    "sigma_crit_enterprise_dominated",
    "sigma_crit_consumer_anchored",
    "sigma_crit_numeric",
]


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentParams:
    """Dual-lognormal parameters for one segment's per-unit monthly spend X = K * I.

    X is lognormal with
        mu_X    = mu_I + mu_K
        sig_X^2 = sig_I^2 + sig_K^2 + 2 * rho * sig_I * sig_K

    Parameters
    ----------
    mu_I, sigma_I : log-mean / log-sd of the income (or firm-scale) factor I.
    mu_K, sigma_K : log-mean / log-sd of the token-penetration factor K.
    rho           : correlation of ln(I), ln(K). 0 reproduces the document.
                    Negative for consumers (Engel curve), positive for firms.
    cap           : optional hard upper bound on X (USD/month). None = no cap.
    """

    mu_I: float
    sigma_I: float
    mu_K: float
    sigma_K: float
    rho: float = 0.0
    cap: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma_I < 0 or self.sigma_K < 0:
            raise ValueError("sigma_I and sigma_K must be non-negative.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("cap must be positive if provided.")

    @property
    def mu_X(self) -> float:
        return self.mu_I + self.mu_K

    @property
    def var_X_log(self) -> float:
        """Variance of ln(X) (the lognormal shape parameter sigma_X^2)."""
        v = (
            self.sigma_I**2
            + self.sigma_K**2
            + 2.0 * self.rho * self.sigma_I * self.sigma_K
        )
        # numerical guard: correlation arithmetic can dip microscopically below 0
        return max(v, 0.0)

    @property
    def sigma_X(self) -> float:
        return np.sqrt(self.var_X_log)


@dataclass(frozen=True)
class CountParams:
    """First three central moments of the adopting-population count N.

    Phase 1 takes these as given. Phase 0/2 will derive them from a
    Binomial-Beta (or similar) adoption model; the moments plug in here
    unchanged.
    """

    mean: float            # E[N]
    var: float             # Var(N)
    mu3: float = 0.0        # third central moment E[(N-E[N])^3]

    def __post_init__(self) -> None:
        if self.mean < 0 or self.var < 0:
            raise ValueError("E[N] and Var(N) must be non-negative.")


# ---------------------------------------------------------------------------
# Lognormal moments (raw and truncated)
# ---------------------------------------------------------------------------

def lognormal_moments(mu: float, sigma: float) -> tuple[float, float, float]:
    """Return (E[X], Var(X), mu3(X)) for X ~ Lognormal(mu, sigma^2).

    mu3 is the third *central* moment.
    """
    s2 = sigma**2
    mean = np.exp(mu + s2 / 2.0)
    var = (np.exp(s2) - 1.0) * np.exp(2.0 * mu + s2)
    # third central moment of a lognormal:
    #   mu3 = (e^{s2} + 2) * sqrt(e^{s2} - 1) * (Var)^{3/2}
    w = np.exp(s2)
    mu3 = (w + 2.0) * np.sqrt(w - 1.0) * var**1.5
    return mean, var, mu3


def truncated_lognormal_moments(
    mu: float, sigma: float, cap: float
) -> tuple[float, float, float]:
    """Raw moments of X ~ Lognormal(mu, sigma^2) truncated to (0, cap].

    Uses the standard result: for a lognormal, the k-th raw moment over the
    truncation is
        E[X^k ; X<=cap] = exp(k*mu + k^2 sigma^2 / 2) * Phi(d_k)
    with d_k = (ln(cap) - mu - k sigma^2) / sigma, and the normalising mass is
    Phi(d_0), d_0 = (ln(cap) - mu)/sigma. Returns (E, Var, mu3) of the
    *conditional* distribution X | X <= cap.
    """
    if sigma == 0.0:
        x = np.exp(mu)
        x = min(x, cap)
        return x, 0.0, 0.0

    ln_cap = np.log(cap)
    d0 = (ln_cap - mu) / sigma
    Z = norm.cdf(d0)            # P(X <= cap)
    if Z <= 0.0:
        # cap is far in the left tail; degenerate. Fall back to the cap value.
        return cap, 0.0, 0.0

    def raw(k: int) -> float:
        dk = (ln_cap - mu - k * sigma**2) / sigma
        return np.exp(k * mu + (k * sigma) ** 2 / 2.0) * norm.cdf(dk) / Z

    m1 = raw(1)
    m2 = raw(2)
    m3 = raw(3)
    mean = m1
    var = m2 - m1**2
    # third central moment from raw moments
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    return mean, var, mu3


def segment_unit_moments(seg: SegmentParams) -> tuple[float, float, float]:
    """(E[X], Var(X), mu3(X)) for one segment's per-unit spend, honoring cap."""
    if seg.cap is None:
        return lognormal_moments(seg.mu_X, seg.sigma_X)
    return truncated_lognormal_moments(seg.mu_X, seg.sigma_X, seg.cap)


# ---------------------------------------------------------------------------
# Compound-sum moments (Wald + third-cumulant expansion)
# ---------------------------------------------------------------------------

def compound_moments(
    seg: SegmentParams, counts: CountParams
) -> tuple[float, float, float]:
    """First three central moments of a compound sum S = sum_{j=1}^N X_j.

    Returns (E[S], Var(S), mu3(S)) using
        E[S]   = E[N] E[X]
        Var(S) = E[N] Var(X) + (E[X])^2 Var(N)
        mu3(S) = E[N] mu3(X) + 3 Var(N) E[X] Var(X) + mu3(N) (E[X])^3
    (X_j i.i.d., independent of N.)
    """
    ex, vx, m3x = segment_unit_moments(seg)
    en, vn, m3n = counts.mean, counts.var, counts.mu3

    e_s = en * ex
    var_s = en * vx + ex**2 * vn
    mu3_s = en * m3x + 3.0 * vn * ex * vx + m3n * ex**3
    return e_s, var_s, mu3_s


# ---------------------------------------------------------------------------
# Whole-market aggregates
# ---------------------------------------------------------------------------

def theta(cons: SegmentParams, ent: SegmentParams) -> float:
    """Structural scale factor E[X_e] / E[X_i] (honors caps)."""
    ex_i, _, _ = segment_unit_moments(cons)
    ex_e, _, _ = segment_unit_moments(ent)
    return ex_e / ex_i


def tam_expected(
    cons: SegmentParams, cons_n: CountParams,
    ent: SegmentParams, ent_n: CountParams,
) -> float:
    """E[TAM] = E[TAM_i] + E[TAM_e]."""
    e_i, _, _ = compound_moments(cons, cons_n)
    e_e, _, _ = compound_moments(ent, ent_n)
    return e_i + e_e


def tam_variance(
    cons: SegmentParams, cons_n: CountParams,
    ent: SegmentParams, ent_n: CountParams,
    cross_cov: float = 0.0,
) -> float:
    """Var(TAM) = Var(TAM_i) + Var(TAM_e) + 2*cross_cov.

    cross_cov defaults to 0 (the document's independence assumption). A common
    macro factor (Phase 5) would supply a positive value here.
    """
    _, v_i, _ = compound_moments(cons, cons_n)
    _, v_e, _ = compound_moments(ent, ent_n)
    return v_i + v_e + 2.0 * cross_cov


def tam_skewness(
    cons: SegmentParams, cons_n: CountParams,
    ent: SegmentParams, ent_n: CountParams,
) -> float:
    """Exact skewness gamma = mu3(TAM) / Var(TAM)^{3/2} under segment independence.

    (Third central moments add across independent segments; this is the
    quantity Section 6 approximates.)
    """
    _, v_i, m3_i = compound_moments(cons, cons_n)
    _, v_e, m3_e = compound_moments(ent, ent_n)
    var = v_i + v_e
    mu3 = m3_i + m3_e
    return mu3 / var**1.5


# ---------------------------------------------------------------------------
# Section 6.2: critical enterprise sigma
# ---------------------------------------------------------------------------

def sigma_crit_enterprise_dominated(e_n_e: float, epsilon: float) -> float:
    """Closed form, enterprise-dominated regime (document's first case, correct):

        sigma_crit = sqrt( (1/3) * ln( epsilon^2 * E[N_e] ) )

    Valid only when epsilon^2 * E[N_e] > 1; raises otherwise.
    """
    arg = epsilon**2 * e_n_e
    if arg <= 1.0:
        raise ValueError(
            "epsilon^2 * E[N_e] must exceed 1 for a real root "
            f"(got {arg:.4g}). The asymptotic regime does not apply."
        )
    return np.sqrt(np.log(arg) / 3.0)


def sigma_crit_consumer_anchored(
    e_n_i: float, e_n_e: float, sigma_i: float, theta_med: float, epsilon: float
) -> float:
    """Closed form, consumer-anchored regime (document's second case, CORRECTED).

    The document gives (1/2) ln(...) with no outer root, which is dimensionally
    wrong. Re-deriving from the same approximation:

        E[N_e] e^{3 s_e^2} theta^3 / ( E[N_i] (e^{s_i^2}-1) )^{3/2} <= epsilon

    solving for s_e gives

        sigma_crit = sqrt( (1/6) * ln( epsilon^2 (E[N_i])^3 (e^{s_i^2}-1)^3
                                       / ( E[N_e]^2 theta^6 ) ) )

    NOTE: theta itself depends on s_e. To use this *closed* form you must pass a
    sigma-independent theta (e.g. a ratio of medians, theta_med = e^{mu_Xe-mu_Xi}).
    For the self-consistent answer use sigma_crit_numeric().
    """
    num = epsilon**2 * e_n_i**3 * (np.exp(sigma_i**2) - 1.0) ** 3
    den = e_n_e**2 * theta_med**6
    arg = num / den
    if arg <= 1.0:
        raise ValueError(
            f"Argument of log must exceed 1 for a real root (got {arg:.4g})."
        )
    return np.sqrt(np.log(arg) / 6.0)


def sigma_crit_numeric(
    cons: SegmentParams,
    cons_n: CountParams,
    ent_template: SegmentParams,
    ent_n: CountParams,
    epsilon: float,
    bracket: Optional[tuple[float, float]] = None,
) -> float:
    """Self-consistent critical enterprise sigma_X via root-find on EXACT skewness.

    Solves gamma_TAM(sigma_X,e) = epsilon for the enterprise log-sd, holding all
    other parameters fixed. This sidesteps the transcendental-theta problem
    entirely and uses the exact moments (no heavy-tail approximation), so it is
    valid across regimes.

    `ent_template` supplies every enterprise parameter except the shape that is
    being solved for; the solver perturbs sigma_X,e by adjusting sigma_K,e so
    that var_X_log hits the target (mu_X,e and E[N_e] are held via mu, rho).

    Returns the sigma_X,e (log-sd of X_e) at which gamma == epsilon. Raises if
    no sign change in the bracket.
    """
    # We vary the *total* sigma_X,e directly by overriding sigma_K so that
    # var_X_log == target^2 with the template's sigma_I and rho held fixed.
    sigI = ent_template.sigma_I
    rho = ent_template.rho

    # Feasible floor: with sigma_K >= 0, the smallest achievable sigma_X is
    # sqrt(sigI^2 (1-rho^2)) (the minimum of sigI^2 + sigK^2 + 2 rho sigI sigK
    # over sigK >= 0 when rho < 0 is at sigK = -rho sigI; when rho >= 0 the min
    # is at sigK = 0, giving sigI). Bracket strictly above that floor.
    if rho < 0:
        sig_floor = sigI * np.sqrt(max(1.0 - rho**2, 0.0))
    else:
        sig_floor = sigI
    if bracket is None:
        bracket = (sig_floor + 1e-4, sig_floor + 5.0)

    def make_ent(sig_x: float) -> SegmentParams:
        # solve sigma_K from: sig_x^2 = sigI^2 + sigK^2 + 2 rho sigI sigK
        # => sigK^2 + 2 rho sigI sigK + (sigI^2 - sig_x^2) = 0
        a, b, c = 1.0, 2.0 * rho * sigI, sigI**2 - sig_x**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            raise ValueError(
                f"No real sigma_K reproduces sigma_X={sig_x:.4g} "
                f"with sigma_I={sigI:.4g}, rho={rho:.4g}."
            )
        sigK = (-b + np.sqrt(disc)) / (2 * a)
        if sigK < 0:
            sigK = (-b - np.sqrt(disc)) / (2 * a)
        return SegmentParams(
            mu_I=ent_template.mu_I,
            sigma_I=sigI,
            mu_K=ent_template.mu_K,
            sigma_K=max(sigK, 0.0),
            rho=rho,
            cap=ent_template.cap,
        )

    def f(sig_x: float) -> float:
        ent = make_ent(sig_x)
        return tam_skewness(cons, cons_n, ent, ent_n) - epsilon

    lo, hi = bracket
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        raise ValueError(
            "No sign change of (gamma - epsilon) in the bracket "
            f"[{lo}, {hi}]: f(lo)={flo:.4g}, f(hi)={fhi:.4g}. "
            "Widen the bracket or check epsilon."
        )
    return brentq(f, lo, hi, xtol=1e-8)
