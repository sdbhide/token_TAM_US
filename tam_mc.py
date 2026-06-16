"""
tam_mc.py
=========
Phase 2: Two-level Monte Carlo engine for the dual-segment AI-inference TAM model.

Architecture
------------
OUTER LOOP   samples model parameters from the Phase 0 priors (parameter
             uncertainty: "which world are we in?").
INNER LOOP   given parameters, simulates one realization of the population's
             total spend (population/aleatory noise: "what happens in that
             world?").

The unconditional TAM distribution mixes both layers; `inner_reps > 1` enables
a variance decomposition (parameter vs. population uncertainty).

Population-simulation strategy (per the Phase 2 design notes)
-------------------------------------------------------------
* CONSUMER segment (N ~ 1e8): never draw 1e8 lognormals. The segment total is
  CLT-friendly, so we draw it from a SHIFTED-GAMMA matched to the exact first
  three compound-sum moments from tam_core (mean, variance, AND skewness —
  strictly better than a plain normal approximation).
* ENTERPRISE segment (N ~ 5e6, heavy tail): HYBRID STRATIFIED simulation.
  The X_e distribution is split at a high quantile q (default 99.9th pct):
    - TAIL stratum (top ~0.1%, ~5k firms): simulated EXACTLY — count drawn
      Binomial(N_e, 1-q), spends drawn from the conditional lognormal above
      the threshold via inverse-CDF sampling. This is where TAM risk lives.
    - BODY stratum (bounded above by the threshold): its sum is extremely
      CLT-friendly; drawn from a shifted-gamma matched to the body's exact
      first three (lower-truncated lognormal) moments times the body count.
  This concentrates random draws where the tail matters — stratification as
  variance reduction — and reduces per-replication cost from O(N_e) to O(N_e/1000).

Units: ANNUAL USD, matching Phase 0 (phase_0_corrected.py). NOTE: the v4 theory
doc pinned monthly; annual is now the project convention — amend §2.1 of the doc.

Validation contract (§9 of theory doc): simulated mean/variance/skewness at
FIXED parameters must match tam_core closed forms within Monte Carlo error.
Run `python3 test_tam_mc.py` for the acceptance suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
from scipy.stats import norm

import tam_core as tc

__all__ = [
    "Prior",
    "PHASE0_PRIORS",
    "ModelDraw",
    "sample_model_draw",
    "tail_lognormal_moments",
    "shifted_gamma_sample",
    "simulate_segment_total",
    "MCConfig",
    "MCResult",
    "run_mc",
    "running_mean_se",
    "bootstrap_quantile_ci",
]


# ---------------------------------------------------------------------------
# 1. Priors (Phase 0 parameter ranges -> sampling distributions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prior:
    """A 1-D prior over a scalar parameter.

    kind:
      "uniform"    : U(low, high)
      "triangular" : Triangular(low, mode, high) — use when Phase 0 supplies a
                     central anchor in addition to a range.
      "fixed"      : degenerate at `mode` (for acceptance tests / scenarios).
    """
    low: float
    high: float
    mode: Optional[float] = None
    kind: Literal["uniform", "triangular", "fixed"] = "uniform"

    def sample(self, rng: np.random.Generator, size=None):
        if self.kind == "fixed":
            m = self.mode if self.mode is not None else 0.5 * (self.low + self.high)
            return np.full(size, m) if size else m
        if self.kind == "triangular":
            m = self.mode if self.mode is not None else 0.5 * (self.low + self.high)
            return rng.triangular(self.low, m, self.high, size=size)
        return rng.uniform(self.low, self.high, size=size)

    @property
    def central(self) -> float:
        return self.mode if self.mode is not None else 0.5 * (self.low + self.high)


# Ranges transcribed from phase_0_corrected.py `parameters` dict.
# Modes = Phase 0 central anchors where documented (FIX-2: mu_K_i=-6.52,
# FIX-3: mu_K_e=-6.74; income params from the CPS calibration; others midpoint).
PHASE0_PRIORS: dict[str, Prior] = {
    # consumer
    "N_i":       Prior(80e6, 150e6, mode=100e6, kind="triangular"),
    "mu_I_i":    Prior(10.50, 10.62, mode=10.6086, kind="triangular"),
    "sigma_I_i": Prior(0.82, 0.93, mode=0.8763, kind="triangular"),
    "mu_K_i":    Prior(-7.21, -5.31, mode=-6.52, kind="triangular"),
    "sigma_K_i": Prior(0.5, 1.0, mode=0.70, kind="triangular"),
    # enterprise
    "N_e":       Prior(5.0e6, 6.3e6, mode=5.0e6, kind="triangular"),
    "mu_I_e":    Prior(11.5, 13.8, mode=12.0, kind="triangular"),
    "sigma_I_e": Prior(1.5, 2.5, mode=1.8, kind="triangular"),
    "mu_K_e":    Prior(-7.5, -5.5, mode=-6.74, kind="triangular"),
    "sigma_K_e": Prior(0.8, 1.5, mode=1.2, kind="triangular"),
}


@dataclass(frozen=True)
class ModelDraw:
    """One outer-loop draw: a fully specified world."""
    cons: tc.SegmentParams
    ent: tc.SegmentParams
    n_i: int
    n_e: int


def sample_model_draw(
    rng: np.random.Generator,
    priors: dict[str, Prior] = PHASE0_PRIORS,
    rho_i: float = 0.0,
    rho_e: float = 0.0,
    cap_e: Optional[float] = None,
) -> ModelDraw:
    """Sample one parameter vector from the priors.

    rho_* and cap_e default to the Phase 0 configuration (no correlation, no
    cap); pass non-defaults for the §3.1/§3.2 extensions.
    """
    p = {k: v.sample(rng) for k, v in priors.items()}
    cons = tc.SegmentParams(
        mu_I=p["mu_I_i"], sigma_I=p["sigma_I_i"],
        mu_K=p["mu_K_i"], sigma_K=p["sigma_K_i"], rho=rho_i)
    ent = tc.SegmentParams(
        mu_I=p["mu_I_e"], sigma_I=p["sigma_I_e"],
        mu_K=p["mu_K_e"], sigma_K=p["sigma_K_e"], rho=rho_e, cap=cap_e)
    return ModelDraw(cons=cons, ent=ent, n_i=int(p["N_i"]), n_e=int(p["N_e"]))


def central_model_draw(
    priors: dict[str, Prior] = PHASE0_PRIORS, **kw
) -> ModelDraw:
    """The Phase 0 central calibration as a fixed ModelDraw (for acceptance tests)."""
    fixed = {k: Prior(v.low, v.high, mode=v.central, kind="fixed")
             for k, v in priors.items()}
    return sample_model_draw(np.random.default_rng(0), fixed, **kw)


# ---------------------------------------------------------------------------
# 2. Distributional helpers
# ---------------------------------------------------------------------------

def tail_lognormal_moments(
    mu: float, sigma: float, threshold: float
) -> tuple[float, float, float, float]:
    """Moments of X ~ LN(mu, sigma^2) conditional on X > threshold.

    Returns (p_tail, E, Var, mu3) where p_tail = P(X > threshold).
    Complement of tam_core.truncated_lognormal_moments (which conditions on
    X <= cap). Closed form via E[X^k; X>t] = e^{k mu + k^2 s^2/2} (1 - Phi(d_k)).
    """
    ln_t = np.log(threshold)
    d0 = (ln_t - mu) / sigma
    p_tail = norm.sf(d0)
    if p_tail <= 0.0:
        return 0.0, threshold, 0.0, 0.0

    def raw(k: int) -> float:
        dk = (ln_t - mu - k * sigma**2) / sigma
        return np.exp(k * mu + (k * sigma) ** 2 / 2.0) * norm.sf(dk) / p_tail

    m1, m2, m3 = raw(1), raw(2), raw(3)
    var = m2 - m1**2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    return p_tail, m1, var, mu3


def shifted_gamma_sample(
    rng: np.random.Generator, mean: float, var: float, mu3: float, size: int
) -> np.ndarray:
    """Sample from a 3-parameter (shifted) gamma matching (mean, var, mu3).

    Gamma(shape k, scale th) + shift c has moments:
        mean = c + k*th,  var = k*th^2,  mu3 = 2*k*th^3
    =>  th = mu3 / (2 var),  k = var / th^2,  c = mean - k*th.

    Matches the first three moments exactly when mu3 > 0; falls back to a
    normal when skewness is negligible or non-positive (lognormal sums always
    have mu3 > 0, so the fallback only triggers on numerical underflow).
    """
    if var <= 0.0:
        return np.full(size, mean)
    skew = mu3 / var**1.5
    if skew <= 1e-8:
        return rng.normal(mean, np.sqrt(var), size=size)
    th = mu3 / (2.0 * var)
    k = var / th**2
    c = mean - k * th
    return c + rng.gamma(shape=k, scale=th, size=size)


# ---------------------------------------------------------------------------
# 3. Segment-total simulators
# ---------------------------------------------------------------------------

def _segment_total_moment_matched(
    rng: np.random.Generator, seg: tc.SegmentParams, n: int, reps: int
) -> np.ndarray:
    """Segment totals via shifted-gamma on exact compound moments (CLT path).

    Used for the consumer segment. Count noise: with N in the 1e8 range and
    parameter uncertainty handled by the outer loop, Binomial-style count noise
    is O(sqrt(N)) ~ 1e4 — utterly negligible against spend dispersion — so N is
    treated as fixed within a replication (deterministic-count CountParams).
    """
    counts = tc.CountParams(mean=float(n), var=0.0, mu3=0.0)
    m, v, m3 = tc.compound_moments(seg, counts)
    return shifted_gamma_sample(rng, m, v, m3, reps)


def _segment_total_hybrid(
    rng: np.random.Generator,
    seg: tc.SegmentParams,
    n: int,
    reps: int,
    tail_q: float = 0.999,
    max_tail_per_rep: int = 20_000,
) -> np.ndarray:
    """Hybrid stratified totals: exact tail draws + moment-matched body sum.

    Split X at its tail_q quantile t:
      BODY  (X <= t): sum ~ shifted-gamma matched to N_body * (truncated moments)
      TAIL  (X > t) : N_tail ~ Binomial(n, 1-tail_q); spends drawn exactly from
                      the conditional lognormal via inverse-CDF:
                          X = exp(mu + sigma * Phi^{-1}(U)), U ~ U(tail_q, 1)
    If seg.cap is set and cap <= t, there is no tail stratum; everything goes
    through the (cap-truncated) body path.

    Adaptive tail: when n*(1-tail_q) would exceed max_tail_per_rep, tail_q is
    raised so the expected exact-tail count per rep stays bounded. This keeps
    cost ~O(reps * max_tail_per_rep) regardless of n, while still simulating
    the heaviest mass exactly. The body shifted-gamma (which matches three
    exact moments) absorbs the now-larger body stratum with no accuracy loss
    at these scales, since that stratum is overwhelmingly CLT-compliant.
    """
    mu, sig = seg.mu_X, seg.sigma_X

    # adapt the split so E[tail firms per rep] = n*(1-tail_q) <= max_tail_per_rep
    if n * (1.0 - tail_q) > max_tail_per_rep:
        tail_q = 1.0 - max_tail_per_rep / n

    t = np.exp(mu + sig * norm.ppf(tail_q))  # tail threshold

    if seg.cap is not None and seg.cap <= t:
        body = tc.truncated_lognormal_moments(mu, sig, seg.cap)
        m, v, m3 = (n * body[0], n * body[1], n * body[2])
        return shifted_gamma_sample(rng, m, v, m3, reps)

    # body moments: lognormal conditioned on X <= t
    mb, vb, m3b = tc.truncated_lognormal_moments(mu, sig, t)

    p_tail = 1.0 - tail_q
    n_tail = rng.binomial(n, p_tail, size=reps)

    # body counts and sums (vectorized over reps). Shifted-gamma moments scale
    # linearly in n_body, and the gamma scale th = m3b/(2 vb) is count-free, so
    # the whole body stratum vectorizes in a single rng.gamma call:
    n_body = n - n_tail
    th = m3b / (2.0 * vb)
    k_r = n_body * vb / th**2
    c_r = n_body * mb - k_r * th
    body_totals = c_r + rng.gamma(shape=k_r, scale=th)

    # tail sums: exact conditional-lognormal draws via inverse CDF, CHUNKED over
    # reps so peak memory stays bounded (~max_block doubles) regardless of
    # reps * E[n_tail]; without chunking, 40k reps x ~5k tail firms allocates
    # 2e8 doubles in one shot and OOMs.
    tail_totals = np.empty(reps)
    max_block = 5_000_000
    start = 0
    while start < reps:
        stop = start
        acc = 0
        while stop < reps and acc + n_tail[stop] <= max_block:
            acc += int(n_tail[stop])
            stop += 1
        stop = max(stop, start + 1)          # always make progress
        blk = n_tail[start:stop]
        tot = int(blk.sum())
        if tot > 0:
            u = rng.uniform(tail_q, 1.0, size=tot)
            x = np.exp(mu + sig * norm.ppf(u))
            if seg.cap is not None:
                x = np.minimum(x, seg.cap)
            idx = np.concatenate([[0], np.cumsum(blk)])
            csum = np.concatenate([[0.0], np.cumsum(x)])
            tail_totals[start:stop] = csum[idx[1:]] - csum[idx[:-1]]
        else:
            tail_totals[start:stop] = 0.0
        start = stop

    return body_totals + tail_totals


def simulate_segment_total(
    rng: np.random.Generator,
    seg: tc.SegmentParams,
    n: int,
    reps: int,
    method: Literal["auto", "moment", "hybrid", "exact"] = "auto",
    tail_q: float = 0.999,
) -> np.ndarray:
    """Simulate `reps` realizations of one segment's total annual spend.

    method:
      "moment" : shifted-gamma on compound moments (consumer default).
      "hybrid" : stratified exact-tail + moment-matched body (enterprise default).
      "exact"  : brute force — n lognormal draws per rep. Validation only.
      "auto"   : "hybrid" if sigma_X >= 1.5 (heavy tail), else "moment".
    """
    if method == "auto":
        method = "hybrid" if seg.sigma_X >= 1.5 else "moment"
    if method == "moment":
        return _segment_total_moment_matched(rng, seg, n, reps)
    if method == "hybrid":
        return _segment_total_hybrid(rng, seg, n, reps, tail_q=tail_q)
    # exact brute force
    totals = np.empty(reps)
    for r in range(reps):
        x = np.exp(rng.normal(seg.mu_X, seg.sigma_X, size=n))
        if seg.cap is not None:
            x = np.minimum(x, seg.cap)
        totals[r] = x.sum()
    return totals


# ---------------------------------------------------------------------------
# 4. Two-level driver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCConfig:
    outer_draws: int = 2000          # parameter-uncertainty draws
    inner_reps: int = 1              # population realizations per parameter draw
    seed: int = 20260612
    priors: dict[str, Prior] = field(default_factory=lambda: dict(PHASE0_PRIORS))
    rho_i: float = 0.0
    rho_e: float = 0.0
    cap_e: Optional[float] = None
    tail_q: float = 0.999
    consumer_method: str = "moment"
    enterprise_method: str = "hybrid"


@dataclass
class MCResult:
    tam: np.ndarray              # (outer*inner,) total TAM draws, USD/yr
    tam_i: np.ndarray            # consumer component
    tam_e: np.ndarray            # enterprise component
    outer_index: np.ndarray      # which outer draw each sample belongs to
    config: MCConfig

    # --- summaries -------------------------------------------------------
    def summary(self, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict:
        t = self.tam
        out = {
            "n_samples": t.size,
            "mean": t.mean(),
            "se_mean": t.std(ddof=1) / np.sqrt(t.size),
            "sd": t.std(ddof=1),
            "skew": float(np.mean((t - t.mean()) ** 3) / t.var() ** 1.5),
            "quantiles": {q: float(np.quantile(t, q)) for q in qs},
            "mean_i": self.tam_i.mean(),
            "mean_e": self.tam_e.mean(),
        }
        return out

    def variance_decomposition(self) -> Optional[dict]:
        """Law-of-total-variance split: parameter vs population uncertainty.

        Requires inner_reps > 1. Var(TAM) = Var_outer(E_inner) + E_outer(Var_inner).
        """
        if self.config.inner_reps < 2:
            return None
        g = self.outer_index
        R = g.max() + 1
        means = np.array([self.tam[g == r].mean() for r in range(R)])
        vars_ = np.array([self.tam[g == r].var(ddof=1) for r in range(R)])
        return {
            "var_parameter": float(means.var(ddof=1)),
            "var_population": float(vars_.mean()),
            "share_parameter": float(
                means.var(ddof=1) / (means.var(ddof=1) + vars_.mean())
            ),
        }


def run_mc(cfg: MCConfig = MCConfig()) -> MCResult:
    rng = np.random.default_rng(cfg.seed)
    R, M = cfg.outer_draws, cfg.inner_reps
    n_tot = R * M
    tam_i = np.empty(n_tot)
    tam_e = np.empty(n_tot)
    outer_index = np.repeat(np.arange(R), M)

    for r in range(R):
        draw = sample_model_draw(
            rng, cfg.priors, rho_i=cfg.rho_i, rho_e=cfg.rho_e, cap_e=cfg.cap_e)
        sl = slice(r * M, (r + 1) * M)
        tam_i[sl] = simulate_segment_total(
            rng, draw.cons, draw.n_i, M, method=cfg.consumer_method)
        tam_e[sl] = simulate_segment_total(
            rng, draw.ent, draw.n_e, M, method=cfg.enterprise_method,
            tail_q=cfg.tail_q)

    return MCResult(tam=tam_i + tam_e, tam_i=tam_i, tam_e=tam_e,
                    outer_index=outer_index, config=cfg)


# ---------------------------------------------------------------------------
# 5. Convergence diagnostics
# ---------------------------------------------------------------------------

def running_mean_se(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running mean and running standard error after each sample.

    Feed to a plot of mean +/- 2*SE vs n to visually confirm convergence.
    """
    n = np.arange(1, samples.size + 1)
    csum = np.cumsum(samples)
    rmean = csum / n
    csum2 = np.cumsum(samples**2)
    rvar = np.maximum(csum2 / n - rmean**2, 0.0) * n / np.maximum(n - 1, 1)
    rse = np.sqrt(rvar / n)
    return rmean, rse


def bootstrap_quantile_ci(
    samples: np.ndarray,
    qs=(0.05, 0.50, 0.95),
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 7,
) -> dict:
    """Percentile-bootstrap CIs for the requested quantiles of the TAM draws."""
    rng = np.random.default_rng(seed)
    n = samples.size
    boots = np.empty((n_boot, len(qs)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.quantile(samples[idx], qs)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        q: {
            "point": float(np.quantile(samples, q)),
            "ci_low": float(np.quantile(boots[:, j], lo)),
            "ci_high": float(np.quantile(boots[:, j], hi)),
        }
        for j, q in enumerate(qs)
    }
