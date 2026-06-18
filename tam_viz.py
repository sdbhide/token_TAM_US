"""
tam_viz.py
==========
Phase 4: the five figures that earn their place. Each builder is a pure
function returning a matplotlib Figure, so they can be composed, re-themed, or
dropped into a report. A driver (run_phase4.py) wires them to data.

Figures
-------
(a) fig_distribution   - TAM on a log axis, mean/median/P5-P95 marked.
(b) fig_lorenz         - Lorenz curve + Gini of per-account enterprise spend;
                         annotates the share of TAM carried by the top 1%/0.1%.
(c) fig_skew_heatmap   - EMPIRICAL skewness over (E[N_e], sigma_X_e) from
                         simulation, with the corrected sigma_crit contour
                         (tam_core.sigma_crit_numeric) overlaid: a direct
                         simulation-vs-theory check of doc §7.
(d) fig_qq_convergence - normal QQ plots of simulated TAM at several adoption
                         scales, visualizing Anscombe convergence and its tail
                         failure.
(e) fig_waterfall      - additive decomposition of E[TAM] and Var(TAM) into
                         consumer vs enterprise layers.

All simulation reuses the validated Phase 2 engine (tam_mc); no new estimators
are introduced here, so Phase 2's acceptance guarantees carry over.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm

import tam_core as tc
import tam_mc as mc

B = 1e9

__all__ = [
    "gini",
    "lorenz_points",
    "sample_enterprise_accounts",
    "empirical_skew_grid",
    "fig_distribution",
    "fig_lorenz",
    "fig_skew_heatmap",
    "fig_qq_convergence",
    "fig_waterfall",
]


# ---------------------------------------------------------------------------
# Concentration helpers
# ---------------------------------------------------------------------------

def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values via the sorted-rank formula."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    # G = (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n   with i = 1..n
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * x) / (n * cum[-1])) - (n + 1.0) / n


def lorenz_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (population_fraction, cumulative_value_fraction) for a Lorenz
    curve, both starting at (0, 0)."""
    x = np.sort(np.asarray(x, dtype=float))
    cum = np.cumsum(x)
    cum_frac = np.concatenate([[0.0], cum / cum[-1]])
    pop_frac = np.linspace(0.0, 1.0, x.size + 1)
    return pop_frac, cum_frac


def sample_enterprise_accounts(
    seg: tc.SegmentParams, n_accounts: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw individual per-account enterprise spends (for concentration views).

    Uses exact lognormal (with cap if set). For the Lorenz/Gini figure we DO
    want per-account draws - that is the object being measured - so this is the
    one place full draws are appropriate. Capped at a few million accounts for
    memory; the Gini of a lognormal is sample-size-stable.
    """
    n = int(min(n_accounts, 3_000_000))
    x = np.exp(rng.normal(seg.mu_X, seg.sigma_X, size=n))
    if seg.cap is not None:
        x = np.minimum(x, seg.cap)
    return x


def empirical_skew_grid(
    n_es: np.ndarray,
    sigma_xes: np.ndarray,
    cons: tc.SegmentParams,
    n_i: int,
    ent_template: tc.SegmentParams,
    reps: int = 4000,
    seed: int = 1234,
) -> np.ndarray:
    """Empirical skewness of simulated TAM over an (E[N_e], sigma_X_e) grid.

    Consumer segment fixed; enterprise sigma_X varied by overriding sigma_K so
    that var_X_log == sigma_xe^2 (mu_X held, so E[X_e] varies with sigma - same
    convention as tam_core.sigma_crit_numeric). Uses the Phase 2 hybrid
    simulator for the enterprise total and the moment-matched draw for the
    consumer total. Returns array shape (len(sigma_xes), len(n_es)).
    """
    rng = np.random.default_rng(seed)
    sigI, rho = ent_template.sigma_I, ent_template.rho

    def make_ent(sig_x: float) -> tc.SegmentParams:
        a, b, c = 1.0, 2 * rho * sigI, sigI**2 - sig_x**2
        disc = b**2 - 4 * a * c
        sigK = (-b + np.sqrt(disc)) / 2.0 if disc >= 0 else 0.0
        return tc.SegmentParams(
            mu_I=ent_template.mu_I, sigma_I=sigI,
            mu_K=ent_template.mu_K, sigma_K=max(sigK, 0.0),
            rho=rho, cap=ent_template.cap)

    out = np.empty((len(sigma_xes), len(n_es)))
    cons_tot = mc.simulate_segment_total(rng, cons, n_i, reps, method="moment")
    for a_i, sx in enumerate(sigma_xes):
        ent = make_ent(sx)
        method = "hybrid" if ent.sigma_X >= 1.5 else "moment"
        for b_j, n_e in enumerate(n_es):
            ent_tot = mc.simulate_segment_total(
                rng, ent, int(n_e), reps, method=method)
            tam = cons_tot + ent_tot
            m = tam.mean()
            out[a_i, b_j] = np.mean((tam - m) ** 3) / tam.var() ** 1.5
    return out


# ---------------------------------------------------------------------------
# (a) Distribution
# ---------------------------------------------------------------------------

def fig_distribution(tam: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    lo, hi = tam.min() / B, tam.max() / B
    bins = np.geomspace(lo, hi, 90)
    ax.hist(tam / B, bins=bins, color="tab:green", alpha=0.7, edgecolor="none")

    mean, med = tam.mean() / B, np.median(tam) / B
    p5, p95 = np.quantile(tam, [0.05, 0.95]) / B
    ax.axvspan(p5, p95, color="grey", alpha=0.12, label="P5–P95")
    ax.axvline(med, color="k", lw=1.6, label=f"median ${med:.0f}B")
    ax.axvline(mean, color="tab:red", lw=1.8, ls="--",
               label=f"mean ${mean:.0f}B")
    for v, txt in [(p5, f"P5 ${p5:.0f}B"), (p95, f"P95 ${p95:.0f}B")]:
        ax.axvline(v, color="grey", lw=1.0, ls=":")
        ax.text(v, ax.get_ylim()[1] * 0.96, " " + txt, rotation=90,
                va="top", ha="left", fontsize=8, color="dimgrey")

    gap = (mean - med) / med
    ax.annotate(f"mean–median gap = {gap:.0%}\n(lognormal right-skew)",
                xy=(mean, ax.get_ylim()[1] * 0.55),
                xytext=(mean * 1.6, ax.get_ylim()[1] * 0.7),
                fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("TAM ($B/yr, log scale)")
    ax.set_ylabel("count")
    ax.set_title("(a) Posterior-predictive TAM distribution")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# (b) Lorenz / Gini
# ---------------------------------------------------------------------------

def fig_lorenz(spends: np.ndarray) -> Figure:
    pop, val = lorenz_points(spends)
    g = gini(spends)

    # top-share annotations
    s_sorted = np.sort(spends)[::-1]
    total = s_sorted.sum()
    def top_share(frac):
        k = max(int(len(s_sorted) * frac), 1)
        return s_sorted[:k].sum() / total

    t1, t01 = top_share(0.01), top_share(0.001)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="line of equality")
    ax.plot(pop, val, color="tab:orange", lw=2,
            label=f"enterprise spend (Gini = {g:.3f})")
    ax.fill_between(pop, val, pop, color="tab:orange", alpha=0.15)

    # mark the top 1%: population fraction 0.99 on the sorted-ascending axis
    ax.axvline(0.99, color="tab:purple", lw=1, ls=":")
    ax.text(0.985, 0.05,
            f"top 1% of accounts\ncarry {t1:.0%} of TAM",
            rotation=90, ha="right", va="bottom", fontsize=9,
            color="tab:purple")
    ax.set_xlabel("cumulative share of accounts (poorest → richest)")
    ax.set_ylabel("cumulative share of total spend")
    ax.set_title("(b) Concentration of enterprise token spend\n"
                 f"top 1% → {t1:.0%},  top 0.1% → {t01:.0%} of enterprise TAM")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# (c) Empirical-skew heatmap with sigma_crit contour
# ---------------------------------------------------------------------------

def fig_skew_heatmap(
    n_es: np.ndarray,
    sigma_xes: np.ndarray,
    skew_grid: np.ndarray,
    cons: tc.SegmentParams,
    n_i: int,
    ent_template: tc.SegmentParams,
    cons_n: tc.CountParams,
    ent_n_fn,
    epsilon: float = 0.5,
) -> Figure:
    """Heatmap of empirical skewness with the theoretical sigma_crit(E[N_e])
    frontier overlaid.

    ent_n_fn(n_e) -> CountParams for the enterprise segment at that E[N_e].
    """
    fig, ax = plt.subplots(figsize=(9.2, 6))
    pc = ax.pcolormesh(n_es / 1e6, sigma_xes, np.log10(np.clip(skew_grid, 1e-4, None)),
                       cmap="magma", shading="auto")
    fig.colorbar(pc, ax=ax, label=r"$\log_{10}$ (empirical skewness)")

    # theoretical sigma_crit per E[N_e] via the exact numeric solver
    sig_crit = []
    for n_e in n_es:
        try:
            s = tc.sigma_crit_numeric(
                cons, cons_n, ent_template, ent_n_fn(n_e), epsilon,
                bracket=(sigma_xes[0], sigma_xes[-1]))
        except ValueError:
            s = np.nan
        sig_crit.append(s)
    sig_crit = np.array(sig_crit)

    ax.plot(n_es / 1e6, sig_crit, color="cyan", lw=2.4,
            label=fr"theoretical $\sigma_{{crit}}$ ($\gamma=\epsilon={epsilon}$)")
    # empirical gamma=epsilon contour for comparison
    cs = ax.contour(n_es / 1e6, sigma_xes, skew_grid, levels=[epsilon],
                    colors="white", linewidths=1.4, linestyles="--")
    ax.clabel(cs, fmt={epsilon: f"empirical γ={epsilon}"}, fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$E[N_e]$ (enterprise accounts, millions)")
    ax.set_ylabel(r"$\sigma_{X,e}$ (enterprise concentration)")
    ax.set_title("(c) Empirical skewness vs theoretical "
                 r"$\sigma_{crit}$ frontier (validates §7)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# (d) QQ convergence
# ---------------------------------------------------------------------------

def fig_qq_convergence(
    cons: tc.SegmentParams,
    ent: tc.SegmentParams,
    scales: Sequence[float],
    base_n_i: int,
    base_n_e: int,
    reps: int = 6000,
    seed: int = 77,
) -> Figure:
    """Normal QQ plots of simulated TAM as adoption scales up by each factor in
    `scales` (counts multiplied, holding per-account distributions fixed).
    Anscombe convergence: points straighten toward the diagonal as scale grows;
    the upper tail is the last to comply."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(scales), figsize=(4.4 * len(scales), 4.4),
                             sharey=False)
    if len(scales) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scales):
        n_i, n_e = int(base_n_i * sc), int(base_n_e * sc)
        ti = mc.simulate_segment_total(rng, cons, n_i, reps, method="moment")
        te = mc.simulate_segment_total(rng, ent, n_e, reps, method="hybrid")
        tam = ti + te
        z = (tam - tam.mean()) / tam.std()
        q_theory = norm.ppf((np.arange(1, reps + 1) - 0.5) / reps)
        q_emp = np.sort(z)
        ax.plot(q_theory, q_emp, ".", ms=2.5, alpha=0.4, color="tab:blue")
        lim = [min(q_theory[0], q_emp[0]), max(q_theory[-1], q_emp[-1])]
        ax.plot(lim, lim, "r-", lw=1)
        g = np.mean((tam - tam.mean()) ** 3) / tam.var() ** 1.5
        ax.set_title(f"scale ×{sc:g}\n(E[N_e]={n_e/1e6:.2g}M, skew={g:.2f})",
                     fontsize=10)
        ax.set_xlabel("normal quantile")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("standardized TAM quantile")
    fig.suptitle("(d) Anscombe convergence - normal QQ of TAM vs adoption scale "
                 "(upper tail complies last)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
# (e) Waterfall decomposition
# ---------------------------------------------------------------------------

def fig_waterfall(
    cons: tc.SegmentParams, cons_n: tc.CountParams,
    ent: tc.SegmentParams, ent_n: tc.CountParams,
) -> Figure:
    """Additive decomposition of E[TAM] and Var(TAM) into segment layers,
    using the exact closed forms (tam_core).

    NOTE: the variance split reflects whatever count uncertainty is encoded in
    cons_n / ent_n. Pass deterministic counts (var=0) to isolate the pure
    spend-dispersion (Wald E[N]Var(X)) layer, or Beta-Binomial moments (§4) to
    include adoption-count risk. The driver passes the latter so the figure
    shows real per-realization risk."""
    e_i, v_i, _ = tc.compound_moments(cons, cons_n)
    e_e, v_e, _ = tc.compound_moments(ent, ent_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # E[TAM] waterfall
    parts = [("consumer", e_i / B, "tab:blue"),
             ("enterprise", e_e / B, "tab:orange")]
    base = 0.0
    for name, val, col in parts:
        ax1.bar(name, val, bottom=base, color=col, alpha=0.85)
        ax1.text(name, base + val / 2, f"${val:.1f}B\n({val/((e_i+e_e)/B):.0%})",
                 ha="center", va="center", fontsize=9)
        base += val
    ax1.bar("TOTAL", (e_i + e_e) / B, color="tab:green", alpha=0.85)
    ax1.text("TOTAL", (e_i + e_e) / B / 2, f"${(e_i+e_e)/B:.1f}B",
             ha="center", va="center", fontsize=9, weight="bold")
    ax1.set_ylabel("E[TAM] ($B/yr)")
    ax1.set_title("E[TAM]: linear additive (mean scales with count)")
    ax1.grid(alpha=0.3, axis="y")

    # Var(TAM) panel - report as the more interpretable SD-contribution sqrt(Var)
    # in $B, with the variance SHARE annotated. Consumer variance is ~5 orders
    # of magnitude below enterprise (the whole point), so a linear bar makes it
    # invisible; we annotate shares explicitly and use a log y-axis.
    sd_i, sd_e = np.sqrt(v_i) / B, np.sqrt(v_e) / B
    sd_tot = np.sqrt(v_i + v_e) / B
    var_tot = v_i + v_e
    bars = [("consumer", sd_i, v_i / var_tot, "tab:blue"),
            ("enterprise", sd_e, v_e / var_tot, "tab:orange"),
            ("TOTAL", sd_tot, 1.0, "tab:green")]
    for name, sd, share, col in bars:
        ax2.bar(name, sd, color=col, alpha=0.85)
        ax2.text(name, sd, f" SD ${sd:.1f}B\n {share:.1%} of Var",
                 ha="center", va="bottom", fontsize=9,
                 weight="bold" if name == "TOTAL" else "normal")
    ax2.set_yscale("log")
    ax2.set_ylim(min(sd_i, sd_e) * 0.3, sd_tot * 3)
    ax2.set_ylabel("SD contribution  sqrt(Var)  ($B/yr, log scale)")
    lead = "consumer" if v_i > v_e else "enterprise"
    ax2.set_title("Var(TAM) by segment "
                  f"({lead} layer leads with count uncertainty)\n"
                  f"consumer = {v_i/var_tot:.0%}, enterprise = {v_e/var_tot:.0%} "
                  "of variance")
    ax2.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("(e) Segment decomposition of E[TAM] and Var(TAM) "
                 "(at Phase 0 central calibration)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
