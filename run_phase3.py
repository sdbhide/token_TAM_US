"""
run_phase3.py
=============
Production Phase 3 run: tornado, Sobol attribution for E[TAM] and tail
exceedance, scenario grid over (theta, sigma_X_e, E[N_e]) with the gamma =
epsilon convergence frontier, and macro-constraint conditioning of the
Phase 2 posterior.

Run:  python3 run_phase3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tam_sensitivity as ts
from tam_mc import MCConfig, run_mc

B = 1e9
OUT_TORNADO = "tam_phase3_tornado.png"
OUT_SOBOL = "tam_phase3_sobol.png"
OUT_GRID = "tam_phase3_scenario_grid.png"


def main():
    # ===================================================================
    # 1. TORNADO
    # ===================================================================
    rows = ts.tornado()
    y0 = rows[0]["y_central"]
    print("=" * 68)
    print(f"TORNADO - E[TAM] (central = ${y0/B:.1f}B), swing = |high - low|")
    print("=" * 68)
    for r in rows:
        print(f"  {r['param']:<10} low=${r['y_low']/B:7.1f}B  "
              f"high=${r['y_high']/B:7.1f}B  swing=${r['swing']/B:7.1f}B")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [r["param"] for r in rows][::-1]
    lo = np.array([r["y_low"] for r in rows])[::-1] / B
    hi = np.array([r["y_high"] for r in rows])[::-1] / B
    yc = y0 / B
    ypos = np.arange(len(names))
    ax.barh(ypos, hi - yc, left=yc, color="tab:red", alpha=0.75,
            label="param at prior high")
    ax.barh(ypos, lo - yc, left=yc, color="tab:blue", alpha=0.75,
            label="param at prior low")
    ax.axvline(yc, color="k", lw=1)
    ax.set_yticks(ypos, names)
    ax.set_xlabel("E[TAM] ($B/yr)")
    ax.set_title("Tornado - one-at-a-time over Phase 0 prior ranges")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT_TORNADO, dpi=150)
    print(f"-> {OUT_TORNADO}")

    # ===================================================================
    # 2. SOBOL - level QoI and tail-exceedance QoIs
    # ===================================================================
    print("\n" + "=" * 68)
    print("SOBOL INDICES (Saltelli design, n_base=2^14; validated on Ishigami)")
    print("=" * 68)

    qois = {
        "E[TAM]": ts.e_tam_vectorized,
        "P(TAM > $50B)": lambda P: (ts.e_tam_vectorized(P) > 50e9).astype(float),
        "P(TAM > $100B)": lambda P: (ts.e_tam_vectorized(P) > 100e9).astype(float),
    }
    results = {}
    for label, q in qois.items():
        res = ts.sobol_indices(q, n_base=2**14, seed=13, n_boot=400)
        results[label] = res
        print(f"\n--- QoI: {label}")
        print(res.table())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (label, res) in zip(axes, results.items()):
        order = np.argsort(res.ST)
        ypos = np.arange(len(order))
        ax.barh(ypos - 0.18, res.S1[order], height=0.36, label="S1 (first order)",
                color="tab:blue", alpha=0.85)
        ax.barh(ypos + 0.18, res.ST[order], height=0.36, label="ST (total)",
                color="tab:orange", alpha=0.85)
        # bootstrap CI whiskers on ST
        ax.errorbar(res.ST[order], ypos + 0.18,
                    xerr=[res.ST[order] - res.ST_ci[order, 0],
                          res.ST_ci[order, 1] - res.ST[order]],
                    fmt="none", ecolor="k", lw=0.9, capsize=2)
        ax.set_yticks(ypos, [res.names[j] for j in order])
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(loc="lower right")
    fig.suptitle("Sobol attribution - what drives the level vs the tail")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_SOBOL, dpi=150)
    print(f"\n-> {OUT_SOBOL}")

    # ===================================================================
    # 3. SCENARIO GRID over (theta, sigma_X_e, E[N_e])
    # ===================================================================
    thetas = np.geomspace(5, 5000, 60)
    sigmas = np.linspace(1.0, 3.0, 60)
    n_es = [1e6, 5e6, 6.3e6]
    grid = ts.scenario_grid(thetas, sigmas, n_es)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    eps_levels = [0.1, 0.5]
    for k, (ax, n_e) in enumerate(zip(axes, n_es)):
        pc = ax.pcolormesh(grid["thetas"], grid["sigma_xes"],
                           np.log10(grid["gamma"][k]), cmap="viridis",
                           shading="auto")
        cs = ax.contour(grid["thetas"], grid["sigma_xes"], grid["gamma"][k],
                        levels=eps_levels, colors=["w", "r"], linewidths=1.6)
        ax.clabel(cs, fmt={0.1: "γ=0.1", 0.5: "γ=0.5"}, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\theta$ (mean-ratio scale gap)")
        ax.set_title(f"E[N_e] = {n_e/1e6:.1f}M firms")
        if k == 0:
            ax.set_ylabel(r"$\sigma_{X,e}$ (enterprise concentration)")
        fig.colorbar(pc, ax=ax, label=r"$\log_{10}\gamma_{TAM}$")
    fig.suptitle(r"Scenario grid - skewness $\gamma_{TAM}$ over "
                 r"$(\theta, \sigma_{X,e}, E[N_e])$; contours = "
                 r"$\sigma_{crit}$ frontier (theory §7), consumer segment at "
                 "Phase 0 central")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_GRID, dpi=150)
    print(f"-> {OUT_GRID}")

    # where does the Phase 0 central calibration sit on this map?
    from tam_mc import central_model_draw
    import tam_core as tc
    draw = central_model_draw()
    th0 = tc.theta(draw.cons, draw.ent)
    sg0 = draw.ent.sigma_X
    print(f"\nPhase 0 central sits at theta={th0:.1f}, sigma_X_e={sg0:.2f}, "
          f"E[N_e]={draw.n_e/1e6:.1f}M")

    # ===================================================================
    # 4. MACRO-CONSTRAINT CONDITIONING of the Phase 2 posterior
    # ===================================================================
    print("\n" + "=" * 68)
    print("MACRO-CONSTRAINT CONDITIONING (current revenue window $14-22B)")
    print("=" * 68)
    res = run_mc(MCConfig(outer_draws=20_000, inner_reps=1, seed=20260612))
    w = ts.macro_constraint_weights(res.tam)   # weights on realized TAM ~ E[TAM|params]
    wn = w / w.sum()
    ess = w.sum() ** 2 / (w**2).sum()

    def wq(q):
        order = np.argsort(res.tam)
        cw = np.cumsum(wn[order])
        return res.tam[order][np.searchsorted(cw, q)]

    mean_u = res.tam.mean()
    mean_c = float(np.sum(wn * res.tam))
    print(f"  ESS = {ess:,.0f} / {res.tam.size:,} "
          f"({ess/res.tam.size:.1%} of draws compatible with constraint)")
    print(f"  {'':<14}{'unconditioned':>16}{'conditioned':>16}")
    print(f"  {'mean':<14}${mean_u/B:>14.1f}B ${mean_c/B:>14.1f}B")
    for q in (0.05, 0.50, 0.95):
        print(f"  {'P'+format(int(q*100),'02d'):<14}"
              f"${np.quantile(res.tam, q)/B:>14.1f}B ${wq(q)/B:>14.1f}B")
    print("\n  NOTE: conditioning collapses TAM uncertainty onto the observed-")
    print("  revenue window - appropriate if the model estimates the CURRENT")
    print("  market. For *equilibrium* TAM, condition only the parameters that")
    print("  should persist (sigma's, N's) and let mu_K grow; that scenario")
    print("  layer belongs to Phase 5 (price/adoption dynamics).")


if __name__ == "__main__":
    main()
