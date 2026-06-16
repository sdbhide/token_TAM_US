"""
run_phase4.py
=============
Produce the five Phase 4 figures. Reuses the saved Phase 2 samples for the
distribution figure and the validated Phase 2 engine for the simulation-based
panels (Lorenz, skew heatmap, QQ).

Run:  python3 run_phase4.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import tam_core as tc
import tam_mc as mc
import tam_viz as tv

B = 1e9
FILES = {
    "a": "tam_phase4_a_distribution.png",
    "b": "tam_phase4_b_lorenz.png",
    "c": "tam_phase4_c_skew_heatmap.png",
    "d": "tam_phase4_d_qq_convergence.png",
    "e": "tam_phase4_e_waterfall.png",
}


def main():
    rng = np.random.default_rng(20260612)
    draw = mc.central_model_draw()             # Phase 0 central calibration
    cons, ent = draw.cons, draw.ent
    n_i, n_e = draw.n_i, draw.n_e
    cons_n = tc.CountParams(mean=float(n_i), var=0.0, mu3=0.0)
    ent_n = tc.CountParams(mean=float(n_e), var=0.0, mu3=0.0)

    # ---- (a) distribution: reuse Phase 2 samples if present, else simulate
    try:
        tam = np.load("tam_phase2_samples.npz")["tam"]
        print("(a) using saved Phase 2 samples")
    except FileNotFoundError:
        print("(a) Phase 2 samples not found — simulating")
        tam = mc.run_mc(mc.MCConfig(outer_draws=20000, inner_reps=1)).tam
    fig = tv.fig_distribution(tam)
    fig.savefig(FILES["a"], dpi=150)
    print(f"    -> {FILES['a']}")

    # ---- (b) Lorenz / Gini of per-account enterprise spend
    spends = tv.sample_enterprise_accounts(ent, n_e, rng)
    fig = tv.fig_lorenz(spends)
    fig.savefig(FILES["b"], dpi=150)
    g = tv.gini(spends)
    s = np.sort(spends)[::-1]
    print(f"(b) enterprise Gini={g:.3f}, top 1% share="
          f"{s[:max(len(s)//100,1)].sum()/s.sum():.1%}  -> {FILES['b']}")

    # ---- (c) empirical-skew heatmap with sigma_crit contour
    n_es = np.geomspace(5e3, 5e6, 28)          # span where CLT bites
    sigma_xes = np.linspace(ent.sigma_I + 0.02, 3.0, 26)
    print("(c) simulating empirical skew grid "
          f"({len(sigma_xes)}x{len(n_es)} cells)...")
    skew_grid = tv.empirical_skew_grid(
        n_es, sigma_xes, cons, n_i, ent, reps=4000, seed=4321)
    fig = tv.fig_skew_heatmap(
        n_es, sigma_xes, skew_grid, cons, n_i, ent, cons_n,
        ent_n_fn=lambda ne: tc.CountParams(mean=float(ne), var=0.0, mu3=0.0),
        epsilon=0.5)
    fig.savefig(FILES["c"], dpi=150)
    print(f"    -> {FILES['c']}")

    # ---- (d) QQ convergence across adoption scales
    fig = tv.fig_qq_convergence(
        cons, ent, scales=(0.001, 0.05, 1.0, 20.0),
        base_n_i=n_i, base_n_e=n_e, reps=6000)
    fig.savefig(FILES["d"], dpi=150)
    print(f"(d) -> {FILES['d']}")

    # ---- (e) waterfall decomposition
    # Variance layers reflect count uncertainty (theory §4 Beta-Binomial), not
    # just spend dispersion, so the figure shows real per-realization risk.
    # Concentration params are illustrative (not Phase 0): consumer adoption is
    # tightly estimable, enterprise less so. The qualitative finding — that the
    # consumer segment's enormous N amplifies even small adoption-rate
    # uncertainty into a variance layer rivaling enterprise — is robust to these.
    def betabinom_counts(pop, p_mean, conc):
        a, b = p_mean * conc, (1 - p_mean) * conc
        var = pop * p_mean * (1 - p_mean) * (1 + (pop - 1) / (a + b + 1))
        return tc.CountParams(mean=pop * p_mean, var=var, mu3=0.0)

    cons_n_bb = betabinom_counts(260e6, n_i / 260e6, conc=400.0)
    ent_n_bb = betabinom_counts(5.9e6, n_e / 5.9e6, conc=50.0)
    fig = tv.fig_waterfall(cons, cons_n_bb, ent, ent_n_bb)
    fig.savefig(FILES["e"], dpi=150)
    e_i, v_i, _ = tc.compound_moments(cons, cons_n_bb)
    e_e, v_e, _ = tc.compound_moments(ent, ent_n_bb)
    print(f"(e) E[TAM] enterprise share={e_e/(e_i+e_e):.0%}, "
          f"Var consumer/enterprise={v_i/(v_i+v_e):.0%}/{v_e/(v_i+v_e):.0%}  "
          f"-> {FILES['e']}")


if __name__ == "__main__":
    main()
