"""
precompute_dashboard.py
=======================
Precompute every scenario the dashboard can display, and dump to a single JSON
the static React app loads. No backend needed — Netlify-friendly.

Scenarios vary the three enterprise levers Phase 3 identified as dominant:
  * enterprise AI propensity  (mu_K_e)   -> "Enterprise AI adoption"  (Low/Base/High)
  * enterprise concentration  (sigma_X_e via sigma_K_e) -> "Spend concentration"
  * enterprise IT-budget scale (mu_I_e)  -> "Enterprise scale"
Consumer params stay at Phase 0 central (Phase 3 showed they barely move TAM).

For each of the 27 combinations we store:
  - TAM histogram (log-spaced bins) + mean/median/P5/P25/P75/P95/skew
  - segment means and variance shares (with Beta-Binomial count uncertainty)
  - Sobol total-order indices for E[TAM] and P(TAM > $100B)
  - the (theta, sigma_X_e) coordinate for the convergence map
"""

import json
import numpy as np

import tam_core as tc
import tam_mc as mc
import tam_sensitivity as ts
from tam_mc import PHASE0_PRIORS, Prior

B = 1e9
RNG_SEED = 20260612

# ---- lever definitions (three levels each) -------------------------------
LEVELS = {
    "adoption": {   # enterprise AI propensity, mu_K_e
        "label": "Enterprise AI adoption",
        "param": "mu_K_e",
        "options": {"Low": -7.4, "Base": -6.74, "High": -5.9},
    },
    "concentration": {   # enterprise concentration, sigma_K_e
        "label": "Spend concentration",
        "param": "sigma_K_e",
        "options": {"Low": 0.85, "Base": 1.2, "High": 1.5},
    },
    "scale": {   # enterprise IT-budget scale, mu_I_e
        "label": "Enterprise scale",
        "param": "mu_I_e",
        "options": {"Small": 11.6, "Base": 12.0, "Large": 12.9},
    },
}


def make_priors(mu_K_e, sigma_K_e, mu_I_e):
    """Phase 0 priors with the three enterprise levers pinned to fixed values
    (so the scenario is deterministic in those, but still carries uncertainty
    in the remaining params for the distribution)."""
    p = dict(PHASE0_PRIORS)
    p["mu_K_e"] = Prior(mu_K_e, mu_K_e, mode=mu_K_e, kind="fixed")
    p["sigma_K_e"] = Prior(sigma_K_e, sigma_K_e, mode=sigma_K_e, kind="fixed")
    p["mu_I_e"] = Prior(mu_I_e, mu_I_e, mode=mu_I_e, kind="fixed")
    return p


def betabinom_counts(pop, p_mean, conc):
    a, b = p_mean * conc, (1 - p_mean) * conc
    var = pop * p_mean * (1 - p_mean) * (1 + (pop - 1) / (a + b + 1))
    return tc.CountParams(mean=pop * p_mean, var=var, mu3=0.0)


def scenario(mu_K_e, sigma_K_e, mu_I_e):
    priors = make_priors(mu_K_e, sigma_K_e, mu_I_e)

    # ---- TAM distribution via the validated MC engine
    cfg = mc.MCConfig(outer_draws=8_000, inner_reps=1, seed=RNG_SEED,
                      priors=priors)
    res = mc.run_mc(cfg)
    tam = res.tam / B

    qs = {q: float(np.quantile(tam, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    mean = float(tam.mean())
    skew = float(np.mean((tam - tam.mean()) ** 3) / tam.var() ** 1.5)

    # log-spaced histogram for the chart
    lo, hi = np.quantile(tam, [0.002, 0.998])
    edges = np.geomspace(max(lo, 1.0), hi, 41)
    counts, _ = np.histogram(tam, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    hist = [{"x": float(c), "y": int(n)} for c, n in zip(centers, counts)]

    # ---- segment decomposition (means + variance shares with count noise)
    draw = mc.central_model_draw(priors)
    cons, ent = draw.cons, draw.ent
    cons_n = betabinom_counts(260e6, draw.n_i / 260e6, 400.0)
    ent_n = betabinom_counts(5.9e6, draw.n_e / 5.9e6, 50.0)
    e_i, v_i, _ = tc.compound_moments(cons, cons_n)
    e_e, v_e, _ = tc.compound_moments(ent, ent_n)

    # ---- Sobol total-order indices (closed-form evaluators, fast)
    def sobol_for(qoi, n_base=2**12):
        # n_boot=1 (not 0) avoids an empty-array quantile; we ignore CIs here
        r = ts.sobol_indices(qoi, n_base=n_base, priors=PHASE0_PRIORS,
                             seed=13, n_boot=1)
        # report only the 4 enterprise drivers + a lumped "other"
        idx = {n: i for i, n in enumerate(ts.PARAM_NAMES)}
        ent_keys = ["mu_I_e", "sigma_I_e", "mu_K_e", "sigma_K_e"]
        out = [{"name": k, "ST": max(float(r.ST[idx[k]]), 0.0)} for k in ent_keys]
        other = max(float(sum(r.ST)) - sum(o["ST"] for o in out), 0.0)
        out.append({"name": "all others", "ST": other})
        s = sum(o["ST"] for o in out) or 1.0
        for o in out:
            o["ST"] = o["ST"] / s     # normalize to shares
        return out

    sobol_mean = sobol_for(ts.e_tam_vectorized)
    sobol_tail = sobol_for(lambda P: (ts.e_tam_vectorized(P) > 100e9).astype(float))

    theta = tc.theta(cons, ent)
    return {
        "stats": {"mean": mean, "median": qs[0.5], "p5": qs[0.05],
                  "p25": qs[0.25], "p75": qs[0.75], "p95": qs[0.95],
                  "skew": skew},
        "hist": hist,
        "segments": {
            "consumer": {"mean": e_i / B, "varShare": float(v_i / (v_i + v_e))},
            "enterprise": {"mean": e_e / B, "varShare": float(v_e / (v_i + v_e))},
        },
        "sobolMean": sobol_mean,
        "sobolTail": sobol_tail,
        "coords": {"theta": float(theta), "sigmaXe": float(ent.sigma_X)},
    }


def main():
    data = {"levels": LEVELS, "scenarios": {}}
    keys_a = list(LEVELS["adoption"]["options"])
    keys_c = list(LEVELS["concentration"]["options"])
    keys_s = list(LEVELS["scale"]["options"])
    total = len(keys_a) * len(keys_c) * len(keys_s)
    i = 0
    for a in keys_a:
        for c in keys_c:
            for s in keys_s:
                i += 1
                key = f"{a}|{c}|{s}"
                mu_K_e = LEVELS["adoption"]["options"][a]
                sigma_K_e = LEVELS["concentration"]["options"][c]
                mu_I_e = LEVELS["scale"]["options"][s]
                print(f"[{i}/{total}] {key} ...", flush=True)
                data["scenarios"][key] = scenario(mu_K_e, sigma_K_e, mu_I_e)

    # convergence map: a coarse theoretical sigma_crit frontier for context
    cons = mc.central_model_draw().cons
    cons_n = tc.CountParams(mean=1e8, var=0.0, mu3=0.0)
    ent_t = mc.central_model_draw().ent
    n_es = np.geomspace(1e4, 6e6, 30)
    frontier = []
    for ne in n_es:
        try:
            sc = tc.sigma_crit_numeric(
                cons, cons_n, ent_t,
                tc.CountParams(mean=float(ne), var=0.0, mu3=0.0),
                epsilon=0.5, bracket=(ent_t.sigma_I + 0.02, 3.0))
        except ValueError:
            sc = None
        frontier.append({"nE": float(ne / 1e6), "sigmaCrit": sc})
    data["convergenceFrontier"] = frontier

    with open("dashboard_data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    import os
    sz = os.path.getsize("dashboard_data.json") / 1024
    print(f"\nWrote dashboard_data.json ({sz:.0f} KB, {total} scenarios)")


if __name__ == "__main__":
    main()
