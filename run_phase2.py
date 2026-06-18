"""
run_phase2.py
=============
Production Phase 2 run: full two-level Monte Carlo under the Phase 0 priors,
with convergence diagnostics, bootstrap quantile CIs, variance decomposition,
and saved outputs (samples .npz + diagnostics figure .png).

Run:  python3 run_phase2.py
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tam_mc as mc

OUT_NPZ = "tam_phase2_samples.npz"
OUT_PNG = "tam_phase2_diagnostics.png"


def main():
    # ----- main posterior-predictive run (parameter + population uncertainty)
    cfg = mc.MCConfig(outer_draws=20_000, inner_reps=1, seed=20260612)
    t0 = time.time()
    res = mc.run_mc(cfg)
    dt = time.time() - t0
    s = res.summary()

    # ----- decomposition run (needs inner_reps > 1)
    cfg_d = mc.MCConfig(outer_draws=1_500, inner_reps=8, seed=99)
    vd = mc.run_mc(cfg_d).variance_decomposition()

    # ----- bootstrap CIs on headline quantiles
    cis = mc.bootstrap_quantile_ci(res.tam, qs=(0.05, 0.25, 0.50, 0.75, 0.95),
                                   n_boot=2000)

    # ----- console report
    B = 1e9
    print(f"Two-level MC: {cfg.outer_draws} outer x {cfg.inner_reps} inner "
          f"in {dt:.1f}s ({res.tam.size/dt:,.0f} samples/s)\n")
    print(f"E[TAM]   = ${s['mean']/B:,.1f}B   (MC SE ${s['se_mean']/B:,.2f}B)")
    print(f"SD[TAM]  = ${s['sd']/B:,.1f}B    skew = {s['skew']:.2f}")
    print(f"  consumer mean  = ${s['mean_i']/B:,.1f}B")
    print(f"  enterprise mean= ${s['mean_e']/B:,.1f}B")
    print("\nQuantiles (bootstrap 95% CIs):")
    for q, c in cis.items():
        print(f"  P{int(q*100):02d} = ${c['point']/B:7.1f}B   "
              f"[{c['ci_low']/B:7.1f}B, {c['ci_high']/B:7.1f}B]")
    print(f"\nVariance decomposition (law of total variance):")
    print(f"  parameter uncertainty : {vd['share_parameter']:.2%}")
    print(f"  population noise      : {1-vd['share_parameter']:.2%}")

    # ----- save samples
    np.savez_compressed(
        OUT_NPZ, tam=res.tam, tam_i=res.tam_i, tam_e=res.tam_e,
        outer_index=res.outer_index)
    print(f"\nSamples saved -> {OUT_NPZ}")

    # ----- diagnostics figure
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("Phase 2 Monte Carlo - US AI Inference TAM (annual USD, "
                 "Phase 0 priors)", fontsize=13)

    # (a) running mean +/- 2SE
    ax = axes[0, 0]
    rmean, rse = mc.running_mean_se(res.tam)
    n_ax = np.arange(1, res.tam.size + 1)
    sl = slice(50, None)  # skip noisy head for display
    ax.plot(n_ax[sl], rmean[sl] / B, lw=1.2, color="tab:blue",
            label="running mean")
    ax.fill_between(n_ax[sl], (rmean[sl] - 2*rse[sl]) / B,
                    (rmean[sl] + 2*rse[sl]) / B, alpha=0.25,
                    color="tab:blue", label="±2 SE")
    ax.set_title("(a) Running mean convergence")
    ax.set_xlabel("samples"); ax.set_ylabel("E[TAM] ($B/yr)")
    ax.set_xscale("log"); ax.legend(); ax.grid(alpha=0.3)

    # (b) TAM distribution, log axis, with key quantiles
    ax = axes[0, 1]
    ax.hist(res.tam / B, bins=np.geomspace(res.tam.min()/B,
            res.tam.max()/B, 80), color="tab:green", alpha=0.7)
    for q, ls in [(0.05, ":"), (0.50, "-"), (0.95, ":")]:
        v = np.quantile(res.tam, q) / B
        ax.axvline(v, color="k", ls=ls, lw=1.2)
        ax.text(v, ax.get_ylim()[1]*0.92, f" P{int(q*100)}\n ${v:.0f}B",
                fontsize=8, va="top")
    ax.axvline(s["mean"]/B, color="tab:red", lw=1.5, label=f"mean ${s['mean']/B:.0f}B")
    ax.set_xscale("log")
    ax.set_title("(b) TAM distribution (mean–median gap = lognormal skew)")
    ax.set_xlabel("TAM ($B/yr, log scale)"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) running SE of the mean (variance-reduction tracking)
    ax = axes[1, 0]
    ax.loglog(n_ax[sl], rse[sl] / B, lw=1.2, color="tab:purple")
    ref = rse[100] / B * np.sqrt(100.0 / n_ax[sl])
    ax.loglog(n_ax[sl], ref, "k--", lw=0.8, label=r"$n^{-1/2}$ reference")
    ax.set_title("(c) Standard error of the mean")
    ax.set_xlabel("samples"); ax.set_ylabel("SE ($B)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # (d) segment decomposition
    ax = axes[1, 1]
    ax.hist(res.tam_i / B, bins=80, alpha=0.55, label="consumer", color="tab:blue")
    ax.hist(res.tam_e / B, bins=np.geomspace(max(res.tam_e.min()/B, 1e-2),
            res.tam_e.max()/B, 80), alpha=0.55, label="enterprise",
            color="tab:orange")
    ax.set_xscale("log")
    ax.set_title("(d) Segment totals (note enterprise dispersion)")
    ax.set_xlabel("segment TAM ($B/yr, log scale)"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Diagnostics figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
