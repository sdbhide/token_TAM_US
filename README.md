# AI Inference Token TAM: Project Documentation

A complete map of the modeling project: every module, test suite, and figure,
with what it is, why it exists, and what it means for the overall estimate.
Read this as the index; the theory lives in `ai_inference_tam_theory_v4.md` and
the derivations/validations live in the code. Check the [dashboard](https://aitokentam.netlify.app/) here. 

**Goal.** Estimate the US Total Addressable Market for LLM inference tokens, in
USD at stable-price equilibrium, as a *distribution* (not a point), split into
consumer and enterprise segments.

**Headline result (under Phase 0 priors).** E[TAM] ≈ **$50B/yr**, median ≈
**$41B**, 90% interval ≈ **$20B–$112B**. Conditioned on the current-revenue
window ($14–22B), the present-market estimate tightens to ≈ **$20B** (mean),
$15–25B (90%). The single most important number for *reducing* uncertainty:
~95% of TAM variance traces to four enterprise spend parameters, so calibration
effort belongs there.

---

## 1. How the pieces fit together

```
Phase 0  calibration (yours)          phase_0.py
   │      priors + macro constraint
   ▼
Phase 1  analytical core              tam_core.py        ← ground truth
   │      closed-form moments         test_tam_core.py     (17/17)
   ▼
Phase 2  Monte Carlo engine           tam_mc.py          ← estimates the dist.
   │      two-level + hybrid tail     test_tam_mc.py       (20/20)
   │                                  run_phase2.py
   ▼
Phase 3  sensitivity / attribution    tam_sensitivity.py ← where to calibrate
   │      tornado + Sobol + grid      test_tam_sensitivity.py (18/18)
   │                                  run_phase3.py
   ▼
Phase 4  visualization                tam_viz.py         ← communicates results
          five figures                test_tam_viz.py      (11/11)
                                       run_phase4.py
```

Each phase is validated against the one above it: Phase 2's acceptance test is
that its simulated moments match Phase 1's closed forms; Phase 4(c) re-validates
Phase 1's convergence theory against Phase 2's simulator. Nothing downstream is
trusted until the layer it rests on passes its tests.

---

## 2. Modules (what each file is and why it exists)

### `tam_core.py` — analytical ground truth
Closed-form moments for the dual-segment compound-lognormal model: per-unit
spend (with optional income–penetration correlation ρ and hard spend caps),
compound-sum moments via Wald + third-cumulant expansion, whole-market
aggregates, and three ways to compute the critical enterprise concentration
σ_crit (including the corrected consumer-anchored form and a numeric solver that
sidesteps the transcendental-θ problem). *Why:* a fast, exact reference so the
simulator can be checked and so sensitivity analysis is essentially free.

### `tam_mc.py` — Monte Carlo engine
Two-level simulation: an outer loop over parameter uncertainty (Phase 0 priors
as triangular distributions) and an inner loop over population realizations. The
consumer total is drawn from a three-moment-matched shifted gamma (never 10^8
literal draws); the enterprise total uses a **hybrid stratified** scheme that
simulates the heavy top ~0.1% of accounts exactly and moment-matches the body.
The tail split is adaptive so cost stays bounded at any scale. *Why:* the
distribution has no closed form; this estimates it accurately and ~150× faster
than brute force.

### `tam_sensitivity.py` — attribution
Vectorized closed-form evaluators (justified because Phase 2 showed population
noise is ~0.04% of variance, so params→E[TAM] *is* the model), a one-at-a-time
tornado, a from-scratch Saltelli/Jansen **Sobol** implementation (validated on
the Ishigami benchmark), the scenario-grid generator, and macro-constraint
importance weighting. *Why:* tells us which parameters actually move the answer.

### `tam_viz.py` — figures
Five figure builders plus their numeric kernels (Gini, Lorenz, empirical-skew
grid). Pure functions returning matplotlib figures. *Why:* turn the results into
the five views that carry the project's findings.

### Test suites (`test_*.py`)
66 checks total. The important ones aren't "does it run" but "does it match an
independent ground truth": closed forms vs numerical integration; simulator vs
closed forms; Sobol vs Ishigami's analytical indices; Gini vs the analytical
lognormal value erf(σ/2). *Why:* every quantitative claim is anchored to
something that can't be fudged.

---

## 3. The visualizations

### Phase 2 — `tam_phase2_diagnostics.png` (2×2 diagnostics)
**What:** four panels — running-mean convergence with ±2 SE, the TAM
distribution on a log axis, the standard error tracking against the n^(−1/2)
reference, and consumer-vs-enterprise segment totals.
**Why:** confirm the Monte Carlo has actually converged before any number is
quoted.
**Means for the project:** the engine is trustworthy — the running mean settles
inside its error band and SE falls at the theoretical rate — so the headline
distribution can be read off with confidence.

### Phase 3 — `tam_phase3_tornado.png` (tornado)
**What:** each parameter swung to its prior low/high with others held central;
bars sorted by the resulting swing in E[TAM].
**Why:** a fast first look at which inputs move the headline number.
**Means for the project:** the four enterprise spend parameters (`mu_I_e`,
`sigma_I_e`, `mu_K_e`, `sigma_K_e`) produce the largest swings — tens of
billions each — while consumer income params and both population counts barely
register. First evidence that enterprise calibration dominates.

### Phase 3 — `tam_phase3_sobol.png` (Sobol indices, 3 panels)
**What:** first-order (S1) and total-order (ST) variance shares for three
quantities of interest: E[TAM], P(TAM>$50B), P(TAM>$100B), with bootstrap CIs.
**Why:** the tornado can't see interactions or tail-specific drivers; global
sensitivity can.
**Means for the project:** ~95% of variance in the *level* comes from the four
enterprise spend parameters. Moving to the *tail* (P(TAM>$100B)), the S1↔ST gap
widens sharply — tail risk is driven by parameter *interactions*, and
`sigma_K_e`'s importance rises, confirming dispersion matters more the deeper
into the tail you look. Actionable: pin the enterprise IT-budget distribution
(its median *and* spread) with one cut of firm-size microdata.

### Phase 3 — `tam_phase3_scenario_grid.png` (γ over θ × σ_X,e × E[N_e])
**What:** heatmap of TAM skewness across the scale-gap θ and enterprise
concentration σ_X,e, at three enterprise-population levels, with γ=0.1 and
γ=0.5 contours.
**Why:** visualize where the market is predictably Gaussian vs heavy-tailed,
in the model's own structural coordinates.
**Means for the project:** the convergence frontier is **flat in θ** — exactly
the enterprise-dominated regime the theory predicts, where σ_crit doesn't depend
on the scale gap. Skewness is governed by σ_X,e and E[N_e], not by how much
bigger enterprise accounts are than consumer ones.

### Phase 4 — `tam_phase4_a_distribution.png` (the distribution)
**What:** posterior-predictive TAM on a log axis with mean, median, and P5–P95
marked.
**Why:** the single headline figure; the mean–median gap is the lognormal-skew
story.
**Means for the project:** mean ($50B) sits 23% above median ($41B). Reporting a
single mean would overstate the typical outcome — the distribution, and its
interval, is the deliverable.

### Phase 4 — `tam_phase4_b_lorenz.png` (Lorenz / Gini)
**What:** Lorenz curve of per-account enterprise spend; Gini 0.87; top-1% and
top-0.1% TAM shares annotated.
**Why:** quantify how concentrated enterprise spend is.
**Means for the project:** the top 1% of enterprise accounts carry ~44% of
enterprise TAM (top 0.1% ≈ 18%). This is *why* the simulator needs exact-tail
treatment and why enterprise dispersion dominates sensitivity — a handful of
hyperscale accounts set the number.

### Phase 4 — `tam_phase4_c_skew_heatmap.png` (theory validation) ★
**What:** empirical skewness from independent simulation over (E[N_e], σ_X,e),
with the theoretical σ_crit frontier (cyan) and the empirical γ=0.5 contour
(white) overlaid.
**Why:** directly test whether the Phase 1 convergence derivation matches what
the Phase 2 simulator actually produces.
**Means for the project:** the two frontiers coincide across four orders of
magnitude in E[N_e] — the standout validation that theory and simulation agree.
When the analytical normal-approximation CIs are trustworthy is now known, not
assumed.

### Phase 4 — `tam_phase4_d_qq_convergence.png` (Anscombe convergence)
**What:** normal QQ plots of simulated TAM at four adoption scales (×0.001 to
×20).
**Why:** show *how* and *where* the market becomes Gaussian as adoption grows.
**Means for the project:** at small scale the upper tail flies off the normal
line (skew ~2.9); by present scale the body is straight with only the extreme
tail deviating. The upper tail complies last — so even when overall skewness is
small, tail quantiles should come from simulation, not a normal approximation.

### Phase 4 — `tam_phase4_e_waterfall.png` (segment decomposition)
**What:** additive split of E[TAM] and Var(TAM) into consumer vs enterprise.
**Why:** show which segment carries the level and which carries the risk.
**Means for the project:** the mean splits ~53/47 consumer/enterprise. Variance
is the surprise: with realistic adoption-count uncertainty the *consumer* layer
slightly leads (~58%), because 100M consumers amplify even a small
adoption-rate uncertainty more than 5M firms do. Flags a Phase 0 follow-up —
pin the consumer adoption-rate prior. *(Concentration params here are
illustrative, not yet Phase 0–calibrated.)*

---

## 4. What the project establishes

1. **A validated, layered model.** Closed forms (exact) → simulator (matches
   them) → sensitivity (free, because population noise is negligible) →
   figures. 66 independent checks.
2. **The number.** ~$50B/yr expected equilibrium TAM, heavily right-skewed,
   90% interval $20–112B; ~$20B as a present-market estimate once conditioned
   on observed revenue.
3. **Where the uncertainty lives.** Almost entirely in enterprise spend
   parameters — and in the tail, in their interactions. This is the single
   most useful output: it says exactly where the next dollar of calibration
   effort should go.
4. **Honesty about tails.** Concentration is extreme (Gini 0.87; top 1% → 44%),
   so tail quantiles come from exact-tail simulation, and the convergence
   theory tells us precisely when the simpler normal approximation is safe.

## 5. Known limitations / open extensions (Phase 5)
- Cross-segment correlation from common macro shocks (variance currently
  understated by the independence assumption).
- Heavier-than-lognormal enterprise tail (Pareto splice) — would widen tail
  quantiles; a pure Pareto tail with shape ≤3 would have no finite skewness,
  itself diagnostic.
- Stable-price assumption: separating *current-market* from *equilibrium* TAM
  properly requires modeling price decline and demand elasticity.
- Consumer adoption-rate prior deserves tightening, per figure (e).

---

## 6. File index
| File | Role |
|------|------|
| `ai_inference_tam_theory.md` | Single source of theory (derivations, corrections) |
| `tam_core.py` / `test_tam_core.py` | Analytical core / 17 checks |
| `tam_mc.py` / `test_tam_mc.py` / `run_phase2.py` | MC engine / 20 checks / driver |
| `tam_sensitivity.py` / `test_tam_sensitivity.py` / `run_phase3.py` | Attribution / 18 checks / driver |
| `tam_viz.py` / `test_tam_viz.py` / `run_phase4.py` | Figures / 11 checks / driver |
| `tam_phase2_diagnostics.png` | MC convergence diagnostics |
| `tam_phase3_tornado.png` | OAT parameter swings |
| `tam_phase3_sobol.png` | Global sensitivity (level + tail) |
| `tam_phase3_scenario_grid.png` | Skewness over (θ, σ_X,e, E[N_e]) |
| `tam_phase4_a_distribution.png` | Headline TAM distribution |
| `tam_phase4_b_lorenz.png` | Enterprise spend concentration |
| `tam_phase4_c_skew_heatmap.png` | Theory-vs-simulation validation |
| `tam_phase4_d_qq_convergence.png` | Anscombe convergence across scale |
| `tam_phase4_e_waterfall.png` | Segment decomposition of mean & variance |
| `tam_phase2_samples.npz` | Saved MC draws (reused by figures) |

