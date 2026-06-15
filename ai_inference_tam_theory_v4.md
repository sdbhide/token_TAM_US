# Structural Framework for AI Inference Token TAM Modeling — Consolidated Theory (v4)

**Status:** Single source of theoretical truth for the modeling project. Supersedes v3.
**Scope:** US market, USD, monthly expenditure, equilibrium (stable token prices).
**Implementation:** All closed forms in this document are implemented in `tam_core.py` and validated against numerical integration / brute-force simulation in `test_tam_core.py` (17/17 checks passing as of this version).

**Changelog from v3:**

| # | Change | Section |
|---|--------|---------|
| 1 | Corrected the consumer-anchored critical sigma (v3 had ½·ln(·) with no root; correct form is √(⅙·ln(·))) | §7.3 |
| 2 | Flagged and resolved the implicit-θ (transcendental) problem in σ_crit; numeric root-find is the production path | §7.4 |
| 3 | Added intra-segment correlation ρ between ln K and ln I (model stays lognormal) | §3.1 |
| 4 | Added hard caps via truncated-lognormal moments (closed form) | §3.2 |
| 5 | Made the count distribution N explicit (Binomial / Beta-Binomial), supplying Var(N) and μ₃(N) | §4 |
| 6 | Added cross-segment covariance hook (common macro factor) | §6.1 |
| 7 | Stated the hidden assumptions of the skewness approximation (deterministic-N terms dropped) | §7.1 |
| 8 | Stated regime-validity conditions and corrected the regime-selection inequality | §7.2–7.3 |
| 9 | Units convention pinned (monthly USD) | §2.1 |
| 10 | Repositioned §7 as a CI-validity diagnostic, not load-bearing estimation machinery | §7.5 |

---

## 1. The Bifurcated Macro Architecture

The total market size is the additive sum of two compound stochastic processes — Consumers ($i$) and Enterprises ($e$):

$$\text{TAM} = \text{TAM}_i + \text{TAM}_e = \sum_{j=1}^{N_i} X_{i,j} + \sum_{k=1}^{N_e} X_{e,k}$$

Where:

* $N_i$, $N_e$ are random variables: the adopting population counts (US individuals; US enterprise accounts).
* $X_{i,j}$ is the monthly token expenditure of individual consumer $j$.
* $X_{e,k}$ is the monthly token expenditure of enterprise account $k$.

**Assumptions (stated explicitly, relaxed where noted):**

* **A1 — Within-segment i.i.d.:** spends $X_{s,1}, X_{s,2}, \dots$ are i.i.d. within segment $s \in \{i, e\}$.
* **A2 — Count–spend independence:** $N_s \perp \{X_{s,j}\}$. *Caveat: selection effects (early adopters skew high-income) violate this at partial penetration; at full-saturation equilibrium it approximately holds. See §8.4.*
* **A3 — Cross-segment independence:** $\text{TAM}_i \perp \text{TAM}_e$ in the baseline; relaxed via a covariance term in §6.1.
* **A4 — Stable prices:** TAM in USD = price × token volume with price fixed; demand-side price endogeneity is out of scope for v1 (§8.5).

---

## 2. Segment-Specific Dual-Lognormal Infrastructure

To untangle underlying wealth from technology consumption propensity, unit expenditures are modeled as products:

$$X_i = K_i \cdot I_i \qquad X_e = K_e \cdot I_e$$

Where:

* $I_i \sim \text{Lognormal}(\mu_{I,i}, \sigma_{I,i}^2)$ — US personal income distribution.
* $I_e \sim \text{Lognormal}(\mu_{I,e}, \sigma_{I,e}^2)$ — corporate IT asset / firm-scale distribution.
* $K_i \sim \text{Lognormal}(\mu_{K,i}, \sigma_{K,i}^2)$ — consumer token-penetration rate (share of income spent on tokens).
* $K_e \sim \text{Lognormal}(\mu_{K,e}, \sigma_{K,e}^2)$ — enterprise token-penetration rate.

### 2.1 Units convention (NEW — pinned)

$X$ is **monthly USD spend per unit**. Income distributions are typically published **annually**, so either:

* define $K$ as *monthly spend as a share of annual income* (fold the $1/12$ into $\mu_K$: subtract $\ln 12$), or
* convert $I$ to monthly before fitting ($\mu_I \to \mu_I - \ln 12$).

Pick one and document it in the calibration file. A silent factor-of-12 error in $\mu_K$ shifts $E[\text{TAM}]$ by 12×.

### 2.2 Soft caps (segment-specific medians)

The bifurcation allows independent calibration. E.g., consumer median penetration $e^{\mu_{K,i}} = 0.005$ versus enterprise $e^{\mu_{K,e}} = 0.02$, reflecting autonomous agent fleets running recursive loops at corporate scale. Soft caps shift the **median** but leave the support unbounded — for genuinely bounded spend use the hard caps of §3.2.

---

## 3. Unit-Level Distribution of X

### 3.1 With intra-segment correlation (NEW)

v3 assumed $K \perp I$, implying spend share is unrelated to income/size. Empirically the share likely **declines** with consumer income (Engel-curve behavior, $\rho_i < 0$) and **rises** with firm size (agent fleets, $\rho_e > 0$). Let $(\ln I, \ln K)$ be bivariate normal with correlation $\rho$. Then $X = K \cdot I$ is **still lognormal**:

$$X_s \sim \text{Lognormal}(\mu_{X,s},\, \sigma_{X,s}^2), \qquad
\mu_{X,s} = \mu_{I,s} + \mu_{K,s}, \qquad
\sigma_{X,s}^2 = \sigma_{I,s}^2 + \sigma_{K,s}^2 + 2\rho_s\,\sigma_{I,s}\,\sigma_{K,s}$$

Setting $\rho_s = 0$ recovers the v3 formulation exactly. One extra parameter per segment, zero loss of tractability. Note $\rho < 0$ *shrinks* $\sigma_X^2$ — correlation structure directly moves the tail mass that drives §5–§7.

**Moments of the (untruncated) lognormal** $X \sim \text{Lognormal}(\mu, \sigma^2)$:

$$E[X] = e^{\mu + \sigma^2/2}$$
$$\text{Var}(X) = \left(e^{\sigma^2} - 1\right)e^{2\mu + \sigma^2}$$
$$\mu_3(X) = \left(e^{\sigma^2} + 2\right)\sqrt{e^{\sigma^2} - 1}\;\bigl(\text{Var}(X)\bigr)^{3/2}$$

(The third central moment is needed for §7; v3 only implicitly used its heavy-tail limit.)

### 3.2 Hard caps via truncation (NEW)

A firm cannot spend more than its IT budget; a consumer cannot spend more than disposable income. Truncate $X$ at cap $c$: work with the conditional distribution $X \mid X \le c$. For a lognormal, raw moments remain closed-form via the normal CDF $\Phi$:

$$E[X^k \mid X \le c] = e^{k\mu + k^2\sigma^2/2}\;\frac{\Phi(d_k)}{\Phi(d_0)}, \qquad
d_k = \frac{\ln c - \mu - k\sigma^2}{\sigma}$$

Central moments follow from raw moments ($\text{Var} = m_2 - m_1^2$; $\mu_3 = m_3 - 3m_1 m_2 + 2m_1^3$). Truncation tames the unphysical tail mass that otherwise inflates $\text{Var}(\text{TAM}_e)$ and $\mu_3(\text{TAM}_e)$ — in smoke tests, an enterprise cap of \$5M/mo pulled aggregate skewness from heavy to near-Gaussian. **The cap is therefore a first-order modeling choice, not a refinement.**

### 3.3 Tail-shape caveat (flagged for Phase 5)

US firm-size distributions are closer to Zipf/Pareto in the upper tail than lognormal, and API spend is empirically concentrated in a handful of hyperscale customers. A lognormal $I_e$ may understate $\text{Var}(\text{TAM}_e)$ even before capping. Phase 5 will compare a lognormal-body/Pareto-tail splice; note a pure Pareto tail with shape $\alpha \le 3$ has **no finite third moment**, which would invalidate §7 entirely — itself a diagnostic.

---

## 4. The Count Distribution N (NEW — made explicit)

v3 treated $E[N]$, $\text{Var}(N)$ as free inputs. We now derive them. Let $\text{Pop}_s$ be the eligible population (US adults; US firms) and $p_s$ the equilibrium adoption probability.

**Known adoption rate:** $N_s \sim \text{Binomial}(\text{Pop}_s, p_s)$:

$$E[N_s] = \text{Pop}_s\,p_s, \qquad \text{Var}(N_s) = \text{Pop}_s\,p_s(1-p_s), \qquad \mu_3(N_s) = \text{Pop}_s\,p_s(1-p_s)(1-2p_s)$$

**Uncertain adoption rate:** $p_s \sim \text{Beta}(\alpha_s, \beta_s)$ gives a Beta-Binomial $N_s$, with over-dispersion:

$$E[N_s] = \text{Pop}_s\,\bar p_s, \qquad
\text{Var}(N_s) = \text{Pop}_s\,\bar p_s(1-\bar p_s)\left[1 + \frac{\text{Pop}_s - 1}{\alpha_s + \beta_s + 1}\right], \quad \bar p_s = \tfrac{\alpha_s}{\alpha_s+\beta_s}$$

Because $\text{Pop}_s$ is huge, the Beta-Binomial variance is dominated by the parameter-uncertainty term and scales like $\text{Pop}_s^2 \text{Var}(p_s)$ — i.e., **adoption-rate uncertainty, not sampling noise, drives $\text{Var}(N)$** at US scale. This feeds §5–§7 directly and is the main reason the deterministic-N approximation in §7.1 must be checked rather than assumed.

The implementation accepts arbitrary $(E[N], \text{Var}(N), \mu_3(N))$, so any count model that supplies its first three central moments plugs in unchanged.

---

## 5. Compound-Sum Moments (Wald + Third Cumulant)

For each segment $s$, with $X$ i.i.d. and independent of $N$ (A1, A2):

$$E[\text{TAM}_s] = E[N_s]\,E[X_s]$$

$$\text{Var}(\text{TAM}_s) = E[N_s]\,\text{Var}(X_s) + (E[X_s])^2\,\text{Var}(N_s)$$

$$\mu_3(\text{TAM}_s) = E[N_s]\,\mu_3(X_s) + 3\,\text{Var}(N_s)\,E[X_s]\,\text{Var}(X_s) + \mu_3(N_s)\,(E[X_s])^3$$

The third line is the exact third-cumulant expansion of a compound sum (cumulants of $S$ obtained from $K_S(t) = K_N(\ln M_X(t))$); v3 stated it correctly. When §3.2 caps are active, the truncated moments of $X$ are substituted throughout — the compound formulas are unchanged.

---

## 6. Whole-Market Aggregates

### 6.1 Expectation and variance (with covariance hook — NEW)

$$E[\text{TAM}] = E[\text{TAM}_i] + E[\text{TAM}_e]$$

$$\text{Var}(\text{TAM}) = \text{Var}(\text{TAM}_i) + \text{Var}(\text{TAM}_e) + 2\,\text{Cov}(\text{TAM}_i, \text{TAM}_e)$$

The baseline sets $\text{Cov} = 0$ (A3). In reality both segments respond to common shocks — model-capability jumps, price moves, macro conditions — so independence **systematically understates total variance**. Phase 5 introduces a latent macro factor $Z$ scaling both segments, which supplies the covariance term; the formula above already accommodates it.

### 6.2 The structural scale factor θ

$$\theta = \frac{E[X_e]}{E[X_i]} = e^{(\mu_{X,e} - \mu_{X,i}) + \frac{\sigma_{X,e}^2 - \sigma_{X,i}^2}{2}} \quad (\text{uncapped case})$$

In the real economy $\theta \gg 1$. **Two definitions, used for different purposes:**

* **Mean-ratio θ** (above): the economically meaningful scale gap. Carries $\sigma_{X,e}^2$ dependence. When caps are active, compute it from truncated means.
* **Median-ratio θ_med** $= e^{\mu_{X,e} - \mu_{X,i}}$: free of $\sigma_{X,e}$, used *only* to keep the closed-form $\sigma_\text{crit}$ in §7.3 from becoming circular.

---

## 7. Gaussian Convergence Diagnostics

As $E[N_i], E[N_e] \to \infty$ the aggregate converges to a normal distribution (random-index CLT / Anscombe's theorem; requires $N/E[N] \to 1$ in probability). Convergence speed is governed by skewness:

$$\gamma_{\text{TAM}} = \frac{\mu_3(\text{TAM}_i) + \mu_3(\text{TAM}_e)}{\bigl(\text{Var}(\text{TAM}_i) + \text{Var}(\text{TAM}_e)\bigr)^{3/2}} \le \epsilon$$

This **exact** expression (via §5) is what the implementation computes. The approximations below exist only to expose structure.

### 7.1 The heavy-tail approximation and its hidden assumptions (CLARIFIED)

The v3 approximation

$$\gamma_{\text{TAM}} \approx \frac{E[N_e]\,e^{3\sigma_{X,e}^2}\,\theta^3}{\left(E[N_i](e^{\sigma_{X,i}^2}-1) + E[N_e]\,e^{\sigma_{X,e}^2}\,\theta^2\right)^{3/2}}$$

requires, explicitly:

1. **Heavy enterprise tail:** $e^{\sigma_{X,e}^2} \gg 1$, so $\mu_3(X_e) \approx e^{3\sigma_{X,e}^2}(E[X_e])^3$ and $e^{\sigma^2}-1 \approx e^{\sigma^2}$.
2. **Near-deterministic counts:** the $\text{Var}(N)$ and $\mu_3(N)$ terms of §5 are dropped. Per §4, Beta-Binomial adoption uncertainty can make these terms **dominant** — in that regime the approximation, and everything in §7.2–7.3 built on it, is invalid. Check before use.
3. **Enterprise skew dominance:** $\mu_3(\text{TAM}_i)$ negligible relative to $\mu_3(\text{TAM}_e)$.
4. **No caps:** truncation (§3.2) kills the $e^{3\sigma^2}$ asymptotics; with caps, only the exact expression applies.

### 7.2 Regime A — Enterprise-dominated variance

**Condition (corrected):** $E[N_e]\,e^{\sigma_{X,e}^2}\,\theta^2 \gg E[N_i](e^{\sigma_{X,i}^2}-1)$.
*(v3 stated the condition as $E[N_e]\theta^2 \gg E[N_i]e^{\sigma_{X,i}^2}$, dropping the $e^{\sigma_{X,e}^2}$ factor — which matters precisely in the high-concentration regime where this analysis is interesting.)*

Then $\gamma \approx e^{\frac{3}{2}\sigma_{X,e}^2}/\sqrt{E[N_e]}$ and

$$\boxed{\;\sigma_{X,e(\text{crit})} = \sqrt{\tfrac{1}{3}\ln\!\left(\epsilon^2\,E[N_e]\right)}\;}$$

valid only when $\epsilon^2 E[N_e] > 1$. (v3's first case — verified correct.)

### 7.3 Regime B — Consumer-anchored variance (CORRECTED)

**Condition:** $E[N_i](e^{\sigma_{X,i}^2}-1) \gg E[N_e]\,e^{\sigma_{X,e}^2}\,\theta^2$.

From $\dfrac{E[N_e]\,e^{3\sigma_{X,e}^2}\,\theta^3}{\left(E[N_i](e^{\sigma_{X,i}^2}-1)\right)^{3/2}} \le \epsilon$, taking logs and solving for $\sigma_{X,e}$:

$$\boxed{\;\sigma_{X,e(\text{crit})} = \sqrt{\frac{1}{6}\ln\!\left(\frac{\epsilon^2\,(E[N_i])^3\,(e^{\sigma_{X,i}^2}-1)^3}{(E[N_e])^2\,\theta_{\text{med}}^6}\right)}\;}$$

**v3 error:** the previous version had $\frac{1}{2}\ln(\cdot)$ with no outer square root — wrong coefficient ($\tfrac12$ vs $\tfrac16$) and wrong dimension (returns a $\sigma^2$-like quantity labeled $\sigma$). In a test regime where both forms produce a number, the v3 formula overstates $\sigma_\text{crit}$ by ~7×.

**Domain note:** the log-argument must exceed 1 for a real root. With realistic $\theta \sim 10^2$–$10^3$, the $\theta^6$ denominator usually pushes the argument far below 1 — i.e., **realistic parameterizations are rarely in Regime B**. A failed domain check is the formula telling you to use Regime A or the numeric solver.

### 7.4 The implicit-θ problem and the production method (NEW)

Mean-ratio $\theta$ contains $\sigma_{X,e}^2$, so "solving for $\sigma_{X,e(\text{crit})}$" with $\theta$ on the right-hand side is a **transcendental equation**, not a closed form. Resolutions:

1. Use $\theta_\text{med}$ in the closed forms (as written in §7.3) — removes circularity at the cost of a reinterpretation, and only inside the approximation's validity envelope (§7.1).
2. **Production path:** solve $\gamma_{\text{TAM}}(\sigma_{X,e}) = \epsilon$ by numerical root-finding on the *exact* skewness of §7 (Brent's method). This is regime-free, honors caps, correlation, and count moments, and is implemented as `sigma_crit_numeric`. Validated: hits $\gamma = \epsilon$ to ~$10^{-9}$ relative error. Feasibility note: with $\sigma_K \ge 0$, the smallest achievable $\sigma_X$ is $\sigma_I\sqrt{1-\rho^2}$ (for $\rho<0$; else $\sigma_I$), which bounds the root-search bracket below.

### 7.5 Role of this section (REPOSITIONED)

The convergence analysis is **not** load-bearing for TAM estimation — the Monte Carlo engine (Phase 2) estimates the full distribution directly. Its purpose is diagnostic: it tells you **when analytical (normal-approximation) confidence intervals are trustworthy** and when only simulated quantiles should be reported. Even when $\gamma \le \epsilon$ passes, the normal approximation can fail in the far tails with heavy-tailed $X_e$ and modest $E[N_e]$; Phase 4's QQ-plot panel validates this empirically. The headline insight survives intact: *the speed of convergence to Gaussian predictability is tethered to the scale gap θ between retail and commercial tiers* — and, post-correction, to the enterprise concentration $\sigma_{X,e}$ and the hard cap, which the smoke tests show are the dominant levers.

---

## 8. Known Limitations and Phase-5 Extensions

**8.1 Cross-segment correlation.** Common macro shocks ⇒ $\text{Cov}(\text{TAM}_i, \text{TAM}_e) > 0$; baseline understates variance. Hook exists in §6.1; latent-factor model in Phase 5.

**8.2 Enterprise tail shape.** Lognormal vs Pareto-tail comparison deferred to Phase 5 (§3.3). Affects tail quantiles far more than the mean.

**8.3 Count model realism.** §4's Binomial/Beta-Binomial is a static equilibrium reduction; a Bass-diffusion layer would give path-to-equilibrium dynamics. Out of scope for the equilibrium model.

**8.4 Selection effects.** Adopters are not income-random (violates A2 at partial penetration); couples $N$ and $X$. Sensitivity check planned, not modeled in v1.

**8.5 Price endogeneity.** Falling token prices + demand elasticity (Jevons effects) break A4. Explicitly out of scope for v1.

---

## 9. Implementation Map

| Theory | Code (`tam_core.py`) | Validation (`test_tam_core.py`) |
|---|---|---|
| §3.1 lognormal + ρ | `SegmentParams`, `lognormal_moments` | quadrature (rel. err ~1e-11); brute MC for ρ ≠ 0 |
| §3.2 truncation | `truncated_lognormal_moments`, `cap` param | quadrature (rel. err ~1e-11) |
| §4 counts | `CountParams` (moments-in interface) | Poisson case via brute MC |
| §5 compound moments | `compound_moments` | brute MC, 200k reps |
| §6 aggregates | `tam_expected`, `tam_variance`, `tam_skewness`, `theta` | two-segment brute MC, 300k reps |
| §7.2 Regime A | `sigma_crit_enterprise_dominated` | self-consistency to machine precision |
| §7.3 Regime B (corrected) | `sigma_crit_consumer_anchored` | domain checks + positivity |
| §7.4 production solver | `sigma_crit_numeric` | hits γ = ε to ~1e-9 |

**Contract for Phase 2 (Monte Carlo):** simulated $E[\text{TAM}]$, $\text{Var}(\text{TAM})$, and $\gamma_{\text{TAM}}$ must match this document's closed forms within Monte Carlo standard error before any downstream result (quantiles, sensitivity, visualization) is trusted.
