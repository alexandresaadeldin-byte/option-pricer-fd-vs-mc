# Sub-project 1 — Black–Scholes core & European options

*Design approved on 2026-09-30.*

## 1. Context and goal

The overall project compares two pricing methods — **finite differences** (Crank–Nicolson) and **Monte Carlo** (with variance reduction) — across several products (European, American, barrier, digital) and models (Black–Scholes, then local volatility and Heston), exposed through a local web dashboard.

It is split into sub-projects, each with its own design → plan → implementation cycle:

1. **BS core + European options** ← *this document*
2. American options (LCP: PSOR / Brennan–Schwartz; Longstaff–Schwartz)
3. Barrier and digital options (pinned nodes, Brownian bridge)
4. Local volatility and Heston (Euler/Milstein, 2D PDE, multilevel)
5. Dashboard enhancements

Sub-project 1 lays down the architecture every later sub-project reuses. For a European call or put under constant-volatility BS, it delivers:
- an MC pricer with the 5 variance reduction techniques from G. Pagès' course;
- a Crank–Nicolson pricer with Rannacher start-up;
- delta and gamma from both methods;
- convergence studies measured against the closed form;
- a Streamlit dashboard.

**Success criterion:** measured convergence slopes match theory (−1/2 for MC vs N, −2 for CN vs N_x). After clicking **SOLVE**, the dashboard shows prices, Greeks, errors, runtimes and convergence curves for both methods.

## 2. Repository layout

```
pricing/                      # numerical core, no UI dependency
  models/black_scholes.py     # BlackScholes + closed-form formulas
  products/vanilla.py         # EuropeanOption
  mc/rng.py                   # generator (PCG64 via numpy.random.Generator)
  mc/engine.py                # price_mc
  mc/variance_reduction.py    # the 5 techniques
  mc/greeks.py                # pathwise / likelihood ratio estimators
  pde/grid.py                 # log(S) grid with pinned nodes
  pde/solver.py               # θ-scheme, Crank–Nicolson + Rannacher, price_pde
  pde/greeks.py               # delta/gamma read off the grid
  results.py                  # PricingResult
  convergence.py              # refinement studies, slope regression
app/dashboard.py              # Streamlit
tests/                        # pytest
docs/specs/                   # design documents
pyproject.toml
README.md
```

Principles:
- **Model / product / method are separate.** A model describes dynamics, a product its payoff, and a method combines the two. Adding a model or a product never requires changing the engines.
- **Common result type** (`PricingResult`) for both methods, so the dashboard and convergence studies treat them identically.
- **The closed form is the referee** used to measure exact errors.

Conventions:
- Everything in the repository is in English (code, docstrings, README, docs).
- Comments in code stay light; explanations live in the conversation and in the personal, unversioned `GUIDE.md`.

Environment: Python 3.14, local venv, editable install (`pip install -e .`). Dependencies: `numpy`, `scipy`, `pandas`, `plotly`, `streamlit`, `pytest`.

## 3. Interfaces

```python
@dataclass(frozen=True)
class BlackScholes:
    s0: float; r: float; sigma: float; q: float = 0.0
    def closed_form(self, product) -> PricingResult: ...   # exact price, delta, gamma

@dataclass(frozen=True)
class EuropeanOption:
    strike: float; maturity: float; kind: Literal["call", "put"]
    def payoff(self, s: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 100_000
    variance_reduction: Literal["none", "antithetic", "control_variate",
                                "importance_sampling", "stratification",
                                "conditioning"] = "none"
    delta_method: Literal["pathwise", "likelihood_ratio"] = "pathwise"
    gamma_method: Literal["likelihood_ratio", "pathwise_lr", "pathwise"] = "likelihood_ratio"
    seed: int | None = 42
    is_drift: float | None = None        # importance sampling; None → heuristic
    n_strata: int = 100                  # stratification
    conditioning_time: float = 0.5       # pre-conditioning: fraction θ of T

@dataclass(frozen=True)
class PDEConfig:
    n_space: int = 400        # target number of x-intervals
    n_time: int = 50          # number of time steps (N_t/N_x = 1/8, see §6)
    width: float = 5.0        # domain half-width, in std devs of log(S_T)
    rannacher_steps: int = 2  # 0 = pure Crank–Nicolson
    theta: float = 0.5        # 0.5 = CN, 1 = implicit Euler

@dataclass(frozen=True)
class PricingResult:
    method: str
    price: float; delta: float; gamma: float
    elapsed: float                        # seconds (time.perf_counter)
    price_stderr: float | None = None     # MC only
    delta_stderr: float | None = None
    gamma_stderr: float | None = None
    extra: dict = field(default_factory=dict)   # e.g. PDE grid, estimated β, μ used…

def price_mc(model, product, config: MCConfig) -> PricingResult
def price_pde(model, product, config: PDEConfig) -> PricingResult
```

Any invalid configuration raises an explicit `ValueError`: `n_paths` ≤ 0, `sigma` ≤ 0, `gamma_method="pathwise"` on a call or put (the payoff is not twice differentiable), `theta` outside [0, 1], etc.

## 4. Closed form (the referee)

d₁ = [ln(S₀/K) + (r − q + σ²/2)T] / (σ√T), d₂ = d₁ − σ√T

| | Call | Put |
|---|---|---|
| Price | S₀e^{−qT}Φ(d₁) − Ke^{−rT}Φ(d₂) | Ke^{−rT}Φ(−d₂) − S₀e^{−qT}Φ(−d₁) |
| Delta | e^{−qT}Φ(d₁) | −e^{−qT}Φ(−d₁) |
| Gamma | e^{−qT}φ(d₁) / (S₀σ√T) | same |

## 5. Monte Carlo engine

**Exact simulation:** S_T = S₀·exp((r − q − σ²/2)T + σ√T·Z), with Z ~ N(0,1). There is no discretisation bias for a European option under BS.

**Fair counting:** `n_paths` = number of **payoff evaluations**, whatever the technique. With antithetics this means n_paths/2 pairs. The actual cost is measured by `elapsed`, and efficiency by variance × time.

**Crude estimator:** P̂ = mean of X = e^{−rT}f(S_T), and the 95% CI is P̂ ± 1.96·σ̂/√N.

**Variance reduction** (one technique at a time):

| Technique | Estimator | Standard error |
|---|---|---|
| `antithetic` | Mean of (X(Z) + X(−Z))/2 over N/2 pairs | Computed from the N/2 pair means |
| `control_variate` | X̄ − β̂(Ȳ − E[Y]) with Y = e^{−rT}S_T, E[Y] = S₀e^{−qT}, and β̂ = Ĉov(X,Y)/V̂ar(Y) estimated on the same sample (O(1/N) bias, negligible) | Empirical std of the residuals X − β̂(Y − E[Y]) |
| `importance_sampling` | Draw Y ~ N(μ, 1), average X(Y)·w(Y) with w(Y) = exp(−μY + μ²/2) (see below) | Std of X·w |
| `stratification` | L = `n_strata` equiprobable strata of N(0,1), proportional allocation N/L, Z = Φ⁻¹((i + U)/L) | √(Σᵢ sᵢ²/nᵢ)/L, where sᵢ is the within-stratum std |
| `conditioning` | Simulate S_{θT} exactly, then X = e^{−rθT}·BS(S_{θT}, K, (1−θ)T) (exact conditional expectation) | Empirical std |

**Importance sampling, precise definition.** We want E[g(Z)] with Z ~ N(0,1). We draw Y ~ N(μ, 1) and use E[g(Z)] = E[g(Y)·exp(−μY + μ²/2)].
- By default (`is_drift=None`) we take μ = −d₂, the value of Z for which S_T = K. For a call it is truncated to μ ≥ 0, for a put to μ ≤ 0, so the distribution is only shifted when the option is out of the money.
- Adaptive tuning of μ by stochastic gradient (Robbins–Monro) is planned for a later iteration.

Constraints: if N is not divisible by L (stratification) or is odd (antithetics), it is rounded down to the nearest compatible integer, and the value actually used is reported in `extra`.

**MC Greeks**, computed on the **same draws** as the price (Z is the Gaussian actually used for S_T):

| Estimator | Delta | Gamma |
|---|---|---|
| `pathwise` | e^{−rT}·f'(S_T)·S_T/S₀, with f' = 𝟙{S_T>K} (call) or −𝟙{S_T<K} (put) | **Undefined**: raises an error |
| `likelihood_ratio` | e^{−rT}·f(S_T)·Z/(S₀σ√T) | e^{−rT}·f(S_T)·(Z² − 1 − Zσ√T)/(S₀²σ²T) |
| `pathwise_lr` (mixed) | — | e^{−rT}·f'(S_T)·(S_T/S₀²)·(Z/(σ√T) − 1) |

How variance reduction applies to the Greeks:
- **antithetics and stratification** carry over naturally, since they only change the draws;
- **importance sampling**: multiply by the weight w;
- **control variate and pre-conditioning** apply **to the price only** in this sub-project. The Greeks are then estimated on standard draws of the same size, and this is flagged in `extra`.

## 6. PDE engine

**Change of variables:** x = ln S and τ = T − t (time to maturity). The coefficients become constant:

u_τ = ½σ²·u_xx + (r − q − ½σ²)·u_x − r·u,  with u(x, 0) = f(eˣ)

**Grid (`grid.py`):**
- The domain is [x₀ − L, x₀ + L], with x₀ = ln S₀ and L = `width`·σ√T.
- The target step is Δx* = 2L/`n_space`.
- **Pinned nodes.** Let d = |ln K − x₀|.
  - If d = 0, the strike is already on node x₀.
  - If d ≥ Δx*/2, take Δx = d/⌈d/Δx*⌉, which satisfies Δx ∈ [Δx*/2, Δx*]. **Both S₀ and K are then nodes**, and no interpolation is needed.
  - If 0 < d < Δx*/2, take Δx = Δx* and replace the initial condition by the **exact average of the payoff over each cell** [xⱼ − Δx/2, xⱼ + Δx/2]. This is the other classical remedy for non-smooth payoffs (Pooley–Vetzal–Forsyth, 2003).
- The nodes are xⱼ = x₀ + jΔx for j = −J…J, with J = ⌈L/Δx⌉. The grid is therefore symmetric and S₀ is always the central node.

**Boundary conditions** (asymptotic Dirichlet, at time τ):

| | Lower boundary (x_min) | Upper boundary (x_max) |
|---|---|---|
| Call | 0 | S_max·e^{−qτ} − K·e^{−rτ} |
| Put | K·e^{−rτ} − S_min·e^{−qτ} | 0 |

**θ-scheme:** L is the centred discrete operator

(Lu)ⱼ = a·(uⱼ₊₁ − 2uⱼ + uⱼ₋₁)/Δx² + b·(uⱼ₊₁ − uⱼ₋₁)/(2Δx) − r·uⱼ,  with a = ½σ² and b = r − q − ½σ².

At each step: (I − θΔτL)·uⁿ⁺¹ = (I + (1−θ)ΔτL)·uⁿ + boundary terms. The tridiagonal system is solved with `scipy.linalg.solve_banded`, in O(N_x) per step.

**Rannacher:** each of the first `rannacher_steps` steps is replaced by **two implicit Euler half-steps** (θ = 1, step Δτ/2). This damps the high-frequency modes created by the payoff kink, which Crank–Nicolson (θ = ½) does not damp and which pollute delta and gamma.

*Prototype finding:* the damage from pure CN depends on the ratio Δτ/Δx. With N_t = N_x/2 (σ = 20%), CN damps the kink on its own over its many steps and Rannacher changes nothing visible. With N_t = N_x/8, the pure-CN gamma error **grows** under refinement (measured slope ≈ +0.9), while Rannacher restores slope −2. The default is therefore N_t/N_x = 1/8: it is a realistic setting (time error constants are small) and it makes the Rannacher effect visible on the dashboard.

**PDE Greeks**, at the central node j = 0:
- u_x ≈ (u₁ − u₋₁)/(2Δx) and u_xx ≈ (u₁ − 2u₀ + u₋₁)/Δx²;
- Δ = u_x/S₀ and Γ = (u_xx − u_x)/S₀².

The full grid solution (S, u, and nodewise delta and gamma) is returned in `extra`, so the gamma profile can be plotted.

## 7. Convergence studies (`convergence.py`)

- `mc_convergence(model, product, config, n_values) -> DataFrame`: one row per N, with columns N, price, error, standard error, delta and gamma errors, time.
- `pde_convergence(model, product, config, n_space_values) -> DataFrame`: `n_space` and `n_time` grow together, keeping the base config's ratio (with `n_time` never below `rannacher_steps`). Columns: N_x (target), actual node count, Δx, N_t, price, price/delta/gamma errors, time. Because pinning adjusts Δx, **slopes are fitted against the actual node count**, not the target N_x (prototype: −2.02 vs −1.91 for K = 110).
- `fit_slope(x, y) -> float`: slope of a least-squares fit on (log x, log |y|).
- Timing: each point takes the **minimum over 3 runs**, which reduces measurement noise.

Expected theoretical slopes:

| Quantity | Expected slope |
|---|---|
| MC CI half-width vs N | −1/2 for every variance reduction technique (only the level changes) |
| CN price error vs node count (with Rannacher) | −2 |
| CN error vs CPU time | −1 |
| MC error vs CPU time | −1/2 |

## 8. Dashboard (`app/dashboard.py`)

**Sidebar:**
- model: S₀, r, q, σ;
- product: call or put, K, T;
- MC: N, variance reduction technique, delta and gamma estimators, seed, and technique-specific parameters (μ, L, θ);
- PDE: N_x, N_t, width, Rannacher steps, θ;
- studies: N_max for MC, number of refinement levels for the PDE.

**SOLVE button.** Runs both pricers, the closed form, and both convergence studies.

**Display:**
1. **Summary table**: price, delta and gamma from the closed form, MC (± CI) and PDE; absolute errors; runtimes.
2. **MC convergence tab**: CI half-width and actual error vs N (log-log). Several variance reduction techniques can be overlaid (multi-select). Measured slopes are shown, along with an efficiency table (variance × time, relative to `none`).
3. **PDE convergence tab**: price, delta and gamma errors vs node count (log-log), with measured slopes, comparing with and without Rannacher.
4. **MC vs PDE tab**: error vs CPU time, both methods on the same axes.
5. **PDE Greek profile tab**: Γ(S) on the grid against the exact Γ, with and without Rannacher.

Charts use Plotly. Computations are cached with `st.cache_data` so switching tabs does not rerun anything. The parameters are snapshotted when SOLVE is clicked: editing the sidebar afterwards does nothing until the next SOLVE.

The MC efficiency of a technique is σ̂²·time, i.e. (variance per path) × (cost per path). This quantity does not depend on N. The gain relative to `none` is the ratio of the two efficiencies.

## 9. Tests (pytest, written before the code)

| Test | Property |
|---|---|
| Closed form: S=42, K=40, r=10%, σ=20%, T=0.5 gives call ≈ 4.7594 and put ≈ 0.8086 (Hull) | Correct referee |
| Put–call parity for the closed form, MC (within 4σ) and PDE (within 10⁻⁴) | Consistency |
| MC, every technique, call and put: \|P̂ − P\| < 4·σ̂/√N | Unbiasedness |
| MC, every technique except `none`: σ̂ lower than `none` at equal N, on a case where the technique is relevant (IS: deep out-of-the-money call) | Effective reduction |
| MC: same seed gives the same result | Reproducibility |
| MC: CI half-width slope ∈ [−0.6, −0.4] | 1/√N convergence |
| PDE: price error < 10⁻⁴ on a fine grid | Accuracy |
| PDE, pure CN with N_t = N_x/2: price and gamma slopes ∈ [−2.2, −1.8] | Second order |
| PDE with N_t = N_x/8: Rannacher gives slopes ∈ [−2.2, −1.8]; pure CN gamma error grows | Rannacher is needed |
| Delta and gamma, MC (within 4σ) and PDE (within 10⁻³), against closed forms | Greeks |
| `gamma_method="pathwise"` raises a `ValueError` | Theoretical limit |
| Grid: S₀ and K are nodes when d ≥ Δx*/2 | Pinning |
| Invalid configurations raise `ValueError` | Robustness |

## 10. Out of scope (later sub-projects)

- American, barrier and digital options.
- Local volatility, Heston, Euler/Milstein schemes, multilevel, Richardson–Romberg extrapolation.
- Quasi-Monte Carlo (Sobol, Halton), stochastic gradient for importance sampling.
- Combinations of variance reduction techniques; applying the control variate and pre-conditioning to the Greeks.
- Vega, theta, rho.
- Numba: not needed here, since everything vectorises in NumPy and the PDE time loop only costs O(N_t) calls to a compiled routine.
