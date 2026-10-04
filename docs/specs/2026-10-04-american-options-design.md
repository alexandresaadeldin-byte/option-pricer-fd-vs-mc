# Sub-project 2 — American options

*Design approved on 2026-10-04.*

## Goal

Price American calls and puts under Black–Scholes with both engines, and compare them when no closed form exists:

- **Finite differences:** the θ-scheme of sub-project 1 turns into a **linear complementarity problem (LCP)** at each time step. It is solved by **PSOR** (iterative) and by **Brennan–Schwartz** (direct), so that the two can be compared.
- **Monte Carlo:** the **Longstaff–Schwartz** algorithm (least-squares regression of the continuation value).
- **Reference price:** a Richardson-extrapolated Brennan–Schwartz solution on very fine grids, since there is no closed form.

## Interfaces

- `AmericanOption(strike, maturity, kind)`: same fields and payoff as `EuropeanOption`. Both share a `_Vanilla` base class, but neither is an instance of the other, so a European pricer can never silently price an American option.
- `price_pde(model, option, PDEConfig)` dispatches on the product type. `PDEConfig` gains `lcp_solver` (`"brennan_schwartz"` | `"psor"`), `omega` (1.6), `tol` (1e-9) and `max_iter` (10 000), all ignored for European options.
- `price_mc(model, option, MCConfig)` dispatches to Longstaff–Schwartz. `MCConfig` gains `n_steps` (50 exercise dates) and `basis_degree` (3). For American options, only `variance_reduction="none"` is supported.
- `reference_price(model, product) -> PricingResult` returns the closed form (European) or the Richardson-extrapolated fine PDE (American, cached). The convergence studies and the dashboard measure errors against it.

## Finite differences

At each step, with obstacle g = intrinsic value on the interior nodes:

  A uⁿ⁺¹ ≥ b,  uⁿ⁺¹ ≥ g,  (A uⁿ⁺¹ − b)·(uⁿ⁺¹ − g) = 0.

- **Brennan–Schwartz:** Gaussian elimination, then a substitution pass with projection u_i = max(u_i, g_i). The substitution must **start inside the exercise region**: from high S for a call (standard LU), from low S for a put (the system is reversed). It is exact for vanilla payoffs, where the exercise region is a single interval, and costs O(N).
- **PSOR:** projected Gauss–Seidel with over-relaxation ω, warm-started from the previous time step, stopped when the max update falls below `tol`. It is general but iterative, and the number of iterations is reported.
- Both loops are compiled with **Numba**, so that runtime comparisons measure the algorithms rather than the Python interpreter.
- Boundaries: put: u(S_min) = K − S_min (deep in the money: exercise), u(S_max) = 0. Call: u(S_min) = 0, u(S_max) = max(S_max e^{−qτ} − K e^{−rτ}, S_max − K).
- Rannacher start-up as in sub-project 1. The **early-exercise boundary** S*(t) is recorded at every step.

## Monte Carlo — Longstaff–Schwartz

- Exact simulation of BS paths on `n_steps` equally spaced dates (`BlackScholes.simulate_paths`).
- **Two independent path sets:** regression coefficients are fitted on a training set (backward induction, regression on in-the-money paths only, basis 1, x, …, x^d with x = S/K). The price is then estimated on a fresh pricing set by applying that exercise rule forward. This makes the estimator a **lower bound** (a sub-optimal rule), with a valid TCL confidence interval. Fitting and pricing on the same paths would introduce a foresight bias in the opposite direction.
- Two sources of downward bias are documented: a finite number of exercise dates (Bermudan rather than American) and a sub-optimal regressed policy.
- Delta is computed **pathwise** with the stopping rule held fixed (envelope theorem): e^{−rτ} f'(S_τ) S_τ/S₀. Gamma is not provided (NaN), because the pathwise gamma is undefined and the likelihood-ratio gamma is too noisy for an optimal stopping problem.

## Tests

- **Literature benchmarks** (Longstaff–Schwartz 2001, Table 1: K = 40, r = 6%, T = 1, σ = 20%): put at S₀ = 36 / 40 / 44 equals 4.478 / 2.314 / 1.110.
- An American call without dividends equals the European call (no early exercise). With q > 0 it is strictly above.
- PSOR and Brennan–Schwartz agree to 1e-6 on the same grid. The American put is ≥ the European put and ≥ the intrinsic value everywhere on the grid.
- The PDE converges to the reference, and the measured order is reported.
- LSM: price below reference + 4σ̂ (lower bound) and within 4σ̂ + 0.03 of it at 50 dates; same seed gives the same result.
- The early-exercise boundary of the put is increasing in t (decreasing in time to maturity) and stays below K.

## Dashboard

A product selector (European / American). For American options: LSM parameters and LCP solver choice, a convergence comparison of PSOR vs Brennan–Schwartz, LSM convergence in the number of paths and of exercise dates, the cost comparison against the PDE, and a plot of the early-exercise boundary.
