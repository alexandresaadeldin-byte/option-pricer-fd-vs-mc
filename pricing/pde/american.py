from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from pricing.models.black_scholes import BlackScholes
from pricing.pde import scheme
from pricing.pde.lcp import brennan_schwartz, psor
from pricing.products.vanilla import AmericanOption
from pricing.results import PricingResult

if TYPE_CHECKING:
    from pricing.pde.solver import PDEConfig


def _boundaries(model: BlackScholes, option: AmericanOption, s_min: float,
                s_max: float) -> scheme.Boundaries:
    k, r, q = option.strike, model.r, model.q

    def values(tau):
        if option.is_call:
            return 0.0, max(s_max * np.exp(-q * tau) - k * np.exp(-r * tau), s_max - k)
        return k - s_min, 0.0

    return values


def _exercise_boundary(s: np.ndarray, u: np.ndarray, obstacle: np.ndarray, is_call: bool) -> float:
    exercised = (obstacle > 0) & (u[1:-1] <= obstacle + 1e-10)
    if not exercised.any():
        return np.nan
    s_in = s[1:-1][exercised]
    return float(s_in.min() if is_call else s_in.max())


def price_american_pde(model: BlackScholes, option: AmericanOption, config: PDEConfig) -> PricingResult:
    start = perf_counter()
    grid, u0, operator = scheme.setup(model, option, config)
    obstacle = option.payoff(grid.s[1:-1])
    sweeps = []

    def solve(sub, main, sup, rhs, u_old):
        if config.lcp_solver == "psor":
            x, n_sweeps = psor(sub, main, sup, rhs, obstacle, u_old, config.omega, config.tol, config.max_iter)
            sweeps.append(n_sweeps)
            return x
        return brennan_schwartz(sub, main, sup, rhs, obstacle, option.is_call)

    boundaries = _boundaries(model, option, grid.s[0], grid.s[-1])
    times, frontier = [], []
    u = u0
    for tau, u in scheme.march(grid, u0, operator, scheme.time_steps(config, option.maturity), boundaries, solve):
        times.append(option.maturity - tau)
        frontier.append(_exercise_boundary(grid.s, u, obstacle, option.is_call))

    price, delta, gamma, extra = scheme.result_extra(grid, u)
    extra.update({
        "lcp_solver": config.lcp_solver,
        "exercise_boundary_t": np.array(times),
        "exercise_boundary_s": np.array(frontier),
    })
    if sweeps:
        extra["psor_sweeps_mean"] = float(np.mean(sweeps))
    return PricingResult("finite_differences", price, delta, gamma, perf_counter() - start, extra=extra)
