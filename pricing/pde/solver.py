from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

from pricing.pde import scheme
from pricing.pde.american import price_american_pde
from pricing.products.vanilla import AmericanOption, EuropeanOption
from pricing.results import PricingResult

LCP_SOLVERS = ("brennan_schwartz", "psor")


@dataclass(frozen=True)
class PDEConfig:
    n_space: int = 400
    n_time: int = 50
    width: float = 5.0
    rannacher_steps: int = 2
    theta: float = 0.5
    lcp_solver: Literal["brennan_schwartz", "psor"] = "brennan_schwartz"
    omega: float = 1.6
    tol: float = 1e-9
    max_iter: int = 10_000

    def __post_init__(self):
        if self.n_space < 4:
            raise ValueError("n_space must be at least 4")
        if self.n_time < 1:
            raise ValueError("n_time must be at least 1")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if not 0 <= self.rannacher_steps <= self.n_time:
            raise ValueError("rannacher_steps must be in [0, n_time]")
        if not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if self.lcp_solver not in LCP_SOLVERS:
            raise ValueError(f"lcp_solver must be one of {LCP_SOLVERS}")
        if not 0 < self.omega < 2:
            raise ValueError("omega must be in (0, 2)")


def _european_boundaries(model, option, s_min, s_max):
    def values(tau):
        k_disc = option.strike * np.exp(-model.r * tau)
        q_disc = np.exp(-model.q * tau)
        if option.is_call:
            return 0.0, s_max * q_disc - k_disc
        return k_disc - s_min * q_disc, 0.0

    return values


def _solve_tridiagonal(sub, main, sup, rhs, u_old):
    ab = np.zeros((3, main.size))
    ab[0, 1:] = sup
    ab[1, :] = main
    ab[2, :-1] = sub
    return solve_banded((1, 1), ab, rhs)


def price_european_pde(model, option, config) -> PricingResult:
    start = perf_counter()
    grid, u0, operator = scheme.setup(model, option, config)
    boundaries = _european_boundaries(model, option, grid.s[0], grid.s[-1])
    u = u0
    for _, u in scheme.march(grid, u0, operator, scheme.time_steps(config, option.maturity),
                             boundaries, _solve_tridiagonal):
        pass
    price, delta, gamma, extra = scheme.result_extra(grid, u)
    return PricingResult("finite_differences", price, delta, gamma, perf_counter() - start, extra=extra)


def price_pde(model, product, config: PDEConfig = PDEConfig()) -> PricingResult:
    if isinstance(product, AmericanOption):
        return price_american_pde(model, product, config)
    if isinstance(product, EuropeanOption):
        return price_european_pde(model, product, config)
    raise TypeError(f"no finite-difference pricer for {type(product).__name__}")
