from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import solve_banded

from pricing.pde.greeks import grid_greeks
from pricing.pde.grid import build_grid
from pricing.results import PricingResult


@dataclass(frozen=True)
class PDEConfig:
    n_space: int = 400
    n_time: int = 50
    width: float = 5.0
    rannacher_steps: int = 2
    theta: float = 0.5

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


def _boundaries(model, product, s_min, s_max, tau):
    k_disc = product.strike * np.exp(-model.r * tau)
    q_disc = np.exp(-model.q * tau)
    if product.is_call:
        return 0.0, s_max * q_disc - k_disc
    return k_disc - s_min * q_disc, 0.0


def _time_steps(config, maturity):
    dt = maturity / config.n_time
    steps = [(1.0, dt / 2)] * (2 * config.rannacher_steps)
    steps += [(config.theta, dt)] * (config.n_time - config.rannacher_steps)
    return steps


def price_pde(model, product, config: PDEConfig = PDEConfig()) -> PricingResult:
    start = perf_counter()
    grid = build_grid(model.s0, product.strike, model.sigma, product.maturity,
                      config.n_space, config.width)
    x, dx, s = grid.x, grid.dx, grid.s

    if grid.strike_pinned:
        u = product.payoff(s)
    else:
        u = product.log_cell_average(x - dx / 2, x + dx / 2)

    a = 0.5 * model.sigma**2
    b = model.r - model.q - 0.5 * model.sigma**2
    lower = a / dx**2 - b / (2 * dx)
    diag = -2 * a / dx**2 - model.r
    upper = a / dx**2 + b / (2 * dx)
    n_in = x.size - 2

    matrices = {}
    tau = 0.0
    left, right = _boundaries(model, product, s[0], s[-1], tau)
    u[0], u[-1] = left, right
    for theta, h in _time_steps(config, product.maturity):
        if (theta, h) not in matrices:
            ab = np.empty((3, n_in))
            ab[0, :] = -theta * h * upper
            ab[1, :] = 1 - theta * h * diag
            ab[2, :] = -theta * h * lower
            matrices[(theta, h)] = ab
        lu = lower * u[:-2] + diag * u[1:-1] + upper * u[2:]
        rhs = u[1:-1] + (1 - theta) * h * lu
        tau += h
        left, right = _boundaries(model, product, s[0], s[-1], tau)
        rhs[0] += theta * h * lower * left
        rhs[-1] += theta * h * upper * right
        u = np.concatenate(([left], solve_banded((1, 1), matrices[(theta, h)], rhs), [right]))

    delta_profile, gamma_profile = grid_greeks(s, u, dx)
    j = grid.center - 1
    extra = {
        "s": s,
        "u": u,
        "delta_profile": delta_profile,
        "gamma_profile": gamma_profile,
        "dx": dx,
        "n_nodes": int(x.size),
        "strike_pinned": grid.strike_pinned,
    }
    return PricingResult(
        "finite_differences", float(u[grid.center]), float(delta_profile[j]),
        float(gamma_profile[j]), perf_counter() - start, extra=extra,
    )
