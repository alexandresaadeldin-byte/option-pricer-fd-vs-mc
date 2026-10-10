"""θ-scheme time stepping on the log-spot grid, shared by the European and American solvers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

from pricing.models.black_scholes import BlackScholes
from pricing.pde.greeks import grid_greeks
from pricing.pde.grid import LogGrid, build_grid
from pricing.products.vanilla import VanillaOption

if TYPE_CHECKING:
    from pricing.pde.solver import PDEConfig

Operator = tuple[float, float, float]
Boundaries = Callable[[float], tuple[float, float]]


def time_steps(config: PDEConfig, maturity: float) -> list[tuple[float, float]]:
    dt = maturity / config.n_time
    steps = [(1.0, dt / 2)] * (2 * config.rannacher_steps)
    steps += [(config.theta, dt)] * (config.n_time - config.rannacher_steps)
    return steps


def setup(model: BlackScholes, product: VanillaOption,
          config: PDEConfig) -> tuple[LogGrid, np.ndarray, Operator]:
    grid = build_grid(model.s0, product.strike, model.sigma, product.maturity, config.n_space, config.width)
    x, dx = grid.x, grid.dx
    if grid.strike_pinned:
        u0 = product.payoff(grid.s)
    else:
        u0 = product.log_cell_average(x - dx / 2, x + dx / 2)
    a = 0.5 * model.sigma**2
    b = model.r - model.q - 0.5 * model.sigma**2
    operator = (a / dx**2 - b / (2 * dx), -2 * a / dx**2 - model.r, a / dx**2 + b / (2 * dx))
    return grid, u0, operator


def march(grid: LogGrid, u0: np.ndarray, operator: Operator, steps: list[tuple[float, float]],
          boundaries: Boundaries, solve: Callable[..., np.ndarray]) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (tau, u) after each step. `solve(sub, main, sup, rhs, u_old_interior)` returns the new interior."""
    lower, diag, upper = operator
    n_in = grid.x.size - 2
    u = u0.copy()
    tau = 0.0
    u[0], u[-1] = boundaries(tau)
    for theta, h in steps:
        lu = lower * u[:-2] + diag * u[1:-1] + upper * u[2:]
        rhs = u[1:-1] + (1 - theta) * h * lu
        tau += h
        left, right = boundaries(tau)
        rhs[0] += theta * h * lower * left
        rhs[-1] += theta * h * upper * right
        sub = np.full(n_in - 1, -theta * h * lower)
        main = np.full(n_in, 1 - theta * h * diag)
        sup = np.full(n_in - 1, -theta * h * upper)
        u = np.concatenate(([left], solve(sub, main, sup, rhs, u[1:-1]), [right]))
        yield tau, u


def result_extra(grid: LogGrid, u: np.ndarray) -> tuple[float, float, float, dict]:
    delta_profile, gamma_profile = grid_greeks(grid.s, u, grid.dx)
    j = grid.center - 1
    extra = {
        "s": grid.s,
        "u": u,
        "delta_profile": delta_profile,
        "gamma_profile": gamma_profile,
        "dx": grid.dx,
        "n_nodes": int(grid.x.size),
        "strike_pinned": grid.strike_pinned,
    }
    return float(u[grid.center]), float(delta_profile[j]), float(gamma_profile[j]), extra
