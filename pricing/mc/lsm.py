"""Longstaff–Schwartz least-squares Monte Carlo for American options."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from pricing.mc.rng import make_rng
from pricing.mc.variance_reduction import mean_and_stderr
from pricing.models.black_scholes import BlackScholes
from pricing.products.vanilla import AmericanOption
from pricing.results import PricingResult

if TYPE_CHECKING:
    from pricing.mc.engine import MCConfig


def _basis(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, degree + 1, increasing=True)


def fit_exercise_policy(paths: np.ndarray, option: AmericanOption, r: float, dt: float,
                        degree: int) -> list[np.ndarray | None]:
    """Backward induction on training paths. Returns one coefficient vector per exercise date 1..n_steps-1."""
    n_steps = paths.shape[1] - 1
    cashflow = option.payoff(paths[:, -1])
    cashflow_step = np.full(paths.shape[0], n_steps)
    coefficients = [None] * n_steps
    for k in range(n_steps - 1, 0, -1):
        exercise_value = option.payoff(paths[:, k])
        itm = exercise_value > 0
        if itm.sum() <= degree + 1:
            continue
        discounted = cashflow[itm] * np.exp(-r * dt * (cashflow_step[itm] - k))
        x = _basis(paths[itm, k] / option.strike, degree)
        beta, *_ = np.linalg.lstsq(x, discounted, rcond=None)
        coefficients[k] = beta
        exercise_now = exercise_value[itm] > x @ beta
        idx = np.flatnonzero(itm)[exercise_now]
        cashflow[idx] = exercise_value[idx]
        cashflow_step[idx] = k
    return coefficients


def apply_exercise_policy(paths: np.ndarray, option: AmericanOption,
                          coefficients: list[np.ndarray | None], degree: int) -> np.ndarray:
    """Forward pass: exercise at the first date where intrinsic value beats the regressed continuation."""
    n_paths, n_dates = paths.shape
    n_steps = n_dates - 1
    stop_step = np.full(n_paths, n_steps)
    alive = np.ones(n_paths, dtype=bool)
    for k in range(1, n_steps):
        beta = coefficients[k]
        if beta is None:
            continue
        exercise_value = option.payoff(paths[:, k])
        candidates = alive & (exercise_value > 0)
        continuation = _basis(paths[candidates, k] / option.strike, degree) @ beta
        stop = np.flatnonzero(candidates)[exercise_value[candidates] > continuation]
        stop_step[stop] = k
        alive[stop] = False
    return stop_step


def price_american_mc(model: BlackScholes, option: AmericanOption, config: MCConfig) -> PricingResult:
    start = perf_counter()
    rng = make_rng(config.seed)
    n_steps = config.n_steps
    dt = option.maturity / n_steps

    training = model.simulate_paths(rng.standard_normal((config.n_paths, n_steps)), option.maturity)
    coefficients = fit_exercise_policy(training, option, model.r, dt, config.basis_degree)

    paths = model.simulate_paths(rng.standard_normal((config.n_paths, n_steps)), option.maturity)
    stop_step = apply_exercise_policy(paths, option, coefficients, config.basis_degree)
    rows = np.arange(config.n_paths)
    s_stop = paths[rows, stop_step]
    discount = np.exp(-model.r * dt * stop_step)

    price, price_se = mean_and_stderr(discount * option.payoff(s_stop))
    delta, delta_se = mean_and_stderr(discount * option.payoff_derivative(s_stop) * s_stop / model.s0)
    intrinsic_now = float(option.payoff(model.s0))
    extra = {
        "n_paths_used": config.n_paths,
        "n_steps": n_steps,
        "basis_degree": config.basis_degree,
        "exercised_early": float(np.mean(stop_step < n_steps)),
    }
    if intrinsic_now > price:
        extra["immediate_exercise"] = True
        price, price_se = intrinsic_now, 0.0
    return PricingResult(
        "monte_carlo", price, delta, float("nan"), perf_counter() - start,
        price_se, delta_se, None, extra,
    )
