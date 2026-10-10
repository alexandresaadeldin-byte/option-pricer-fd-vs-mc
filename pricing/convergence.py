from collections.abc import Callable, Sequence
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pricing.mc.engine import MCConfig, price_mc
from pricing.models.black_scholes import BlackScholes
from pricing.pde.solver import PDEConfig, price_pde
from pricing.products.vanilla import AmericanOption, VanillaOption
from pricing.reference import reference_price
from pricing.results import PricingResult


def fit_slope(x: ArrayLike, y: ArrayLike) -> float:
    return float(np.polyfit(np.log(x), np.log(np.abs(y)), 1)[0])


def _best_of(pricer: Callable[[], PricingResult], repeats: int) -> tuple[PricingResult, float]:
    results = [pricer() for _ in range(repeats)]
    return results[-1], min(r.elapsed for r in results)


def _warm_up(pricer: Callable[[], PricingResult], seconds: float = 0.05) -> None:
    """Run untimed until the CPU has left its idle state and caches are hot (otherwise small runs look slow)."""
    start = perf_counter()
    while perf_counter() - start < seconds:
        pricer()


def mc_convergence(model: BlackScholes, product: VanillaOption, config: MCConfig, n_values: Sequence[int],
                   repeats: int = 3) -> pd.DataFrame:
    exact = reference_price(model, product)
    rows = []
    _warm_up(lambda: price_mc(model, product, replace(config, n_paths=int(min(n_values)))))
    for n in n_values:
        res, elapsed = _best_of(lambda: price_mc(model, product, replace(config, n_paths=int(n))), repeats)
        rows.append({
            "n_paths": res.extra["n_paths_used"],
            "price": res.price,
            "error": abs(res.price - exact.price),
            "stderr": res.price_stderr,
            "ci_half_width": 1.96 * res.price_stderr,
            "delta_error": abs(res.delta - exact.delta),
            "gamma_error": abs(res.gamma - exact.gamma),
            "elapsed": elapsed,
        })
    return pd.DataFrame(rows)


def pde_convergence(model: BlackScholes, product: VanillaOption, config: PDEConfig,
                    n_space_values: Sequence[int], repeats: int = 3) -> pd.DataFrame:
    exact = reference_price(model, product)
    ratio = config.n_time / config.n_space
    rows = []

    def config_for(n):
        return replace(config, n_space=int(n), n_time=max(1, config.rannacher_steps, round(n * ratio)))

    _warm_up(lambda: price_pde(model, product, config_for(min(n_space_values))))
    for n in n_space_values:
        cfg = config_for(n)
        res, elapsed = _best_of(lambda: price_pde(model, product, cfg), repeats)
        rows.append({
            "n_space": cfg.n_space,
            "n_nodes": res.extra["n_nodes"],
            "dx": res.extra["dx"],
            "n_time": cfg.n_time,
            "price": res.price,
            "error": abs(res.price - exact.price),
            "delta_error": abs(res.delta - exact.delta),
            "gamma_error": abs(res.gamma - exact.gamma),
            "elapsed": elapsed,
        })
    return pd.DataFrame(rows)


def exercise_dates_study(model: BlackScholes, product: AmericanOption, config: MCConfig,
                         n_steps_values: Sequence[int]) -> pd.DataFrame:
    """Longstaff-Schwartz price as a function of the number of exercise dates (Bermudan bias)."""
    exact = reference_price(model, product)
    rows = []
    for n_steps in n_steps_values:
        res = price_mc(model, product, replace(config, n_steps=int(n_steps)))
        rows.append({
            "n_steps": int(n_steps),
            "price": res.price,
            "stderr": res.price_stderr,
            "bias": res.price - exact.price,
            "elapsed": res.elapsed,
        })
    return pd.DataFrame(rows)
