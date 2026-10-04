from dataclasses import replace

import numpy as np
import pandas as pd

from pricing.mc.engine import price_mc
from pricing.pde.solver import price_pde
from pricing.reference import reference_price


def fit_slope(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(np.abs(y)), 1)[0])


def _best_of(pricer, repeats):
    results = [pricer() for _ in range(repeats)]
    return results[-1], min(r.elapsed for r in results)


def mc_convergence(model, product, config, n_values, repeats=3) -> pd.DataFrame:
    exact = reference_price(model, product)
    rows = []
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


def pde_convergence(model, product, config, n_space_values, repeats=3) -> pd.DataFrame:
    exact = reference_price(model, product)
    ratio = config.n_time / config.n_space
    rows = []
    for n in n_space_values:
        cfg = replace(config, n_space=int(n), n_time=max(1, config.rannacher_steps, round(n * ratio)))
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


def exercise_dates_study(model, product, config, n_steps_values) -> pd.DataFrame:
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
