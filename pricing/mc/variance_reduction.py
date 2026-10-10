from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from pricing.models.black_scholes import BlackScholes, bs_price
from pricing.products.vanilla import EuropeanOption


def mean_and_stderr(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


@dataclass(frozen=True)
class GaussianSample:
    z: np.ndarray
    weights: np.ndarray
    aggregate: Callable[[np.ndarray], tuple[float, float]]


def plain_sample(rng: np.random.Generator, n: int) -> GaussianSample:
    return GaussianSample(rng.standard_normal(n), np.ones(n), mean_and_stderr)


def antithetic_sample(rng: np.random.Generator, n: int) -> GaussianSample:
    half = n // 2
    z = rng.standard_normal(half)

    def aggregate(x):
        return mean_and_stderr(0.5 * (x[:half] + x[half:]))

    return GaussianSample(np.concatenate([z, -z]), np.ones(2 * half), aggregate)


def control_variate_estimate(x: np.ndarray, y: np.ndarray, y_mean: float) -> tuple[float, float, float]:
    beta = np.cov(x, y)[0, 1] / y.var(ddof=1)
    price, stderr = mean_and_stderr(x - beta * (y - y_mean))
    return price, stderr, float(beta)


def default_is_drift(model: BlackScholes, product: EuropeanOption) -> float:
    _, d2 = model.d1_d2(product)
    mu = -d2
    return float(max(mu, 0.0) if product.is_call else min(mu, 0.0))


def importance_sample(rng: np.random.Generator, n: int, mu: float) -> GaussianSample:
    y = rng.standard_normal(n) + mu
    weights = np.exp(-mu * y + 0.5 * mu**2)
    return GaussianSample(y, weights, mean_and_stderr)


def stratified_sample(rng: np.random.Generator, n: int, n_strata: int) -> GaussianSample:
    per_stratum = n // n_strata
    u = rng.random((n_strata, per_stratum))
    z = norm.ppf((np.arange(n_strata)[:, None] + u) / n_strata).ravel()

    def aggregate(x):
        strata = x.reshape(n_strata, per_stratum)
        price = strata.mean(axis=1).mean()
        variance = np.sum(strata.var(axis=1, ddof=1) / per_stratum) / n_strata**2
        return float(price), float(np.sqrt(variance))

    return GaussianSample(z, np.ones(z.size), aggregate)


def conditional_prices(model: BlackScholes, product: EuropeanOption, rng: np.random.Generator,
                       n: int, theta: float) -> np.ndarray:
    t1 = theta * product.maturity
    s_t1 = model.terminal_value(rng.standard_normal(n), t1)
    remaining = bs_price(s_t1, product.strike, product.maturity - t1,
                         model.r, model.sigma, model.q, product.is_call)
    return np.exp(-model.r * t1) * remaining
