from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from pricing.products.vanilla import EuropeanOption, VanillaOption
from pricing.results import PricingResult


def _d1_d2(s: ArrayLike, strike: float, maturity: float, r: float, sigma: float, q: float):
    vol = sigma * np.sqrt(maturity)
    d1 = (np.log(s / strike) + (r - q + 0.5 * sigma**2) * maturity) / vol
    return d1, d1 - vol


def bs_price(s: ArrayLike, strike: float, maturity: float, r: float, sigma: float, q: float,
             is_call: bool) -> np.ndarray:
    d1, d2 = _d1_d2(s, strike, maturity, r, sigma, q)
    if is_call:
        return s * np.exp(-q * maturity) * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
    return strike * np.exp(-r * maturity) * norm.cdf(-d2) - s * np.exp(-q * maturity) * norm.cdf(-d1)


def bs_delta(s: ArrayLike, strike: float, maturity: float, r: float, sigma: float, q: float,
             is_call: bool) -> np.ndarray:
    d1, _ = _d1_d2(s, strike, maturity, r, sigma, q)
    if is_call:
        return np.exp(-q * maturity) * norm.cdf(d1)
    return -np.exp(-q * maturity) * norm.cdf(-d1)


def bs_gamma(s: ArrayLike, strike: float, maturity: float, r: float, sigma: float,
             q: float) -> np.ndarray:
    d1, _ = _d1_d2(s, strike, maturity, r, sigma, q)
    return np.exp(-q * maturity) * norm.pdf(d1) / (s * sigma * np.sqrt(maturity))


@dataclass(frozen=True)
class BlackScholes:
    s0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self):
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def terminal_value(self, z: np.ndarray, t: float) -> np.ndarray:
        drift = (self.r - self.q - 0.5 * self.sigma**2) * t
        return self.s0 * np.exp(drift + self.sigma * np.sqrt(t) * z)

    def simulate_paths(self, z: np.ndarray, maturity: float) -> np.ndarray:
        """Exact paths on an equally spaced grid: z has shape (n_paths, n_steps); returns (n_paths, n_steps + 1)."""
        z = np.asarray(z, dtype=float)
        dt = maturity / z.shape[1]
        increments = (self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        return self.s0 * np.exp(log_paths)

    def d1_d2(self, product: VanillaOption) -> tuple[float, float]:
        return _d1_d2(self.s0, product.strike, product.maturity, self.r, self.sigma, self.q)

    def closed_form(self, product: EuropeanOption) -> PricingResult:
        start = perf_counter()
        args = (self.s0, product.strike, product.maturity, self.r, self.sigma, self.q)
        price = float(bs_price(*args, product.is_call))
        delta = float(bs_delta(*args, product.is_call))
        gamma = float(bs_gamma(*args))
        return PricingResult("closed_form", price, delta, gamma, perf_counter() - start)
