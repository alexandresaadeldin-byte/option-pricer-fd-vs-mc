import numpy as np

from pricing.models.black_scholes import BlackScholes
from pricing.products.vanilla import EuropeanOption

DELTA_METHODS = ("pathwise", "likelihood_ratio")
GAMMA_METHODS = ("likelihood_ratio", "pathwise_lr", "pathwise")


def check_methods(delta_method: str, gamma_method: str) -> None:
    if delta_method not in DELTA_METHODS:
        raise ValueError(f"delta_method must be one of {DELTA_METHODS}")
    if gamma_method not in GAMMA_METHODS:
        raise ValueError(f"gamma_method must be one of {GAMMA_METHODS}")
    if gamma_method == "pathwise":
        raise ValueError(
            "pathwise gamma is undefined for a vanilla payoff: the payoff derivative jumps "
            "at the strike, so its second derivative is a Dirac mass. "
            "Use 'likelihood_ratio' or 'pathwise_lr'."
        )


def delta_samples(model: BlackScholes, product: EuropeanOption, s_t: np.ndarray, z: np.ndarray,
                  method: str) -> np.ndarray:
    discount = np.exp(-model.r * product.maturity)
    vol = model.sigma * np.sqrt(product.maturity)
    if method == "pathwise":
        return discount * product.payoff_derivative(s_t) * s_t / model.s0
    return discount * product.payoff(s_t) * z / (model.s0 * vol)


def gamma_samples(model: BlackScholes, product: EuropeanOption, s_t: np.ndarray, z: np.ndarray,
                  method: str) -> np.ndarray:
    discount = np.exp(-model.r * product.maturity)
    vol = model.sigma * np.sqrt(product.maturity)
    if method == "likelihood_ratio":
        return discount * product.payoff(s_t) * (z**2 - 1 - z * vol) / (model.s0**2 * vol**2)
    return discount * product.payoff_derivative(s_t) * s_t / model.s0**2 * (z / vol - 1)
