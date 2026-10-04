from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from pricing.mc import greeks as mc_greeks
from pricing.mc import variance_reduction as vr
from pricing.mc.rng import make_rng
from pricing.results import PricingResult

VR_TECHNIQUES = (
    "none",
    "antithetic",
    "control_variate",
    "importance_sampling",
    "stratification",
    "conditioning",
)
PRICE_ONLY_TECHNIQUES = ("control_variate", "conditioning")


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 100_000
    variance_reduction: Literal[
        "none", "antithetic", "control_variate",
        "importance_sampling", "stratification", "conditioning",
    ] = "none"
    delta_method: Literal["pathwise", "likelihood_ratio"] = "pathwise"
    gamma_method: Literal["likelihood_ratio", "pathwise_lr", "pathwise"] = "likelihood_ratio"
    seed: int | None = 42
    is_drift: float | None = None
    n_strata: int = 100
    conditioning_time: float = 0.5

    def __post_init__(self):
        if self.n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if self.variance_reduction not in VR_TECHNIQUES:
            raise ValueError(f"variance_reduction must be one of {VR_TECHNIQUES}")
        if self.n_strata < 1:
            raise ValueError("n_strata must be at least 1")
        if self.variance_reduction == "stratification" and self.n_paths < 2 * self.n_strata:
            raise ValueError("stratification needs at least 2 paths per stratum")
        if not 0 < self.conditioning_time < 1:
            raise ValueError("conditioning_time must be in (0, 1)")


def _draw(model, product, config, rng, extra):
    technique = config.variance_reduction
    if technique == "antithetic":
        return vr.antithetic_sample(rng, config.n_paths)
    if technique == "stratification":
        return vr.stratified_sample(rng, config.n_paths, config.n_strata)
    if technique == "importance_sampling":
        mu = vr.default_is_drift(model, product) if config.is_drift is None else config.is_drift
        extra["is_drift"] = mu
        return vr.importance_sample(rng, config.n_paths, mu)
    return vr.plain_sample(rng, config.n_paths)


def price_mc(model, product, config: MCConfig = MCConfig()) -> PricingResult:
    mc_greeks.check_methods(config.delta_method, config.gamma_method)
    start = perf_counter()
    rng = make_rng(config.seed)
    technique = config.variance_reduction
    extra = {"variance_reduction": technique}
    discount = np.exp(-model.r * product.maturity)

    sample = _draw(model, product, config, rng, extra)
    s_t = model.terminal_value(sample.z, product.maturity)

    if technique == "conditioning":
        values = vr.conditional_prices(model, product, rng, config.n_paths, config.conditioning_time)
        price, price_se = vr.mean_and_stderr(values)
    elif technique == "control_variate":
        payoff = discount * product.payoff(s_t)
        forward = model.s0 * np.exp(-model.q * product.maturity)
        price, price_se, beta = vr.control_variate_estimate(payoff, discount * s_t, forward)
        extra["beta"] = beta
    else:
        price, price_se = sample.aggregate(discount * product.payoff(s_t) * sample.weights)

    delta_x = mc_greeks.delta_samples(model, product, s_t, sample.z, config.delta_method)
    gamma_x = mc_greeks.gamma_samples(model, product, s_t, sample.z, config.gamma_method)
    delta, delta_se = sample.aggregate(delta_x * sample.weights)
    gamma, gamma_se = sample.aggregate(gamma_x * sample.weights)

    if technique in PRICE_ONLY_TECHNIQUES:
        extra["greeks_variance_reduction"] = "none"
    extra["n_paths_used"] = int(sample.z.size)
    return PricingResult(
        "monte_carlo", price, delta, gamma, perf_counter() - start,
        price_se, delta_se, gamma_se, extra,
    )
