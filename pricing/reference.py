"""Reference ("exact") prices used to measure the error of each method."""

from dataclasses import replace
from functools import lru_cache

from pricing.pde.solver import PDEConfig, price_pde
from pricing.products.vanilla import AmericanOption, EuropeanOption

AMERICAN_REFERENCE_GRID = PDEConfig(n_space=8000, n_time=4000)


@lru_cache(maxsize=128)
def reference_price(model, product):
    """Closed form for European options; very fine Brennan–Schwartz PDE (error ~1e-5) for American options."""
    if isinstance(product, EuropeanOption):
        return model.closed_form(product)
    if isinstance(product, AmericanOption):
        res = price_pde(model, product, AMERICAN_REFERENCE_GRID)
        return replace(res, method="reference_pde", extra={"grid": "8000 x 4000, Brennan-Schwartz"})
    raise TypeError(f"no reference price for {type(product).__name__}")
