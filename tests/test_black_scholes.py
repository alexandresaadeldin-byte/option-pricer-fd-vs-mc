import numpy as np
import pytest

from pricing import BlackScholes, EuropeanOption


def test_hull_textbook_values():
    model = BlackScholes(s0=42, r=0.10, sigma=0.20)
    call = model.closed_form(EuropeanOption(40, 0.5, "call"))
    put = model.closed_form(EuropeanOption(40, 0.5, "put"))
    assert call.price == pytest.approx(4.7594, abs=1e-4)
    assert put.price == pytest.approx(0.8086, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    model = BlackScholes(s0=100, r=0.05, sigma=0.25, q=q)
    call = model.closed_form(EuropeanOption(110, 2.0, "call")).price
    put = model.closed_form(EuropeanOption(110, 2.0, "put")).price
    assert call - put == pytest.approx(100 * np.exp(-q * 2) - 110 * np.exp(-0.05 * 2), abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences_of_the_price(kind):
    option = EuropeanOption(105, 0.75, kind)
    h = 1e-3

    def price(s0):
        return BlackScholes(s0=s0, r=0.03, sigma=0.3, q=0.01).closed_form(option).price

    exact = BlackScholes(s0=100, r=0.03, sigma=0.3, q=0.01).closed_form(option)
    assert exact.delta == pytest.approx((price(100 + h) - price(100 - h)) / (2 * h), abs=1e-6)
    assert exact.gamma == pytest.approx((price(100 + h) - 2 * price(100) + price(100 - h)) / h**2, abs=1e-4)


def test_terminal_value_has_the_forward_as_mean():
    model = BlackScholes(s0=100, r=0.05, sigma=0.2, q=0.01)
    z = np.random.default_rng(0).standard_normal(1_000_000)
    assert model.terminal_value(z, 1.0).mean() == pytest.approx(100 * np.exp(0.04), rel=1e-3)


@pytest.mark.parametrize("kwargs", [{"s0": 0, "r": 0.05, "sigma": 0.2}, {"s0": 100, "r": 0.05, "sigma": 0}])
def test_invalid_model_raises(kwargs):
    with pytest.raises(ValueError):
        BlackScholes(**kwargs)
