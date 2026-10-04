import numpy as np
import pytest

from pricing import BlackScholes, EuropeanOption, PDEConfig, price_pde

FINE = PDEConfig(n_space=1600, n_time=800)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_fine_grid_matches_closed_form(model, strike, kind):
    option = EuropeanOption(strike, 1.0, kind)
    exact = model.closed_form(option)
    res = price_pde(model, option, FINE)
    assert res.price == pytest.approx(exact.price, abs=1e-4)
    assert res.delta == pytest.approx(exact.delta, abs=1e-3)
    assert res.gamma == pytest.approx(exact.gamma, abs=1e-3)


def test_dividend_yield_is_handled():
    model = BlackScholes(s0=100, r=0.03, sigma=0.3, q=0.02)
    option = EuropeanOption(95, 0.5, "put")
    assert price_pde(model, option, FINE).price == pytest.approx(model.closed_form(option).price, abs=1e-4)


def test_cell_averaging_fallback_is_accurate(model):
    option = EuropeanOption(100.03, 1.0, "call")
    res = price_pde(model, option, FINE)
    assert not res.extra["strike_pinned"]
    assert res.price == pytest.approx(model.closed_form(option).price, abs=1e-4)


def test_put_call_parity(model):
    call = price_pde(model, EuropeanOption(100, 1.0, "call"), FINE).price
    put = price_pde(model, EuropeanOption(100, 1.0, "put"), FINE).price
    assert call - put == pytest.approx(100 - 100 * np.exp(-0.05), abs=1e-4)


def _errors(model, option, n_time_ratio, **kwargs):
    exact = model.closed_form(option)
    n_values = np.array([100, 200, 400, 800])
    results = [price_pde(model, option, PDEConfig(n_space=int(n), n_time=int(n * n_time_ratio), **kwargs))
               for n in n_values]
    price_err = [abs(r.price - exact.price) for r in results]
    gamma_err = [abs(r.gamma - exact.gamma) for r in results]
    return n_values, price_err, gamma_err


def test_crank_nicolson_is_second_order_with_small_time_steps(model, option):
    n, price_err, gamma_err = _errors(model, option, n_time_ratio=1 / 2, rannacher_steps=0)
    assert -2.2 < np.polyfit(np.log(n), np.log(price_err), 1)[0] < -1.8
    assert -2.2 < np.polyfit(np.log(n), np.log(gamma_err), 1)[0] < -1.8


def test_rannacher_restores_second_order_with_large_time_steps(model, option):
    n, price_err, gamma_err = _errors(model, option, n_time_ratio=1 / 8, rannacher_steps=2)
    assert -2.2 < np.polyfit(np.log(n), np.log(price_err), 1)[0] < -1.8
    assert -2.2 < np.polyfit(np.log(n), np.log(gamma_err), 1)[0] < -1.8


def test_pure_crank_nicolson_gamma_does_not_converge_with_large_time_steps(model):
    n, _, gamma_err = _errors(model, EuropeanOption(100, 1.0, "call"), n_time_ratio=1 / 8, rannacher_steps=0)
    assert gamma_err[-1] > gamma_err[0]


def test_greek_profiles_are_returned(model):
    res = price_pde(model, EuropeanOption(100, 1.0, "call"), PDEConfig())
    assert res.extra["gamma_profile"].size == res.extra["s"].size - 2


@pytest.mark.parametrize("kwargs", [
    {"n_space": 2},
    {"n_time": 0},
    {"width": -1.0},
    {"theta": 1.5},
    {"n_time": 2, "rannacher_steps": 3},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        PDEConfig(**kwargs)
