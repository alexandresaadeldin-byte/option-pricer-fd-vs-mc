import numpy as np
import pytest

from pricing import EuropeanOption, MCConfig, price_mc


def test_crude_mc_is_within_four_standard_errors(model, option):
    exact = model.closed_form(option)
    res = price_mc(model, option, MCConfig(n_paths=200_000))
    assert abs(res.price - exact.price) < 4 * res.price_stderr


def test_same_seed_gives_same_price(model, option):
    a = price_mc(model, option, MCConfig(n_paths=10_000, seed=7))
    b = price_mc(model, option, MCConfig(n_paths=10_000, seed=7))
    c = price_mc(model, option, MCConfig(n_paths=10_000, seed=8))
    assert a.price == b.price
    assert a.price != c.price


def test_stderr_decays_like_one_over_sqrt_n(model):
    option = EuropeanOption(100, 1.0, "call")
    n_values = np.array([1_000, 4_000, 16_000, 64_000, 256_000])
    stderrs = [price_mc(model, option, MCConfig(n_paths=int(n))).price_stderr for n in n_values]
    slope = np.polyfit(np.log(n_values), np.log(stderrs), 1)[0]
    assert -0.6 < slope < -0.4


def test_put_call_parity_holds_within_mc_error(model):
    call = price_mc(model, EuropeanOption(100, 1.0, "call"), MCConfig(n_paths=200_000, seed=1))
    put = price_mc(model, EuropeanOption(100, 1.0, "put"), MCConfig(n_paths=200_000, seed=2))
    parity = 100 - 100 * np.exp(-0.05)
    assert abs(call.price - put.price - parity) < 4 * np.hypot(call.price_stderr, put.price_stderr)


@pytest.mark.parametrize("kwargs", [
    {"n_paths": 1},
    {"variance_reduction": "magic"},
    {"conditioning_time": 1.0},
    {"n_strata": 0},
    {"variance_reduction": "stratification", "n_paths": 100, "n_strata": 100},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        MCConfig(**kwargs)
