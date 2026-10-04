import numpy as np
import pytest

from pricing import EuropeanOption


def test_call_and_put_payoffs():
    s = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(EuropeanOption(100, 1, "call").payoff(s), [0, 0, 20])
    np.testing.assert_allclose(EuropeanOption(100, 1, "put").payoff(s), [20, 0, 0])


def test_payoff_derivative_is_an_indicator():
    s = np.array([80.0, 120.0])
    np.testing.assert_allclose(EuropeanOption(100, 1, "call").payoff_derivative(s), [0, 1])
    np.testing.assert_allclose(EuropeanOption(100, 1, "put").payoff_derivative(s), [-1, 0])


@pytest.mark.parametrize("kind", ["call", "put"])
def test_log_cell_average_matches_numerical_integral(kind):
    option = EuropeanOption(100, 1, kind)
    x_left, x_right = np.log(98.0), np.log(103.0)
    fine = np.linspace(x_left, x_right, 200_001)
    expected = option.payoff(np.exp(fine)).mean()
    assert option.log_cell_average(x_left, x_right) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("kwargs", [
    {"strike": -1, "maturity": 1},
    {"strike": 100, "maturity": 0},
    {"strike": 100, "maturity": 1, "kind": "straddle"},
])
def test_invalid_option_raises(kwargs):
    with pytest.raises(ValueError):
        EuropeanOption(**kwargs)
