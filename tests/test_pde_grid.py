import numpy as np
import pytest

from pricing.pde.grid import build_grid


def test_spot_is_the_central_node():
    grid = build_grid(s0=100, strike=110, sigma=0.2, maturity=1.0, n_space=200, width=5.0)
    assert grid.x[grid.center] == pytest.approx(np.log(100), abs=1e-14)


@pytest.mark.parametrize("strike", [100.0, 110.0, 80.0, 101.0])
def test_strike_is_pinned_on_a_node(strike):
    grid = build_grid(s0=100, strike=strike, sigma=0.2, maturity=1.0, n_space=200, width=5.0)
    assert grid.strike_pinned
    assert np.min(np.abs(grid.x - np.log(strike))) < 1e-12


def test_pinned_step_stays_between_half_and_full_target():
    dx_target = 2 * 5.0 * 0.2 / 200
    grid = build_grid(s0=100, strike=110, sigma=0.2, maturity=1.0, n_space=200, width=5.0)
    assert dx_target / 2 <= grid.dx <= dx_target


def test_strike_too_close_to_spot_falls_back_to_cell_averaging():
    grid = build_grid(s0=100, strike=100.1, sigma=0.2, maturity=1.0, n_space=200, width=5.0)
    assert not grid.strike_pinned


def test_domain_covers_the_requested_width():
    grid = build_grid(s0=100, strike=110, sigma=0.2, maturity=1.0, n_space=200, width=5.0)
    assert grid.x[0] <= np.log(100) - 1.0 and grid.x[-1] >= np.log(100) + 1.0
    np.testing.assert_allclose(np.diff(grid.x), grid.dx)
