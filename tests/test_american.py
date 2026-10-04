import numpy as np
import pytest

from pricing import (AmericanOption, BlackScholes, EuropeanOption, MCConfig, PDEConfig,
                     price_mc, price_pde, reference_price)
from pricing.pde.lcp import brennan_schwartz, psor

# American puts, K = 40, r = 6%, sigma = 20%, T = 1 (Longstaff-Schwartz 2001, Table 1).
# Values from a Broadie-Detemple BBSR binomial tree (20 000 steps + Richardson); the published
# finite-difference column (4.478 / 2.314 / 1.110) was computed on a coarse grid.
BENCHMARKS = {36.0: 4.486675, 40.0: 2.319575, 44.0: 1.112963}


def _benchmark_case(s0):
    return BlackScholes(s0=s0, r=0.06, sigma=0.2), AmericanOption(40.0, 1.0, "put")


@pytest.mark.parametrize("s0", BENCHMARKS)
@pytest.mark.parametrize("solver", ["brennan_schwartz", "psor"])
def test_pde_matches_benchmark(s0, solver):
    model, option = _benchmark_case(s0)
    res = price_pde(model, option, PDEConfig(n_space=1600, n_time=400, lcp_solver=solver))
    assert res.price == pytest.approx(BENCHMARKS[s0], abs=2e-4)


def test_psor_and_brennan_schwartz_agree():
    model, option = _benchmark_case(36.0)
    bs = price_pde(model, option, PDEConfig(lcp_solver="brennan_schwartz"))
    ps = price_pde(model, option, PDEConfig(lcp_solver="psor"))
    np.testing.assert_allclose(bs.extra["u"], ps.extra["u"], atol=1e-6)
    assert ps.extra["psor_sweeps_mean"] > 1


def _check_lcp(sub, main, sup, rhs, g, x):
    a_x = main * x + np.r_[0.0, sub * x[:-1]] + np.r_[sup * x[1:], 0.0]
    assert np.all(x >= g - 1e-12)
    assert np.all(a_x >= rhs - 1e-9)
    assert np.max(np.abs((a_x - rhs) * (x - g))) < 1e-9


def test_psor_solves_a_generic_lcp():
    n = 50
    sub, main, sup = np.full(n - 1, -1.0), np.full(n, 2.5), np.full(n - 1, -1.0)
    rhs = np.linspace(-1.0, 1.0, n)
    g = np.sin(np.linspace(0, 6, n))
    x, _ = psor(sub, main, sup, rhs, g, np.zeros(n), 1.5, 1e-13, 100_000)
    _check_lcp(sub, main, sup, rhs, g, x)


@pytest.mark.parametrize("exercise_at_high", [False, True])
def test_brennan_schwartz_matches_psor_when_exercise_region_touches_one_end(exercise_at_high):
    n = 50
    sub, main, sup = np.full(n - 1, -1.0), np.full(n, 2.5), np.full(n - 1, -1.0)
    rhs = np.full(n, 0.05)
    g = np.maximum(1.0 - np.linspace(0, 2, n), 0.0)
    if exercise_at_high:
        g = g[::-1].copy()
    x_bs = brennan_schwartz(sub, main, sup, rhs, g, exercise_at_high)
    x_ps, _ = psor(sub, main, sup, rhs, g, np.zeros(n), 1.5, 1e-13, 100_000)
    _check_lcp(sub, main, sup, rhs, g, x_bs)
    np.testing.assert_allclose(x_bs, x_ps, atol=1e-9)


def test_american_put_dominates_european_and_intrinsic():
    model, option = _benchmark_case(40.0)
    am = price_pde(model, option, PDEConfig())
    eu = price_pde(model, EuropeanOption(40.0, 1.0, "put"), PDEConfig())
    assert np.all(am.extra["u"] >= eu.extra["u"] - 1e-10)
    assert np.all(am.extra["u"] >= option.payoff(am.extra["s"]) - 1e-10)
    assert am.price > eu.price


def test_american_call_without_dividends_equals_european():
    model = BlackScholes(s0=100, r=0.05, sigma=0.2)
    am = price_pde(model, AmericanOption(100, 1.0, "call"), PDEConfig(n_space=800, n_time=200))
    assert am.price == pytest.approx(model.closed_form(EuropeanOption(100, 1.0, "call")).price, abs=1e-3)


def test_american_call_with_dividends_is_worth_more():
    model = BlackScholes(s0=100, r=0.05, sigma=0.2, q=0.08)
    am = price_pde(model, AmericanOption(100, 1.0, "call"), PDEConfig(n_space=800, n_time=200))
    assert am.price > model.closed_form(EuropeanOption(100, 1.0, "call")).price + 0.1


def test_put_exercise_boundary_is_below_strike_and_monotone():
    model, option = _benchmark_case(36.0)
    res = price_pde(model, option, PDEConfig(n_space=800, n_time=200))
    t, s_star = res.extra["exercise_boundary_t"], res.extra["exercise_boundary_s"]
    order = np.argsort(t)
    s_sorted = s_star[order]
    assert np.all(s_sorted < 40.0)
    assert np.all(np.diff(s_sorted) >= -1e-9)


def test_reference_price_is_accurate():
    model, option = _benchmark_case(40.0)
    assert reference_price(model, option).price == pytest.approx(BENCHMARKS[40.0], abs=3e-5)


@pytest.mark.parametrize("s0", BENCHMARKS)
def test_lsm_is_close_to_benchmark(s0):
    model, option = _benchmark_case(s0)
    res = price_mc(model, option, MCConfig(n_paths=100_000, n_steps=50))
    assert abs(res.price - BENCHMARKS[s0]) < 4 * res.price_stderr + 0.02
    assert res.price < BENCHMARKS[s0] + 4 * res.price_stderr


def test_lsm_bermudan_bias_shrinks_with_more_exercise_dates():
    model, option = _benchmark_case(36.0)
    few = price_mc(model, option, MCConfig(n_paths=100_000, n_steps=5))
    many = price_mc(model, option, MCConfig(n_paths=100_000, n_steps=50))
    assert few.price < many.price


def test_lsm_pathwise_delta_is_close_to_pde_delta():
    model, option = _benchmark_case(36.0)
    mc = price_mc(model, option, MCConfig(n_paths=100_000, n_steps=50))
    pde = reference_price(model, option)
    assert mc.delta == pytest.approx(pde.delta, abs=0.01)
    assert np.isnan(mc.gamma)


def test_lsm_is_reproducible():
    model, option = _benchmark_case(40.0)
    a = price_mc(model, option, MCConfig(n_paths=5_000, seed=3))
    b = price_mc(model, option, MCConfig(n_paths=5_000, seed=3))
    assert a.price == b.price


def test_lsm_rejects_variance_reduction():
    model, option = _benchmark_case(40.0)
    with pytest.raises(ValueError):
        price_mc(model, option, MCConfig(variance_reduction="antithetic"))


def test_simulated_paths_have_the_right_terminal_law():
    model = BlackScholes(s0=100, r=0.05, sigma=0.2, q=0.01)
    z = np.random.default_rng(0).standard_normal((200_000, 10))
    paths = model.simulate_paths(z, 1.0)
    assert paths.shape == (200_000, 11)
    assert np.all(paths[:, 0] == 100)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.04), rel=2e-3)
    assert np.log(paths[:, -1] / 100).std() == pytest.approx(0.2, rel=1e-2)


def test_european_pricers_reject_unknown_products():
    model = BlackScholes(s0=100, r=0.05, sigma=0.2)
    with pytest.raises(TypeError):
        price_pde(model, object(), PDEConfig())
    with pytest.raises(TypeError):
        price_mc(model, object(), MCConfig())
