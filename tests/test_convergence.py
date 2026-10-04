import numpy as np
import pytest

from pricing import EuropeanOption, MCConfig, PDEConfig
from pricing.convergence import fit_slope, mc_convergence, pde_convergence


def test_fit_slope_recovers_a_power_law():
    x = np.array([10.0, 100.0, 1000.0])
    assert fit_slope(x, 3 * x**-1.5) == pytest.approx(-1.5)


def test_mc_convergence_table(model):
    option = EuropeanOption(100, 1.0, "call")
    df = mc_convergence(model, option, MCConfig(), [1_000, 10_000, 100_000], repeats=1)
    assert list(df["n_paths"]) == [1_000, 10_000, 100_000]
    assert {"price", "error", "stderr", "ci_half_width", "delta_error", "gamma_error", "elapsed"} <= set(df.columns)
    assert fit_slope(df["n_paths"], df["ci_half_width"]) == pytest.approx(-0.5, abs=0.1)


def test_pde_convergence_keeps_enough_time_steps_for_rannacher(model):
    option = EuropeanOption(100, 1.0, "call")
    df = pde_convergence(model, option, PDEConfig(n_space=400, n_time=50, rannacher_steps=2), [8, 16], repeats=1)
    assert list(df["n_time"]) == [2, 2]


def test_pde_convergence_table(model):
    option = EuropeanOption(110, 1.0, "call")
    df = pde_convergence(model, option, PDEConfig(n_space=100, n_time=50), [100, 200, 400, 800], repeats=1)
    assert list(df["n_time"]) == [50, 100, 200, 400]
    assert fit_slope(df["n_nodes"], df["error"]) == pytest.approx(-2.0, abs=0.2)


def test_american_studies_use_the_reference_price():
    from pricing import AmericanOption, BlackScholes
    from pricing.convergence import exercise_dates_study
    model, option = BlackScholes(s0=36, r=0.06, sigma=0.2), AmericanOption(40, 1.0, "put")
    pde = pde_convergence(model, option, PDEConfig(n_space=200, n_time=50), [200, 400, 800], repeats=1)
    assert pde["error"].iloc[-1] < pde["error"].iloc[0]
    assert fit_slope(pde["n_nodes"], pde["error"]) < -1.0
    steps = exercise_dates_study(model, option, MCConfig(n_paths=50_000), [4, 16, 64])
    assert steps["bias"].iloc[0] < 0
    assert steps["price"].iloc[0] < steps["price"].iloc[-1]
