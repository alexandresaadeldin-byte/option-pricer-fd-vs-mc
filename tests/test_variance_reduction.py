import pytest

from pricing import EuropeanOption, MCConfig, price_mc

TECHNIQUES = ["antithetic", "control_variate", "importance_sampling", "stratification", "conditioning"]


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_technique_is_unbiased(model, option, technique):
    exact = model.closed_form(option)
    res = price_mc(model, option, MCConfig(n_paths=200_000, variance_reduction=technique))
    assert abs(res.price - exact.price) < 4 * res.price_stderr
    assert abs(res.delta - exact.delta) < 4 * res.delta_stderr
    assert abs(res.gamma - exact.gamma) < 4 * res.gamma_stderr


@pytest.mark.parametrize("technique", ["antithetic", "control_variate", "stratification", "conditioning"])
def test_technique_reduces_variance_at_the_money(model, option, technique):
    crude = price_mc(model, option, MCConfig())
    reduced = price_mc(model, option, MCConfig(variance_reduction=technique))
    assert reduced.price_stderr < crude.price_stderr


def test_importance_sampling_reduces_variance_deep_out_of_the_money(model):
    option = EuropeanOption(160, 1.0, "call")
    crude = price_mc(model, option, MCConfig())
    reduced = price_mc(model, option, MCConfig(variance_reduction="importance_sampling"))
    assert reduced.extra["is_drift"] > 0
    assert reduced.price_stderr < crude.price_stderr / 5


def test_importance_sampling_default_drift_is_zero_in_the_money(model):
    res = price_mc(model, EuropeanOption(80, 1.0, "call"), MCConfig(variance_reduction="importance_sampling"))
    assert res.extra["is_drift"] == 0.0


def test_antithetic_uses_the_same_number_of_payoff_evaluations(model, option):
    res = price_mc(model, option, MCConfig(n_paths=10_001, variance_reduction="antithetic"))
    assert res.extra["n_paths_used"] == 10_000


def test_control_variate_reports_beta(model, option):
    res = price_mc(model, option, MCConfig(variance_reduction="control_variate"))
    assert "beta" in res.extra
    assert res.extra["greeks_variance_reduction"] == "none"


def test_stratification_rounds_paths_to_a_multiple_of_strata(model, option):
    res = price_mc(model, option, MCConfig(n_paths=10_050, variance_reduction="stratification", n_strata=100))
    assert res.extra["n_paths_used"] == 10_000
