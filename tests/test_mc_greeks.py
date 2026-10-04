import pytest

from pricing import MCConfig, price_mc


@pytest.mark.parametrize("delta_method", ["pathwise", "likelihood_ratio"])
def test_delta_estimators_are_unbiased(model, option, delta_method):
    exact = model.closed_form(option)
    res = price_mc(model, option, MCConfig(n_paths=200_000, delta_method=delta_method))
    assert abs(res.delta - exact.delta) < 4 * res.delta_stderr


@pytest.mark.parametrize("gamma_method", ["likelihood_ratio", "pathwise_lr"])
def test_gamma_estimators_are_unbiased(model, option, gamma_method):
    exact = model.closed_form(option)
    res = price_mc(model, option, MCConfig(n_paths=200_000, gamma_method=gamma_method))
    assert abs(res.gamma - exact.gamma) < 4 * res.gamma_stderr


def test_pathwise_delta_has_lower_variance_than_likelihood_ratio(model, option):
    pw = price_mc(model, option, MCConfig(delta_method="pathwise"))
    lr = price_mc(model, option, MCConfig(delta_method="likelihood_ratio"))
    assert pw.delta_stderr < lr.delta_stderr


def test_pathwise_gamma_is_rejected(model, option):
    with pytest.raises(ValueError, match="pathwise gamma"):
        price_mc(model, option, MCConfig(gamma_method="pathwise"))
