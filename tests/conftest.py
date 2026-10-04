import pytest

from pricing import BlackScholes, EuropeanOption


@pytest.fixture
def model():
    return BlackScholes(s0=100, r=0.05, sigma=0.2)


@pytest.fixture(params=["call", "put"])
def option(request):
    return EuropeanOption(strike=100, maturity=1.0, kind=request.param)
