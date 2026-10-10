from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class _Vanilla:
    strike: float
    maturity: float
    kind: Literal["call", "put"] = "call"

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    def payoff(self, s: ArrayLike) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.is_call:
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)

    def payoff_derivative(self, s: ArrayLike) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.is_call:
            return (s > self.strike).astype(float)
        return -(s < self.strike).astype(float)

    def log_cell_average(self, x_left: ArrayLike, x_right: ArrayLike) -> np.ndarray:
        """Exact average of the payoff, in x = ln S, over each cell [x_left, x_right]."""
        x_left = np.asarray(x_left, dtype=float)
        x_right = np.asarray(x_right, dtype=float)
        k = np.log(self.strike)
        if self.is_call:
            lo = np.clip(k, x_left, x_right)
            integral = np.exp(x_right) - np.exp(lo) - self.strike * (x_right - lo)
        else:
            hi = np.clip(k, x_left, x_right)
            integral = self.strike * (hi - x_left) - (np.exp(hi) - np.exp(x_left))
        return integral / (x_right - x_left)


@dataclass(frozen=True)
class EuropeanOption(_Vanilla):
    """Vanilla call or put, exercisable at maturity only."""


@dataclass(frozen=True)
class AmericanOption(_Vanilla):
    """Vanilla call or put, exercisable at any time up to maturity."""


VanillaOption = EuropeanOption | AmericanOption
