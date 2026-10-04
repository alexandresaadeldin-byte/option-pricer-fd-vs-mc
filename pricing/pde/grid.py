from dataclasses import dataclass
from math import ceil, log, sqrt

import numpy as np


@dataclass(frozen=True)
class LogGrid:
    x: np.ndarray
    dx: float
    center: int
    strike_pinned: bool

    @property
    def s(self) -> np.ndarray:
        return np.exp(self.x)


def build_grid(s0, strike, sigma, maturity, n_space, width) -> LogGrid:
    half_width = width * sigma * sqrt(maturity)
    x0 = log(s0)
    dx_target = 2 * half_width / n_space
    d = abs(log(strike) - x0)

    if d < 1e-12:
        dx, pinned = dx_target, True
    elif d >= dx_target / 2:
        dx, pinned = d / ceil(d / dx_target), True
    else:
        dx, pinned = dx_target, False

    n_half = ceil(half_width / dx - 1e-9)
    x = x0 + dx * np.arange(-n_half, n_half + 1)
    return LogGrid(x=x, dx=dx, center=n_half, strike_pinned=pinned)
