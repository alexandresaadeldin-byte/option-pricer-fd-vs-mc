"""Solvers for the tridiagonal linear complementarity problem

    A x >= b,   x >= g,   (A x - b) . (x - g) = 0,

where A has sub-diagonal `sub`, diagonal `main` and super-diagonal `sup`.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _projected_thomas(sub, main, sup, rhs, obstacle):
    n = main.size
    c = main.copy()
    d = rhs.copy()
    for i in range(1, n):
        w = sub[i - 1] / c[i - 1]
        c[i] -= w * sup[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(n)
    x[n - 1] = max(d[n - 1] / c[n - 1], obstacle[n - 1])
    for i in range(n - 2, -1, -1):
        x[i] = max((d[i] - sup[i] * x[i + 1]) / c[i], obstacle[i])
    return x


def brennan_schwartz(sub, main, sup, rhs, obstacle, exercise_at_high_index):
    """Direct O(n) solver. The projected substitution must start inside the exercise region."""
    if exercise_at_high_index:
        return _projected_thomas(sub, main, sup, rhs, obstacle)
    reversed_x = _projected_thomas(sup[::-1].copy(), main[::-1].copy(), sub[::-1].copy(),
                                   rhs[::-1].copy(), obstacle[::-1].copy())
    return reversed_x[::-1].copy()


@njit(cache=True)
def psor(sub, main, sup, rhs, obstacle, x0, omega, tol, max_iter):
    """Projected successive over-relaxation. Returns (solution, number of sweeps)."""
    n = main.size
    x = np.maximum(x0, obstacle)
    for sweep in range(1, max_iter + 1):
        max_change = 0.0
        for i in range(n):
            s = rhs[i]
            if i > 0:
                s -= sub[i - 1] * x[i - 1]
            if i < n - 1:
                s -= sup[i] * x[i + 1]
            new = max(obstacle[i], x[i] + omega * (s / main[i] - x[i]))
            change = abs(new - x[i])
            if change > max_change:
                max_change = change
            x[i] = new
        if max_change < tol:
            return x, sweep
    return x, max_iter
