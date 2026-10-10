import numpy as np


def grid_greeks(s: np.ndarray, u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Delta and gamma at the interior nodes s[1:-1], from x = ln S derivatives."""
    u_x = (u[2:] - u[:-2]) / (2 * dx)
    u_xx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    s_in = s[1:-1]
    return u_x / s_in, (u_xx - u_x) / s_in**2
