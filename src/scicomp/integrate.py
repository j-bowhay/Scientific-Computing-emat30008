from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def euler_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h * f(t, y)


def rk4_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h


@dataclass
class ODEResult:
    y: np.ndarray
    t: np.ndarray


def _solve_to_fixed_step(
    f: callable, y0: np.ndarray, t_span: tuple[float, float], h: float, method: callable
) -> ODEResult:
    t = [t_span[0]]
    y = [np.asarray(y0)]

    while t[-1] < t_span[-1]:
        t.append(t[-1] + h)
        y.append(method(f, t[-1], y[-1], h))

    return ODEResult(np.asarray(y), np.asarray(t))


_fixed_step_methods = {"euler": euler_step, "rk4": rk4_step}


def solve_ode(
    f: callable, y0: np.ndarray, t_span: tuple[float, float], h: float, method: str
) -> ODEResult:
    # Incase ICs aren't already an array
    y0 = np.asarray(y0)

    # wrap the function given by the user so we know it returns an array
    def f_wrapper(t, y):
        return np.asarray(f(t, y))

    if method in _fixed_step_methods:
        return _solve_to_fixed_step(f_wrapper, y0, t_span, h, _fixed_step_methods[method])
    else:
        raise ValueError(f"{method} is not a valid option for 'method'")


if __name__ == "__main__":
    ode = lambda t, y: [y[1], -y[0]]
    res = solve_ode(ode, [1, 0], [0, 1], 1e-3, "rk4")
    
    plt.plot(res.t, res.y[:, 0])
    plt.show()
