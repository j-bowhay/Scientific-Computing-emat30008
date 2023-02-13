from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import matplotlib.pyplot as plt


def _euler_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h * f(t, y)


def _rk4_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
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

    return ODEResult(np.asarray(y).T, np.asarray(t))


_fixed_step_methods = {"euler": _euler_step, "rk4": _rk4_step}

_embedded_methods = {}


def solve_ivp(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    method: str,
    h: float = None,
    r_tol: float = 0,
    a_tol: float = 0,
) -> ODEResult:
    if not callable(f):
        raise ValueError("'f' must be callable")
    else:
        f_sig = inspect.signature(f)

        if list(f_sig.parameters) != ["t", "y"]:
            raise ValueError("'f' has an invalid signature")

    if len(t_span) != 2 or t_span[0] > t_span[1]:
        raise ValueError("Invalid values for 't_span'")

    if h is None and r_tol == 0 and a_tol == 0:
        # If running in fixed step mode the user must provide the step size
        raise ValueError(
            "Step size must be provided if running in fixed step mode"
        )  # TODO: write a test for this

    # Incase ICs aren't already an array
    y0 = np.asarray(y0)

    # wrap the function given by the user so we know it returns an array
    def f_wrapper(t, y):
        return np.asarray(f(t, y))

    if h is None and (r_tol != 0 or a_tol != 0):
        # compute initial step size
        ...

    if method in _fixed_step_methods:
        if r_tol == 0 and a_tol == 0:
            # run in fixed mode
            return _solve_to_fixed_step(
                f_wrapper, y0, t_span, h, _fixed_step_methods[method]
            )
        else:
            # run in adaptive step mode
            ...
    elif method in _embedded_methods:
        ...
    else:
        raise ValueError(f"{method} is not a valid option for 'method'")


if __name__ == "__main__":
    ode = lambda t, y: [y[1], -y[0]]
    res = solve_ivp(ode, [1, 0], [0, 1], h=1e-3, method="rk4")

    plt.plot(res.t, res.y[0, :])
    plt.show()
