from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

# =====================================================================================
# Step Routines
# =====================================================================================

# Standard Runge Kutta Type Steps


def _butcher_tableau_step(
    f: callable,
    t: float,
    y: np.ndarray,
    h: float,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    s = B.size

    ks = np.empty((y.size, s))
    for i in range(s):
        ks[:, i] = f(t + C[i] * h, y + h * np.sum(A[i, : i + 1] * ks[: i + 1], axis=-1))

    return y + h * np.sum(B * ks, axis=-1)

def _euler_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Performs one step of the forward euler method

    Parameters
    ----------
    f : callable
        The function to integrate
    t : float
        Current time
    y : np.ndarray
        Current state
    h : float
        Step size

    Returns
    -------
    np.ndarray
        Solution after one step
    """
    A = np.array([[0]])
    B = np.array([1])
    C = np.array([0])
    return _butcher_tableau_step(f, t, y, h, A, B, C)


def _rk4_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Performs one step of the fourth order Runge Kutta method.
    Parameters
    ----------
    f : callable
        The function to integrate
    t : float
        Current time
    y : np.ndarray
        Current state
    h : float
        Step size

    Returns
    -------
    np.ndarray
        Solution after one step
    """
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h

# Embedded Error Estimate Steps


# =====================================================================================
# Stepper Routines
# =====================================================================================

@dataclass
class ODEResult:
    y: np.ndarray
    t: np.ndarray


def _solve_to_fixed_step(
    f: callable, y0: np.ndarray, t_span: tuple[float, float], h: float, method: callable
) -> ODEResult:
    t = [t_span[0]]
    y = [np.asarray(y0)]

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        # check if the step is going to overshoot and adjust accordingly
        if (t[-1] + h - t_span[-1]) > 0:
            h = t_span[-1] - t[-1]
        t.append(t[-1] + h)
        y.append(method(f, t[-1], y[-1], h))

    return ODEResult(np.asarray(y).T, np.asarray(t))

# =====================================================================================
# Driver
# =====================================================================================

_fixed_step_methods = {"euler": _euler_step, "rk4": _rk4_step}

_embedded_methods = {}


def solve_ivp(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
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

    if y0.ndim != 1:
        raise ValueError("Initial conditions must be 1 dimensional.")

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
