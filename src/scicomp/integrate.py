from __future__ import annotations

import inspect
import math
from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

# =====================================================================================
# Step Routines
# =====================================================================================

# Standard Runge Kutta Type Steps


@dataclass
class _StepResult:
    y: np.ndarray
    error_estimate: Optional[np.ndarray] = None


class _RungeKuttaStep(ABC):
    @abstractproperty
    def A(self) -> np.ndarray:
        ...

    @abstractproperty
    def B(self) -> np.ndarray:
        ...

    @abstractproperty
    def C(self) -> np.ndarray:
        ...

    @abstractproperty
    def order(self) -> int:
        ...

    def __init__(self) -> None:
        self.s = self.B.size

    def __call__(self, f: Callable, t: float, y: np.ndarray, h: float) -> _StepResult:
        ks = np.empty((y.size, self.s))
        for i in range(self.s):
            ks[:, i] = f(
                t + self.C[i] * h,
                y
                + h * np.sum(self.A[i, np.newaxis, : i + 1] * ks[:, : i + 1], axis=-1),
            )

        y1 = y + h * np.inner(self.B, ks)

        # return the error estimate if there is an embedded formula
        if hasattr(self, "B_hat"):
            return _StepResult(y1, h * np.inner(self.B - self.B_hat, ks))
        return _StepResult(y1)


class _EulerStep(_RungeKuttaStep):
    A = np.array([[0]])
    B = np.array([1])
    C = np.array([0])
    order = 1


class _ExplicitMidpointStep(_RungeKuttaStep):
    A = np.array([[0, 0], [0.5, 0]])
    B = np.array([0, 1])
    C = np.array([0, 0.5])
    order = 2


class _HeunsStep(_RungeKuttaStep):
    A = np.array([[0, 0], [1, 0]])
    B = np.array([0.5, 0.5])
    C = np.array([0, 1])
    order = 2


class _RalstonStep(_RungeKuttaStep):
    A = np.array([[0, 0], [2 / 3, 0]])
    B = np.array([1 / 4, 3 / 4])
    C = np.array([0, 2 / 3])
    order = 2


class _Kutta3Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
    B = np.array([1 / 6, 2 / 3, 1 / 6])
    C = np.array([0, 1 / 2, 1])
    order = 3


class _Heun3Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]])
    B = np.array([1 / 4, 0, 3 / 4])
    C = np.array([0, 1 / 3, 2 / 3])
    order = 3


class _Wray3Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0], [8 / 15, 0, 0], [1 / 4, 5 / 12, 0]])
    B = np.array([1 / 4, 0, 3 / 4])
    C = np.array([0, 8 / 15, 2 / 3])
    order = 3


class _Ralston3Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    C = np.array([0, 1 / 2, 3 / 4])
    order = 3


class _SSPRK3Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
    B = np.array([1 / 6, 1 / 6, 2 / 3])
    C = np.array([0, 1, 1 / 2])
    order = 3


class _RK4Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
    B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    C = np.array([0, 0.5, 0.5, 1])
    order = 4


class _RK38Step(_RungeKuttaStep):
    A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])

    B = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    C = np.array([0, 1 / 3, 2 / 3, 1])
    order = 4


class _Ralston4Step(_RungeKuttaStep):
    A = np.array(
        [
            [0, 0, 0, 0],
            [0.4, 0, 0, 0],
            [0.29697761, 0.15875964, 0, 0],
            [0.21810040, -3.050965161, 3.83286476, 0],
        ]
    )
    B = np.array([0.17476028, -0.55148066, 1.20553560, 0.17118478])
    C = np.array([0, 0.4, 0.45573725, 1])
    order = 4


# Embedded Error Estimate Steps


class _BogackiShampineStep(_RungeKuttaStep):
    A = np.array(
        [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
    )
    B = np.array([2 / 9, 1 / 3, 4 / 9, 0])
    B_hat = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
    C = np.array([0, 1 / 2, 3 / 4, 1])
    order = 3


class _RKF45Step(_RungeKuttaStep):
    A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 4, 0, 0, 0, 0, 0],
            [3 / 32, 9 / 32, 0, 0, 0, 0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
            [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        ]
    )
    B = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    B_hat = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
    C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    order = 5


# =====================================================================================
# Stepper Routines
# =====================================================================================


@dataclass(frozen=True, slots=True)
class ODEResult:
    y: np.ndarray
    t: np.ndarray


def _solve_to_fixed_step(
    f: Callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    h: float,
    method: _RungeKuttaStep,
) -> ODEResult:
    """Solves an ivp by taking fixed steps.

    Parameters
    ----------
    f : callable
        The rhs function for the ODE.
    y0 : np.ndarray
        The initial conditions.
    t_span : tuple[float, float]
        The time range to solve over.
    h : float
        The fixed step size to use.
    method : callable
        Function that returns the next step of the ODE.

    Returns
    -------
    ODEResult
        Results object containing the solution of the IVP.
    """
    t = [t_span[0]]
    y = [np.asarray(y0)]

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        # check if the step is going to overshoot and adjust accordingly
        if (t[-1] + h - t_span[-1]) > 0:
            h = t_span[-1] - t[-1]
        t.append(t[-1] + h)
        y.append(method(f, t[-1], y[-1], h).y)

    return ODEResult(np.asarray(y).T, np.asarray(t))


def _richardson_error_estimate(
    f: Callable,
    t: list,
    y: list,
    h: float,
    method: _RungeKuttaStep,
):
    # take two small steps to find y1
    y1 = y
    for i in range(2):
        y1 = method(f, t + i * h, y1, h / 2).y
    # take one large step two find w
    w = method(f, t, y, h).y

    # eq 4.4 page 165 Hairer Solving ODEs 1
    local_err = (y1 - w) / (2**method.order - 1)

    return y1, local_err


def _embedded_error_estimate(
    f: Callable,
    t: list,
    y: list,
    h: float,
    method: _RungeKuttaStep,
):
    step = method(f, t, y, h)
    y1 = step.y
    local_err = step.error_estimate

    return y1, local_err


def _solve_to_adaptive(
    f: Callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    h: float,
    method: _RungeKuttaStep,
    r_tol: float,
    a_tol: float,
    max_step: float,
    error_estimate: Callable,
) -> ODEResult:
    """_summary_

    Parameters
    ----------
    f : callable
        The rhs function for the ODE.
    y0 : np.ndarray
        The initial conditions.
    t_span : tuple[float, float]
        The time range to solve over.
    h : float
        The initial step size to use.
    method : callable
        Function that returns the next step of the ODE.
    r_tol : float
        The relative tolerance (the correct number of digits).
    a_tol : float
        The absolute tolerance (the correct number of decimal places).
    max_step : float
        The maximum acceptable step size to take.
    error_estimate : callable
        The function which generates the next point, error estimate and the scale

    Returns
    -------
    ODEResult
        Results object containing the solution of the IVP.
    """
    t = [t_span[0]]
    y = [np.asarray(y0)]

    final_step = False

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        step_accepted = False
        while not step_accepted:
            y1, local_err = error_estimate(f, t[-1], y[-1], h, method)

            scale = a_tol + np.maximum(np.abs(y1), np.abs(y[-1])) * r_tol

            # eq 4.11 page 168 Hairer
            err = np.sqrt(np.sum((local_err / scale) ** 2) / local_err.size)

            # adjust step size
            fac_max = 1.5
            fac_min = 0.5
            safety_fac = 0.9
            h_new = h * np.minimum(
                fac_max,
                np.maximum(fac_min, safety_fac * (1 / err) ** (1 / (method.order + 1))),
            )
            h_new = max_step if h_new > max_step else h_new

            # accept the step
            if (err <= 1 and h <= max_step) or final_step:
                if (t[-1] + h - t_span[-1]) > 0 and not final_step:
                    final_step = True
                    h_new = t_span[-1] - t[-1]
                else:
                    step_accepted = True
                    t.append(t[-1] + h)
                    y.append(y1)
            h = h_new

    return ODEResult(np.asarray(y).T, np.asarray(t))


# =====================================================================================
# Driver
# =====================================================================================

_fixed_step_methods = {
    "euler": _EulerStep,
    "midpoint": _ExplicitMidpointStep,
    "heun": _HeunsStep,
    "ralston": _RalstonStep,
    "kutta2": _Kutta3Step,
    "heun3": _Heun3Step,
    "wray3": _Wray3Step,
    "ralston3": _Ralston3Step,
    "ssprk3": _SSPRK3Step,
    "rk4": _RK4Step,
    "rk38": _RK38Step,
    "ralston4": _Ralston4Step,
}

_embedded_methods = {
    "bogacki_shampine": _BogackiShampineStep,
    "rkf45": _RKF45Step,
}

_all_methods = {**_fixed_step_methods, **_embedded_methods}


def _estimate_initial_step_size(f, y0, t0, method, r_tol, a_tol, max_step):
    scale = a_tol + np.abs(y0) * r_tol
    d0 = np.sqrt(np.sum((y0 / scale) ** 2) / y0.size)
    # not sure if this should be a different scale?
    f0 = f(t0, y0)
    scale = a_tol + np.abs(f0) * r_tol
    d1 = np.sqrt(np.sum((f0 / scale) ** 2) / y0.size)

    if d0 < 1e-5 or d1 < 1e-5 or math.isnan(d0) or math.isnan(d1):
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    y1 = _EulerStep()(f, t0, y0, h0).y
    diff = f(t0 + h0, y1) - f(t0, y0)
    scale = a_tol + np.abs(diff) * r_tol
    d2 = np.sqrt(np.sum((diff / scale) ** 2) / y0.size) / h0

    if np.maximum(d1, d2) <= 1e-15 or math.isnan(d1) or math.isnan(d2):
        h1 = np.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / np.maximum(d1, d2)) ** (1 / (method.order + 1))
    h = np.maximum(100 * h0, h1)
    return h if h < max_step else max_step


def solve_ivp(
    f: Callable,
    y0: npt.ArrayLike,
    t_span: tuple[float, float],
    *,
    method: str,
    h: Optional[float] = None,
    r_tol: float = 0.0,
    a_tol: float = 0.0,
    max_step: float = np.inf,
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
        raise ValueError("Step size must be provided if running in fixed step mode")

    if max_step < 0 or (h is not None and h < 0) or r_tol < 0 or a_tol < 0:
        raise ValueError("Invalid negative option.")

    # Incase ICs aren't already an array
    y0 = np.asarray(y0)

    if y0.ndim != 1:
        raise ValueError("Initial conditions must be 1 dimensional.")

    # wrap the function given by the user so we know it returns an array
    def f_wrapper(t, y):
        return np.asarray(f(t, y))

    if method in _all_methods.keys():
        method_step = _all_methods[method]()  # type: ignore
    else:
        raise ValueError(f"{method} is not a valid option for 'method'")

    if h is None:
        # compute initial step size
        h = _estimate_initial_step_size(
            f_wrapper, y0, t_span[0], method_step, r_tol, a_tol, max_step
        )

    if method in _fixed_step_methods:
        if r_tol == 0 and a_tol == 0:
            # run in fixed mode
            return _solve_to_fixed_step(f_wrapper, y0, t_span, h, method_step)
        else:
            return _solve_to_adaptive(
                f_wrapper,
                y0,
                t_span,
                h,
                method_step,
                r_tol,
                a_tol,
                max_step,
                _richardson_error_estimate,
            )
    elif method in _embedded_methods:
        return _solve_to_adaptive(
            f_wrapper,
            y0,
            t_span,
            h,
            method_step,
            r_tol,
            a_tol,
            max_step,
            _embedded_error_estimate,
        )
