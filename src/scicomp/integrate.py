from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

# =====================================================================================
# Step Routines
# =====================================================================================

# Standard Runge Kutta Type Steps


class _RungeKuttaStep:
    __slots__ = "A", "B", "C", "s"

    def __init__(self) -> None:
        self.s = self.B.size

    def __call__(self, f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
        ks = np.empty((y.size, self.s))
        for i in range(self.s):
            ks[:, i] = f(
                t + self.C[i] * h,
                y
                + h * np.sum(self.A[i, np.newaxis, : i + 1] * ks[:, : i + 1], axis=-1),
            )

        y1 = y + h * np.sum(self.B * ks, axis=-1)

        # return the error estimate if there is an embedded formula
        if hasattr(self, "B_hat"):
            return y1, np.sum((self.B - self.B_hat) * ks, axis=-1)
        return y1


class _EulerStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0]])
        self.B = np.array([1])
        self.C = np.array([0])
        self.order = 1
        super().__init__()


class _ExplicitMidpointStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0], [0.5, 0]])
        self.B = np.array([0, 1])
        self.C = np.array([0, 0.5])
        self.order = 2
        super().__init__()


class _HeunsStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0], [1, 0]])
        self.B = np.array([0.5, 0.5])
        self.C = np.array([0, 1])
        self.order = 2
        super().__init__()


class _RalstonStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0], [2 / 3, 0]])
        self.B = np.array([1 / 4, 3 / 4])
        self.C = np.array([0, 2 / 3])
        self.order = 2
        super().__init__()


class _Kutta3Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
        self.B = np.array([1 / 6, 2 / 3, 1 / 6])
        self.C = np.array([0, 1 / 2, 1])
        self.order = 3
        super().__init__()


class _Heun3Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]])
        self.B = np.array([1 / 4, 0, 3 / 4])
        self.C = np.array([0, 1 / 3, 2 / 3])
        self.order = 3
        super().__init__()


class _Wray3Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [8 / 15, 0, 0], [1 / 4, 5 / 12, 0]])
        self.B = np.array([1 / 4, 0, 3 / 4])
        self.C = np.array([0, 8 / 15, 2 / 3])
        self.order = 3
        super().__init__()


class _Ralston3Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
        self.B = np.array([2 / 9, 1 / 3, 4 / 9])
        self.C = np.array([0, 1 / 2, 3 / 4])
        self.order = 3
        super().__init__()


class _SSPRK3Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
        self.B = np.array([1 / 6, 1 / 6, 2 / 3])
        self.C = np.array([0, 1, 1 / 2])
        self.order = 3
        super().__init__()


class _RK4Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        self.B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        self.C = np.array([0, 0.5, 0.5, 1])
        self.order = 4
        super().__init__()


class _RK38Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array(
            [[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]]
        )
        self.B = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        self.C = np.array([0, 1 / 3, 2 / 3, 1])
        self.order = 4
        super().__init__()


class _Ralston4Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array(
            [
                [0, 0, 0, 0],
                [0.4, 0, 0, 0],
                [0.29697761, 0.15875964, 0, 0],
                [0.21810040, -3.050965161, 3.83286476, 0],
            ]
        )
        self.B = np.array([0.17476028, -0.55148066, 1.20553560, 0.17118478])
        self.C = np.array([0, 0.4, 0.45573725, 1])
        self.order = 4
        super().__init__()


# Embedded Error Estimate Steps


class _HeunEulerStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0], [1, 0]])
        self.B = np.array([1 / 2, 1 / 2])
        self.B_hat = np.array([1, 0])
        self.C = np.array([0, 1])
        self.order = 2
        super().__init__()


class _RKF12Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array([[0, 0, 0], [1 / 2, 0, 0], [1 / 256, 255 / 256, 0]])
        self.B = np.array([1 / 512, 255 / 256, 1 / 512])
        self.B_hat = np.array([1 / 255, 255 / 256, 0])
        self.C = np.array([0, 1 / 2, 1])
        self.order = 2
        super().__init__()


class _BogackiShampineStep(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array(
            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
        )
        self.B = np.array([2 / 9, 1 / 3, 4 / 9, 0])
        self.B_hat = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
        self.C = np.array([0, 1 / 2, 3 / 4, 1])
        self.order = 3
        super().__init__()


class _RKF45Step(_RungeKuttaStep):
    def __init__(self) -> None:
        self.A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 4, 0, 0, 0, 0, 0],
                [3 / 32, 9 / 32, 0, 0, 0, 0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
            ]
        )
        self.B = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
        self.B_hat = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
        self.C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
        self.order = 5
        super().__init__()


# =====================================================================================
# Stepper Routines
# =====================================================================================


@dataclass(frozen=True, slots=True)
class ODEResult:
    y: np.ndarray
    t: np.ndarray


def _solve_to_fixed_step(
    f: callable, y0: np.ndarray, t_span: tuple[float, float], h: float, method: callable
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
        y.append(method(f, t[-1], y[-1], h))

    return ODEResult(np.asarray(y).T, np.asarray(t))


def _solve_to_richardson_extrapolation(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    h: float,
    method: callable,
    r_tol: float,
    a_tol: float,
    max_step: float,
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

    Returns
    -------
    ODEResult
        Results object containing the solution of the IVP.
    """
    t = [t_span[0]]
    y = [np.asarray(y0)]

    # need to half the max step as we are taking two steps at time
    max_step /= 2

    final_step = False

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        step_accepted = False
        while not step_accepted:
            # take two small steps to find y2
            y2 = y[-1]
            for _ in range(2):
                y2 = method(f, t[-1], y2, h)
            # take one large step two find w
            w = method(f, t[-1], y[-1], 2 * h)

            # eq 4.4 page 165 Hairer Solving ODEs 1
            local_err = (y2 - w) / (2**method.order - 1)

            # account for both relative and absolute error tolerances
            # eq 4.10 page 167 Hairer Solving ODEs 1
            scale = a_tol + np.maximum(np.abs(y2), np.abs(w)) * r_tol

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
                if (t[-1] + 2 * h - t_span[-1]) > 0 and not final_step:
                    final_step = True
                    h_new = (t_span[-1] - t[-1]) / 2
                else:
                    step_accepted = True
                    t.append(t[-1] + 2 * h)
                    y.append(y2)
            h = h_new

    return ODEResult(np.asarray(y).T, np.asarray(t))


def _solve_to_embedded(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    h: float,
    method: callable,
    r_tol: float,
    a_tol: float,
    max_step: float,
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
            # take two small steps to find y2
            y2, local_err = method(f, t[-1], y[-1], h)

            # account for both relative and absolute error tolerances
            # eq 4.10 page 167 Hairer Solving ODEs 1
            scale = a_tol + np.abs(y2) * r_tol

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
                    y.append(y2)
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
    "heun_euler": _HeunEulerStep,
    "rkf12": _RKF12Step,
    "bogacki_shampine": _BogackiShampineStep,
    "rkf45": _RKF45Step,
}


def solve_ivp(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    method: str,
    h: float = None,
    r_tol: float = 0,
    a_tol: float = 0,
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
        raise NotImplementedError

    if method in _fixed_step_methods:
        method = _fixed_step_methods[method]()
        if r_tol == 0 and a_tol == 0:
            # run in fixed mode
            return _solve_to_fixed_step(f_wrapper, y0, t_span, h, method)
        else:
            return _solve_to_richardson_extrapolation(
                f_wrapper, y0, t_span, h, method, r_tol, a_tol, max_step
            )
    elif method in _embedded_methods:
        method = _embedded_methods[method]()
        return _solve_to_embedded(
            f_wrapper, y0, t_span, h, method, r_tol, a_tol, max_step
        )
    else:
        raise ValueError(f"{method} is not a valid option for 'method'")
