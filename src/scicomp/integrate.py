from __future__ import annotations

import inspect
import math
from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

# =====================================================================================
# Utils
# =====================================================================================


def _scale(r_tol: float, a_tol: float, *args) -> np.ndarray:
    """Calculates the scale term for the error norm. Based on eq 4.10 from Hairer.

    Parameters
    ----------
    r_tol : float
        The desired relative tolerance
    a_tol : float
        The desired absolute tolerance

    Returns
    -------
    np.ndarray
        Array containing the scale term for each component of the solution.
    """
    if len(args) > 1:
        y = np.maximum(*tuple(map(np.abs, args)))
    else:
        y = np.abs(args[0])
    return a_tol + y * r_tol


def _error_norm(x: np.ndarray, /) -> float:
    """Calculates the error norm based on eq 4.11 from Hairer.

    Parameters
    ----------
    x : np.ndarray
        Vector to compute the norm of

    Returns
    -------
    float
        Error norm of `x`
    """
    return np.linalg.norm(x) / (x.size**0.5)


# =====================================================================================
# Step Routines
# =====================================================================================

# Standard Runge Kutta Type Steps


@dataclass
class _StepResult:
    """Used internally to store the result of a Runge Kutta step."""

    y: np.ndarray
    error_estimate: Optional[np.ndarray] = None


class _RungeKuttaStep(ABC):
    @abstractproperty
    def A(self) -> np.ndarray:
        """The ``A`` matrix of the Butcher tableau"""
        ...

    @abstractproperty
    def B(self) -> np.ndarray:
        """The ``B`` vector of the Butcher tableau"""
        ...

    @abstractproperty
    def C(self) -> np.ndarray:
        """The ``C`` vector of the Butcher tableau"""
        ...

    @abstractproperty
    def order(self) -> int:
        """The order of the integrator"""
        ...

    @property
    def B_hat(self) -> Optional[np.ndarray]:
        """The optional embedded error estimate of the integrator"""
        return None

    def __init__(self) -> None:
        self.s = self.B.size

    def __call__(self, f: Callable, t: float, y: np.ndarray, h: float) -> _StepResult:
        r"""Computes one step of the ode ``y' = f(t,y)``.

        Based on the following equation,

        .. math::

            y_{n+1} = y_n + h \sum_{i=1}^s b_i k_i,

        where

        .. math::

            k_i = f(t_n + c_i h, y_n + h \sum_{i=1}^s a_{ij}k_j).

        Parameters
        ----------
        f : Callable
            The rhs function.
        t : float
            The current time.
        y : np.ndarray
            The current state.
        h : float
            The step size.

        Returns
        -------
        _StepResult
            Result object containing the next value for `y` and the error estimate if
            integrator has one.
        """
        k = np.empty((y.size, self.s))
        # calculate k values
        for i in range(self.s):
            k[:, i] = f(
                t + self.C[i] * h,
                y + h * np.sum(self.A[i, np.newaxis, : i + 1] * k[:, : i + 1], axis=-1),
            )

        y1 = y + h * np.inner(self.B, k)

        # return the error estimate if there is an embedded formula
        if self.B_hat is not None:
            return _StepResult(y1, h * np.inner(self.B - self.B_hat, k))
        return _StepResult(y1)


class _EulerStep(_RungeKuttaStep):
    """Defines Butcher Tableau for the Forward Euler method.
    https://en.wikipedia.org/wiki/Euler_method
    """

    A = np.array([[0]])
    B = np.array([1])
    C = np.array([0])
    order = 1


class _ExplicitMidpointStep(_RungeKuttaStep):
    """Defines the Butcher Tableau for the explicit midpoint method.
    https://en.wikipedia.org/wiki/Midpoint_method
    """

    A = np.array([[0, 0], [0.5, 0]])
    B = np.array([0, 1])
    C = np.array([0, 0.5])
    order = 2


class _HeunsStep(_RungeKuttaStep):
    """Defines the Butcher Tableau for Huan's method.
    https://en.wikipedia.org/wiki/Heun%27s_method
    """

    A = np.array([[0, 0], [1, 0]])
    B = np.array([0.5, 0.5])
    C = np.array([0, 1])
    order = 2


class _RalstonStep(_RungeKuttaStep):
    """Defines the Butcher Tableau for Ralston's method."""

    A = np.array([[0, 0], [2 / 3, 0]])
    B = np.array([1 / 4, 3 / 4])
    C = np.array([0, 2 / 3])
    order = 2


class _Kutta3Step(_RungeKuttaStep):
    """Defines the Butcher Tableau for Kutta's third-order method."""

    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
    B = np.array([1 / 6, 2 / 3, 1 / 6])
    C = np.array([0, 1 / 2, 1])
    order = 3


class _Heun3Step(_RungeKuttaStep):
    """Defines the Butcher Tableau for Heun's third-order method"""

    A = np.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]])
    B = np.array([1 / 4, 0, 3 / 4])
    C = np.array([0, 1 / 3, 2 / 3])
    order = 3


class _Wray3Step(_RungeKuttaStep):
    """Defines the Butcher Tableau for Van der Houwen's/Wray third-order method"""

    A = np.array([[0, 0, 0], [8 / 15, 0, 0], [1 / 4, 5 / 12, 0]])
    B = np.array([1 / 4, 0, 3 / 4])
    C = np.array([0, 8 / 15, 2 / 3])
    order = 3


class _Ralston3Step(_RungeKuttaStep):
    """Define the Butcher Tableau for Ralston's third order method.
    https://www.ams.org/journals/mcom/1962-16-080/S0025-5718-1962-0150954-0/
    """

    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    C = np.array([0, 1 / 2, 3 / 4])
    order = 3


class _SSPRK3Step(_RungeKuttaStep):
    """Defines the Butcher Tableau for the Third-order Strong Stability Preserving
    Runge-Kutta.
    """

    A = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
    B = np.array([1 / 6, 1 / 6, 2 / 3])
    C = np.array([0, 1, 1 / 2])
    order = 3


class _RK4Step(_RungeKuttaStep):
    """Butcher Tableau for the classic fourth-order Rung-Kutta method."""

    A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
    B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    C = np.array([0, 0.5, 0.5, 1])
    order = 4


class _RK38Step(_RungeKuttaStep):
    """Butcher Tableau for the Runge Kutta 3/8-rule fourth-order method"""

    A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])

    B = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    C = np.array([0, 1 / 3, 2 / 3, 1])
    order = 4


class _Ralston4Step(_RungeKuttaStep):
    """Butcher Tableau for the Ralston's fourth order method.
    https://www.ams.org/journals/mcom/1962-16-080/S0025-5718-1962-0150954-0/
    """

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
    """Butcher tableau for the Bogacki-Shampine method
    https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    """

    A = np.array(
        [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
    )
    B_hat = np.array([2 / 9, 1 / 3, 4 / 9, 0])
    B = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
    C = np.array([0, 1 / 2, 3 / 4, 1])
    order = 3


class _RKF45Step(_RungeKuttaStep):
    """Butcher Tableau for the Runge-Kutta-Fehlberg method
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    """

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
    B_hat = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    B = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
    C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    order = 5


class _CashKarpStep(_RungeKuttaStep):
    """Butcher Tableau for the Cash-Karp method
    https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method
    """

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [3 / 10, -9 / 10, 6 / 5, 0, 0, 0],
            [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0, 0],
            [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096, 0],
        ]
    )
    B_hat = np.array([37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771])
    B = np.array([2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4])
    C = np.array([0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8])
    order = 5


class _DomandPrinceStep(_RungeKuttaStep):
    """Butcher Tableau for the Dormand-Prince method
    https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    """

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
    )
    B_hat = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    B = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    order = 5


# =====================================================================================
# Stepper Routines
# =====================================================================================


@dataclass(frozen=True, slots=True)
class ODEResult:
    """Results object to store the solution of the ODE.

    Has the following attributes:
        y : np.ndarray
            The solution of the system at each timestep
        t : np.ndarray
            Time points of the solution
    """

    y: np.ndarray
    t: np.ndarray


def _solve_to_fixed_step(
    f: Callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    h: float,
    method: _RungeKuttaStep,
) -> ODEResult:
    """Private function for solving an ODE using a fixed timestep.

    Parameters
    ----------
    f : Callable
        ODE to solve
    y0 : np.ndarray
        Initial condition
    t_span : tuple[float, float]
        Time span to solve over
    h : float
        Step size to use
    method : _RungeKuttaStep
        Stepper to use

    Returns
    -------
    ODEResult
        Solution to `f`
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
    t: float,
    y: np.ndarray,
    h: float,
    method: _RungeKuttaStep,
) -> tuple[np.ndarray, np.ndarray]:
    """Error estimate for solving from ``y_n`` to ``y_n+1`` using Richardson
    extrapolation.

    Parameters
    ----------
    f : Callable
        RHS of the ODE
    t : float
        Time of current state occurs at
    y : np.ndarray
        Current state of the ODE
    h : float
        Step size
    method : _RungeKuttaStep
        Stepper to use for integration

    Returns
    -------
    tuple[np.ndarray, float]
        Solution of the ODE at the next timestep and estimated error in getting there
    """
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
    t: float,
    y: np.ndarray,
    h: float,
    method: _RungeKuttaStep,
) -> tuple[np.ndarray, np.ndarray]:
    """Error estimate for solving from ``y_n`` to ``y_n+1`` using the steppers
    embedded error estimate.

    Parameters
    ----------
    f : Callable
        RHS of the ODE
    t : float
        Time of current state occurs at
    y : np.ndarray
        Current state of the ODE
    h : float
        Step size
    method : _RungeKuttaStep
        Stepper to use for integration

    Returns
    -------
    tuple[np.ndarray, float]
        Solution of the ODE at the next timestep and estimated error in getting there
    """
    step = method(f, t, y, h)
    y1 = step.y
    local_err = step.error_estimate

    return y1, local_err  # type:ignore


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
    """Private function for solving ODE using an adaptive timestep.

    Parameters
    ----------
    f : Callable
        RSH of the ODE.
    y0 : np.ndarray
        Initial conditions
    t_span : tuple[float, float]
        Time span to solve over
    h : float
        Initial timestep
    method : _RungeKuttaStep
        Stepper to use
    r_tol : float
        The relative error tolerance
    a_tol : float
        The absolute error tolerance
    max_step : float
        Maximum allowable step size to take
    error_estimate : Callable
        Function to use as the error estimate

    Returns
    -------
    ODEResult
        The solution to the ODE

    Raises
    ------
    RuntimeError
        Raised if no step size satisfies the error criteria
    """
    fac_max = 1.5
    fac_min = 0.5
    safety_fac = 0.9

    t = [t_span[0]]
    y = [np.asarray(y0)]

    final_step = False

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        step_accepted = False
        while not step_accepted:
            y1, local_err = error_estimate(f, t[-1], y[-1], h, method)

            scale = _scale(r_tol, a_tol, y1, y[-1])
            err = _error_norm(local_err / scale)

            # adjust step size
            # eq 4.13 Hairer
            if err == 0:
                h_new = h * fac_max
            else:
                h_new = h * min(
                    fac_max,
                    max(fac_min, safety_fac * (1 / err) ** (1 / (method.order + 1))),
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
    "kutta3": _Kutta3Step,
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
    "ck45": _CashKarpStep,
    "dopri45": _DomandPrinceStep,
}

_all_methods = {**_fixed_step_methods, **_embedded_methods}


def _estimate_initial_step_size(
    f: Callable,
    y0: np.ndarray,
    t0: float,
    method: _RungeKuttaStep,
    r_tol: float,
    a_tol: float,
    max_step: float,
) -> float:
    """Private function to estimate a suitable initial step size. Algorithm described
    on page 169 of Hairer Solving Ordinary Differential Equations 1.

    Parameters
    ----------
    f : Callable
        RHS function of the ODE
    y0 : np.ndarray
        Initial conditions
    t0 : float
        Initial Time
    method : _RungeKuttaStep
        Integrator being used
    r_tol : float
        Relative error tolerance
    a_tol : float
        Absolute error tolerance
    max_step : float
        Maximum allowable stepsize

    Returns
    -------
    float
        Estimate for the initial step size to use
    """
    # step a
    f0 = f(t0, y0)
    scale = _scale(r_tol, a_tol, f0, y0)
    d0 = _error_norm(y0 / scale)
    d1 = _error_norm(f0 / scale)

    # step b
    if d0 < 1e-5 or d1 < 1e-5 or math.isnan(d0) or math.isnan(d1):
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    # step c
    y1 = _EulerStep()(f, t0, y0, h0).y

    # step d
    diff = f(t0 + h0, y1) - f(t0, y0)
    scale = _scale(r_tol, a_tol, diff)
    d2 = _error_norm(diff / scale)

    # step e
    if max(d1, d2) <= 1e-15 or math.isnan(d1) or math.isnan(d2):
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (method.order + 1))

    # step d
    h = max(100 * h0, h1)
    return h if h < max_step else max_step


def solve_ivp(
    f: Callable,
    *,
    y0: npt.ArrayLike,
    t_span: tuple[float, float],
    method: str = "rkf45",
    mode: str = "adaptive",
    h: Optional[float] = None,
    r_tol: float = 1e-3,
    a_tol: float = 1e-6,
    max_step: float = np.inf,
) -> ODEResult:
    """Solves the IVP from a system of ODEs

    Has three primary modes of execution, see examples for further details.
    1. Solve the ODE using a fixed timestep. This is used if ``mode=="fixed"``.
    2. Solve the ODE using a adaptive timestep using Richardson Extrapolation as the
    error estimate. This is used if ``mode=="adaptive"`` and `method` does not
    have an embedded error estimate.
    3. Solve the ODE using a adaptive timestep using an embedded error estimate. This
    is used if ``mode=="adaptive"`` and `method` does have an embedded error
    estimate.

    `h` is optional if operating as in modes (2) or (3) as a suitable initial step
    size will be determined algorithmically.

    Parameters
    ----------
    f : Callable
        RHS function of the ODE. Must have signature ``f(t,y) -> array_like``. Any
        parameters to the ODE should be handled by wrapping the ODE in a function or
        anonymous function
    y0 : npt.ArrayLike
        Initial conditions
    t_span : tuple[float, float]
        Interval of integration
    method : str
        The integrator to use.

        Methods with embedded error estimates:
        - ``"bogacki_shampine"``
        - ``"rkf45"``: The classic Runge-Kutta-Fehlberg method
        - ``"ck45"``: Cash-Karp
        - ``"dopri45"``: Forth order Dormand-Prince

        Methods without embedded error estimate:
        -``"euler"``: Forward Euler (first order)
        -``"midpoint"``: Explicit midpoint (second order)
        -``"heun"``: Heun's method (second order). Also known as the explicit trapezoid
        rule or modified Euler's method.
        -``"ralston"``: Ralston's method (second order)
        -``"kutta3"``: Kutta's third-order method
        -``"heun3"``: Heun's third-order method
        -``"wray3"``: Van der Houwen's/Wray third-order method
        -``"ralston3"``: Ralston's third-order method
        -``"ssprk3"``: Third-order Strong Stability Preserving Runge-Kutta
        -``"rk4"``: Classic fourth order Runge-Kutta
        -``"rk38"``: Fourth order Runge-Kutta 3/8-rule
        -``"ralston4"``: Ralston's fourth-order method
    h : Optional[float], optional
        Step size. If `mode=="fixed"`` then this is used as a
        fixed step size for the integration scheme. Otherwise this is the initial step
        size used in the adaptive integration scheme.
    r_tol : float, optional
        Relative error tolerance, by default 1e-3. Adaptive solver keeps the local error
        bellow this tolerance.
    a_tol : float, optional
        Absolute error tolerance, by default 1e-6. Adaptive solver keeps the local error
        bellow this tolerance.
    max_step : float, optional
        Maximum allowable step size, by default np.inf

    Returns
    -------
    ODEResult
        Results object with the following attributes:
            - ``y``: np.ndarray
                Solution at t.
            - ``t``: np.ndarray
                Time correspond to the solution.

    Examples
    --------

    In this example we will demonstrate solving an ODE with the different solver modes

    >>> from scicomp.integrate import shm_ode
    >>> from scicomp.integrate import solve_ivp
    >>> def rhs(t,y):
    ...     return shm_ode(t, y, 1)
    >>> t_span = (0, 10)
    >>> y0 = [0.5, 0.5]

    Solving using a fixed timestep.

    >>> solve_ivp(rhs, y0=y0, t_span=t_span, method="rk4", mode="fixed, h=1e-2)
    ODEResult(y=array([[ 0.5       ,  0.50497492,  0.50989934, ..., -0.69003652,
            -0.69154632, -0.69154632],
           [ 0.5       ,  0.49497508,  0.48990067, ..., -0.15443318,
            -0.14752521, -0.14752521]]), t=array([ 0.  ,  0.01,  0.02, ...,  9.99, 10.  , 10.  ]))

    Solve using an adaptive timestep with Richardson Extrapolation as the error estimate
    with an initial timestep provided

    >>> solve_ivp(rhs, y0=y0, t_span=t_span, method="rk4", h=1e-2)
    ODEResult(y=array([[ 0.5       ,  0.50497492,  0.51234246,  0.52317711,  0.53893083,
             0.56140509,  0.59240227,  0.63244547,  0.67700782,  0.70663495,
             0.66474459,  0.42583101, -0.1532335 , -0.66590121, -0.51947298,
             0.13562252,  0.66956765,  0.50947758, -0.15183137, -0.67320671,
            -0.68945685],
           [ 0.5       ,  0.49497508,  0.48734506,  0.47569497,  0.4577702 ,
             0.429912  ,  0.38608231,  0.31624789,  0.20410881,  0.02582602,
            -0.24106818, -0.56449915, -0.69023335, -0.23717996,  0.4790413 ,
             0.69332804,  0.22469763, -0.48876838, -0.6893248 , -0.21150438,
            -0.15030182]]), t=array([ 0.        ,  0.01      ,  0.025     ,  0.0475    ,  0.08125   ,
             0.131875  ,  0.2078125 ,  0.32171875,  0.49257813,  0.74886719,
             1.13330078,  1.70995117,  2.57492676,  3.58559481,  4.67340325,
             5.69296438,  6.74742511,  7.83657281,  8.86004401,  9.91023249,
             10.        ]))

    Solve using an adaptive timestep with Richardson Extrapolation as the error estimate
    with no initial timestep provided

    >>> solve_ivp(rhs, y0=y0, t_span=t_span, method="rk4")
    ODEResult(y=array([[ 0.5       ,  0.69081285,  0.21648852, -0.49580923, -0.68732638,
            -0.20301247,  0.50573613,  0.68248609,  0.18936009, -0.51541622,
            -0.68893827],
           [ 0.5       , -0.15022447, -0.67279137, -0.50332778,  0.1627995 ,
             0.67633675,  0.49246833, -0.17960291, -0.67964657, -0.4814253 ,
            -0.15094109]]), t=array([ 0.        ,  1.        ,  2.04594544,  3.1358502 ,  4.16188388,
             5.20909215,  6.30011048,  7.33009085,  8.37241704,  9.46434924,
            10.        ]))

    Solve using adaptive timestep using an embedded error estimate and no initial timestep

    >>> solve_ivp(rhs, y0=y0, t_span=t_span, method="rkf45")
        ODEResult(y=array([[ 0.5       ,  0.70345933,  0.39396531, -0.18910256, -0.63384444,
                -0.65128745, -0.25118104,  0.34182193,  0.6847462 ,  0.58109135,
                 0.09805386, -0.47581897, -0.69332318],
               [ 0.5       , -0.07410707, -0.58775393, -0.68205051, -0.31534837,
                 0.27807466,  0.66227544,  0.62061071,  0.18255243, -0.40600934,
                -0.7022262 , -0.52597345, -0.14967229]]), t=array([ 0.        ,  0.89064755,  1.76623584,  2.62743512,  3.46630238,
                 4.3317307 ,  5.13663241,  6.00282512,  6.80978582,  7.68042683,
                 8.50280141,  9.37716803, 10.        ]))
    """  # noqa: E501
    if not callable(f):
        raise ValueError("'f' must be callable")
    else:
        f_sig = inspect.signature(f)

        if list(f_sig.parameters) != ["t", "y"]:
            raise ValueError("'f' has an invalid signature")

    if len(t_span) != 2 or t_span[0] > t_span[1]:
        raise ValueError("Invalid values for 't_span'")

    if h is None and mode == "fixed":
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

    if mode not in ["adaptive", "fixed"]:
        raise ValueError(f"{mode} is not a valid option for 'mode'")

    if h is None:
        # compute initial step size
        h = _estimate_initial_step_size(
            f_wrapper, y0, t_span[0], method_step, r_tol, a_tol, max_step
        )

    if mode == "fixed":
        # run in fixed mode
        res = _solve_to_fixed_step(f_wrapper, y0, t_span, h, method_step)
    else:
        if method in _fixed_step_methods:
            err_estimate = _richardson_error_estimate
        else:
            err_estimate = _embedded_error_estimate
        res = _solve_to_adaptive(
            f_wrapper,
            y0,
            t_span,
            h,
            method_step,
            r_tol,
            a_tol,
            max_step,
            err_estimate,
        )
    return res
