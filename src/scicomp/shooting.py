from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import root

from scicomp.integrate import solve_ivp


class LimitCycleNotFound(Exception):
    """Custom error for when a limit cycle is not found."""

    ...


class PhaseCondition(ABC):
    """Base class for defining phase conditions for limit cycle detection using
    numerical shooting.
    """

    @abstractmethod
    def __call__(self, f: Callable, y0: np.ndarray, **ode_params) -> float:
        """Implements the phase condition.

        Parameters
        ----------
        f : Callable
            RHS function of the ODE
        y0 : np.ndarray
            Current initial conditions

        Returns
        -------
        float
            Phase condition value
        """
        ...


class ICPhaseCondition(PhaseCondition):
    def __init__(self, value: float, component: int) -> None:
        """Phase condition for fixing one of the values of initial conditions eg.
        ``x(0)=0.4``.

        Parameters
        ----------
        value : float
            Value of the initial condition
        component : int
            Which variable to fix
        """
        self.value = value
        self.component = component

    def __call__(self, f: Callable, y0: np.ndarray, **ode_params) -> float:
        return y0[self.component] - self.value


class DerivativePhaseCondition(PhaseCondition):
    def __init__(self, component: int) -> None:
        """Implements a derivate phase condition eg. `x'(0) = 0`.

        Parameters
        ----------
        component : int
            Which component of the derivate to set to zero.
        """
        self.component = component

    def __call__(self, f: Callable, y0: np.ndarray, **ode_params) -> float:
        return f(0, y0, **ode_params)[self.component]


@dataclass(frozen=True, slots=True)
class LimitCycleResult:
    """Result object to store the detected limit cycle.

    Has the following attributes:
        y0 : ndarray
            The initial conditions to start on the limit cycle
        T : float
            The time period of the limit cycle
    """

    y0: np.ndarray
    T: float


def limit_cycle_shooting_func(
    x: np.ndarray,
    f: Callable,
    phase_condition: Callable,
    ivp_solver: Callable = solve_ivp,
    ivp_solver_kwargs: Optional[dict] = None,
    **ode_params,
) -> npt.ArrayLike:
    """Defines shooting function to fine the zeros of

    Parameters
    ----------
    x : np.ndarray
        Current augmented solution vector
    ivp_solver : Callable
        Solver to integrate the ODE
    f : Callable
        ODE to find limit cycle of
    phase_condition : Callable
        Phase condition to isolate limit cycle
    ivp_solver_kwargs : dict
        Options to pass to the ODE solver

    Returns
    -------
    npt.ArrayLike
        Current value of augmented function
    """
    ivp_solver_kwargs = {} if ivp_solver_kwargs is None else ivp_solver_kwargs

    # y(0) - y(T)
    period_condition = (
        x[:-1]
        - ivp_solver(
            f,
            t_span=(0, x[-1]),
            y0=x[:-1],
            **ivp_solver_kwargs,
            ode_params=ode_params,
        ).y[:, -1]
    )
    return [*period_condition, phase_condition(f, x[:-1], **ode_params)]


def find_limit_cycle(
    f: Callable,
    *,
    y0: npt.ArrayLike,
    T: float,
    phase_condition: PhaseCondition,
    ivp_solver: Callable = solve_ivp,
    root_finder: Callable = root,
    ivp_solver_kwargs: Optional[dict] = None,
    root_finder_kwargs: Optional[dict] = None,
    ode_params: Optional[dict] = None,
) -> LimitCycleResult:
    """Finds a limit cycle in the given ODE using numerical shooting.

    Parameters
    ----------
    f : Callable
        The RHS function of the ODE
    y0 : npt.ArrayLike
        Initial guess for initial conditions that lead to a limit cycle
    T : float
        Initial guess for the period of the limit cycle
    phase_condition : PhaseCondition
        `PhaseCondition` object which describes the phase condition to be applied
    ivp_solver : Callable, optional
        The IVP solver to use, by default `scicomp.integrate.solve_ivp`
    root_finder : Callable, optional
        The root finder to use, by default `scipy.optimize.root`
    ivp_solver_kwargs : Optional[dict], optional
        Keyword arguments to parse to the IVP solver, by default None
    root_finder_kwargs : Optional[dict], optional
        Keyword arguments to parse to the root finder, by default None
    ode_params: dict, optional
        Keyword arguments to be passed to `f`


    Returns
    -------
    LimitCycleResult
        Results object with the following attributes:
            y0 : ndarray
                The initial conditions to start on the limit cycle
            T : float
                The time period of the limit cycle

    Raises
    ------
    LimitCycleNotFound
        Raised if a limit cycle cannot be found. This may be as a result of a bad
        initial guess or if the ODE does not have a limit cycle.

    Examples
    --------

    In this example we will find a limit cycle in the predator prey equation.

    >>> from scicomp.odes import predator_prey
    >>> from scicomp.shooting import find_limit_cycle, DerivativePhaseCondition
    >>> params = {"a": 1, "d": 0.1, "b": 0.1}
    >>> pc = DerivativePhaseCondition(0)
    >>> solver_args = {"method": "rkf45", "r_tol": 1e-5}
    >>> find_limit_cycle(lambda t, y: predator_prey(t, y, a, b, d), y0=[0.8, 0.2],
    T=30, phase_condition=pc, ivp_solver_kwargs=solver_args, ode_params=ode_params)
    LimitCycleResult(y0=array([0.81897015, 0.16636103]), T=34.066559310372)
    """
    if T <= 0:
        raise ValueError("Initial guess of period 'T' must be positive")

    y0 = np.asarray(y0)
    ivp_solver_kwargs = dict() if ivp_solver_kwargs is None else ivp_solver_kwargs
    root_finder_kwargs = dict() if root_finder_kwargs is None else root_finder_kwargs
    ode_params = {} if ode_params is None else ode_params

    sol = root_finder(
        lambda x: limit_cycle_shooting_func(
            x, f, phase_condition, ivp_solver, ivp_solver_kwargs, **ode_params
        ),
        [*y0, T],
        **root_finder_kwargs,
    )

    if not sol.success:
        raise LimitCycleNotFound("No limit cycle found; potential bad initial guess.")

    return LimitCycleResult(sol.x[:-1], sol.x[-1])
