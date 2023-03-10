from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import scipy
from scipy.optimize import root

from scicomp.integrate import solve_ivp


class PhaseCondition(ABC):
    @abstractmethod
    def __call__(self, f: Callable, y0: np.ndarray) -> float:
        ...


class ICPhaseCondition(PhaseCondition):
    def __init__(self, value: float, component: int) -> None:
        self.value = value
        self.component = component

    def __call__(self, f: Callable, y0: np.ndarray) -> float:
        return y0[self.component] - self.value


class DerivativePhaseCondition(PhaseCondition):
    def __init__(self, component: int) -> None:
        self.component = component

    def __call__(self, f: Callable, y0: np.ndarray) -> float:
        return f(0, y0)[self.component]


@dataclass(frozen=True, slots=True)
class LimitCycleResult:
    y0: np.ndarray
    T: float


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
) -> LimitCycleResult:
    y0 = np.asarray(y0)
    ivp_solver_kwargs = dict() if ivp_solver_kwargs is None else ivp_solver_kwargs
    root_finder_kwargs = dict() if root_finder_kwargs is None else root_finder_kwargs

    def G(x):
        period_condition = (
            x[:-1]
            - ivp_solver(f, t_span=(0, x[-1]), y0=x[:-1], **ivp_solver_kwargs).y[:, -1]
        )
        return [*period_condition, phase_condition(f, x[:-1])]

    sol = root_finder(G, [*y0, T], **root_finder_kwargs)
    if not sol.success:
        raise RuntimeError("No limit cycle found")
    return LimitCycleResult(sol.x[:-1], sol.x[-1])


def _find_limit_cycle(ode: Callable, x0: np.ndarray) -> np.ndarray:
    def condition(x):
        condition_1 = (
            x[:2] - scipy.integrate.solve_ivp(ode, (0, x[2]), x[:2], rtol=1e-5).y[:, -1]
        )
        # make this generic
        condition_2 = ode(np.nan, x[:2])[0]
        return [*condition_1, condition_2]

    return scipy.optimize.root(condition, x0).x
