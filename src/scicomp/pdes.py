from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy

from scicomp.finite_diff import (
    Grid,
    apply_BCs_to_soln,
    get_A_mat_from_BCs,
    get_b_vec_from_BCs,
)


@dataclass
class PDEResult:
    t: np.ndarray
    u: np.ndarray


def solve_diffusion_method_lines(
    grid: Grid,
    D: float,
    u0_func: Callable[[np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    source_term: Callable[
        [np.ndarray, float, np.ndarray], np.ndarray
    ] = lambda u, t, x: 0,
    integrator: Callable = scipy.integrate.solve_ivp,
    integrator_kwargs: Optional[dict] = None,
) -> PDEResult:
    integrator_kwargs = {} if integrator_kwargs is None else integrator_kwargs

    A = get_A_mat_from_BCs(2, grid=grid)
    b = get_b_vec_from_BCs(grid)

    def rhs(t, y):
        return (D / (grid.dx) ** 2) * (A @ y + b) + source_term(y, t, grid.x_inner)

    u0 = u0_func(grid.x_inner)

    sol = integrator(rhs, y0=u0, t_span=t_span, **integrator_kwargs)

    return PDEResult(sol.t, apply_BCs_to_soln(sol.y.T, grid))


def solve_diffusion_implicit(
    grid: Grid,
    D: float,
    dt: float,
    steps: int,
    u0_func: Callable[[np.ndarray], np.ndarray],
    method: str = "crank-nicolson",
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("Invalid 'dt'")
    if steps <= 0:
        raise ValueError("Invalid 'steps'")
    if method not in ["euler", "crank-nicolson"]:
        raise ValueError(f"{method} is not a valid method.")

    C = (dt * D) / (grid.dx**2)

    # preallocate and apply ICs
    u = np.empty((steps + 1, grid.N_inner))
    u[0, :] = u0_func(grid.x_inner)

    A = get_A_mat_from_BCs(2, grid=grid)
    b = get_b_vec_from_BCs(grid)

    if method == "crank-nicolson":
        lhs = np.eye(*A.shape) - 0.5 * C * A
    else:
        lhs = np.eye(*A.shape) - C * A

    # step through time
    for i in range(1, steps + 1):
        if method == "crank-nicolson":
            rhs = (np.eye(*A.shape) + 0.5 * C * A) @ u[i - 1, :] + C * b
        else:
            rhs = u[i - 1, :] + C * b
        u[i, :] = scipy.linalg.solve(lhs, rhs)

    return apply_BCs_to_soln(u, grid=grid)
