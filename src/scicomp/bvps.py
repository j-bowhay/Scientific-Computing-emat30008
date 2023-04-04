from __future__ import annotations

from typing import Callable

import numpy as np
import scipy

from scicomp.finite_diff import (
    Grid,
    BoundaryCondition,
    get_central_diff_matrix,
    get_b_vec_from_BCs,
    apply_BCs_soln,
)


def solve_linear_poisson_eq(
    grid: Grid,
    left_BC: BoundaryCondition,
    right_BC: BoundaryCondition,
    D: float,
    q: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    r"""Convenience function for solving the linear poisson equation.

    .. math::
        D \frac{du}{dx} + q(x) = 0

    Parameters
    ----------
    grid : Grid
        Discretisation of the domain
    left_BC : BoundaryCondition
        Boundary condition at the left of the domain
    right_BC : BoundaryCondition
        Boundary condition at the right of the domain
    D : float
        Coefficient of diffusivity
    q : Callable[[np.ndarray], np.ndarray]
        Source term, must have signature `q(x)`

    Returns
    -------
    np.ndarray
        Solution to the linear poisson equation
    """
    A = D * get_central_diff_matrix(grid.N_inner, derivative=2)
    b_DD = get_b_vec_from_BCs(grid, left_BC, right_BC)
    rhs = -D * b_DD - (grid.dx**2) * q(grid.x_inner)

    u_inner = scipy.linalg.solve(A, rhs).squeeze()

    return apply_BCs_soln(u_inner, left_BC, right_BC)


def solve_nonlinear_poisson_eq():
    ...
