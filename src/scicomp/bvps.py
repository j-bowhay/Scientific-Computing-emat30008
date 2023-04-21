from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy

from scicomp.finite_diff import (
    Grid,
    apply_BCs_to_soln,
    get_A_mat_from_BCs,
    get_b_vec_from_BCs,
)


def solve_linear_poisson_eq(
    *,
    grid: Grid,
    D: float,
    q: Callable[[np.ndarray], np.ndarray],
    sparse: bool = False,
) -> np.ndarray:
    r"""Convenience function for solving the linear poisson equation.

    .. math::

        D \frac{du}{dx} + q(x) = 0

    Parameters
    ----------
    grid : Grid
        Discretisation of the domain
    D : float
        Coefficient of diffusivity
    q : Callable[[np.ndarray], np.ndarray]
        Source term, must have signature ``q(x)``
    sparse : bool
        Whether to use sparse linear algebra.

    Returns
    -------
    np.ndarray
        Solution to the linear poisson equation
    """
    # generate finite difference matrix
    A = D * get_A_mat_from_BCs(2, grid=grid, sparse=sparse)
    b = get_b_vec_from_BCs(grid=grid)
    rhs = -D * b - (grid.dx**2) * q(grid.x_inner)

    if sparse:
        u_inner = scipy.sparse.linalg.spsolve(A, rhs).squeeze()
    else:
        u_inner = scipy.linalg.solve(A, rhs).squeeze()

    return apply_BCs_to_soln(u_inner, grid=grid)


def solve_nonlinear_poisson_eq(
    *,
    u0: np.ndarray,
    grid: Grid,
    D: float,
    q: Callable[[np.ndarray, np.ndarray], np.ndarray],
    root_finder_kwargs: Optional[dict] = None,
    sparse: bool = False,
) -> np.ndarray:
    r"""Convenience function for solving the non-linear poisson equation.

    .. math::

        D \frac{du}{dx} + q(u,x) = 0

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess at the solution
    grid : Grid
        Discretisation of the domain
    D : float
        Coefficient of diffusivity
    q : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Source term, must have signature ``q(u,x)``
    root_finder_kwargs : dict, optional
        Keyword arguments to pass to the root finder
    sparse : bool
        Whether to use sparse linear algebra

    Returns
    -------
    np.ndarray
        Solution to the non-linear poisson equation
    """
    root_finder_kwargs = {} if root_finder_kwargs is None else root_finder_kwargs

    # generate finite difference matrix
    A = get_A_mat_from_BCs(2, grid=grid, sparse=sparse)
    b = get_b_vec_from_BCs(grid=grid)

    # function to find the root of
    def eq(u):
        return D * (A @ u + b) + (grid.dx**2) * q(u, grid.x_inner)

    sol = scipy.optimize.root(eq, u0, **root_finder_kwargs)

    if sol.success:
        return apply_BCs_to_soln(sol.x, grid=grid)
    else:
        raise RuntimeError("Solution failed to converge.")
