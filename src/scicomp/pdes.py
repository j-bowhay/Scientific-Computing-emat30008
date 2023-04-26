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
from scicomp.integrate import solve_ivp


@dataclass
class PDEResult:
    """Results class for storing the solution of a PDE.

    Has the following attributes:
        t : np.ndarray
            Time corresponding to the solution values
        u : np.ndarray
            Value of the solution at the grid points at times given by `t`
    """

    t: np.ndarray
    u: np.ndarray


def solve_diffusion_method_lines(
    *,
    grid: Grid,
    D: float,
    u0_func: Callable[[np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    q: Callable[
        [np.ndarray, np.ndarray, float], np.ndarray
    ] = lambda u, x, t: np.zeros_like(x),
    integrator: Callable = solve_ivp,
    integrator_kwargs: Optional[dict] = None,
    sparse: bool = False,
) -> PDEResult:
    """Solves the diffusion PDE with a source term using the method of lines. Uses
    `scicomp.integrate.solve_ivp` to integrate the solution forward in time.
    Configuration can be passed to the integrator using the `integrator_kwargs`
    parameter. This can be used specify if Eulers method, Runge Kutta or adaptive step
    sizing should be used.

    ``u_tt = D u_xx + q(u,x,t)``

    Parameters
    ----------
    grid : Grid
        Grid object defining the domain and boundary conditions to solve the problem
        over
    D : float
        Diffusion coefficient
    u0_func : Callable
        Function that describes the initial conditions of the problem. Must have
        signature ``u0_func(x) -> np.ndarray``.
    t_span : tuple
        Time period to solve pde over
    q : Callable, optional
        Function that defines the source term of the PDE. Must have signature
        ``q(u, x, t) -> np.ndarray``. Defaults to no source term.
    integrator : Callable, optional
        Function to use to integrate forwards in time, by default
        `scicomp.integrate.solve_ivp`
    integrator_kwargs : Optional, optional
        Optional arguments to be passed to the integrator to control method and
        tolerances etc, by default None
    sparse : bool, optional
        Whether to use sparse linear algebra. Defaults to False.

    Returns
    -------
    PDEResult
        Result object containing the solution to the PDE.
        Has the following attributes:
            t : np.ndarray
                Time corresponding to the solution values
            u : np.ndarray
                Value of the solution at the grid points at times given by `t`

    Examples
    --------

    In this example we will solve the dynamic Bratu problem. This is
    :math:`u_t = D u_{xx} + e^{\mu u}`. We will take ``D = 1.0`` and :math:`\mu=0.1`
    and use the boundary conditions ``u(0)=u(1)=0``.

    >>> from scicomp.pdes import solve_diffusion_method_lines
    >>> from scicomp.finite_diff import DirichletBC, Grid
    >>> D = 1
    >>> mu = 0.1
    >>> left_bc = right_bc = DirichletBC(0)
    >>> grid = Grid(a=0, b=1, N=20, left_BC=left_bc, right_BC=right_bc)
    >>> solve_diffusion_method_lines(grid=grid, D=D,
    ...                              u0_func=lambda x: np.zeros_like(x),
    ...                              q=lambda u, x, t: np.exp(mu * u),
    ...                              t_span=[0, 0.5])  # doctest: +ELLIPSIS
    PDEResult(t=array(...), u=array(...))
    """
    integrator_kwargs = {} if integrator_kwargs is None else integrator_kwargs

    A = get_A_mat_from_BCs(2, grid=grid, sparse=sparse)
    b = get_b_vec_from_BCs(grid=grid)

    # method of lines discretisation
    def rhs(t, y):
        return (D / (grid.dx) ** 2) * (A @ y + b) + q(y, grid.x_inner, t)

    u0 = u0_func(grid.x_inner)

    # integrate the solution forwards in time
    sol = integrator(rhs, y0=u0, t_span=t_span, **integrator_kwargs)

    return PDEResult(sol.t, apply_BCs_to_soln(sol.y.T, grid=grid))


def solve_diffusion_implicit(
    *,
    grid: Grid,
    D: float,
    dt: float,
    steps: int,
    u0_func: Callable[[np.ndarray], np.ndarray],
    q: Callable[
        [np.ndarray, np.ndarray, float], np.ndarray
    ] = lambda u, x, t: np.zeros_like(x),
    method: str = "crank-nicolson",
    sparse: bool = False,
) -> PDEResult:
    """Solve the diffusion equation with a source term using implicit method (either
    implicit Euler or Crank-Nicolson).

    ``u_tt = D u_xx + q(x,t,u)``

    Parameters
    ----------
    grid : Grid
        Grid object defining the domain and boundary conditions to solve
        the problem over
    D : float
        Diffusion coefficient
    dt : float
        Size of time step to take
    steps : int
        Number of time steps
    u0_func : Callable
        Function that describes the initial conditions of the problem.
        Must have signature ``u0_func(x) -> np.ndarray``.
    q : Callable, optional
        Function that defines the source term of the PDE. Must have signature
        ``q(u, x, t) -> np.ndarray``. If the source term contains ``u`` then the source
        term is calculated explicitly while the rest of the problem is calculated
        implicitly (IMEX). Defaults to no source term.
    method : str, optional
        Implicit method to use either "crank-nicolson" or "euler", by default
        "crank-nicolson"
    sparse : bool, optional
        Whether to use sparse linear algebra. Defaults to False.

    Returns
    -------
    PDEResult
        Result object containing the solution to the PDE.
        Has the following attributes:
            t : np.ndarray
                Time corresponding to the solution values
            u : np.ndarray
                Value of the solution at the grid points at times given by `t`

    Examples
    --------

    In this example we will solve the dynamic Bratu problem. This is
    :math:`u_t = D u_{xx} + e^{\mu u}`. We will take ``D = 1.0`` and :math:`\mu=0.1`
    and use the boundary conditions ``u(0)=u(1)=0``.

    >>> from scicomp.pdes import solve_diffusion_implicit
    >>> from scicomp.finite_diff import DirichletBC, Grid
    >>> D = 1
    >>> mu = 0.1
    >>> left_bc = right_bc = DirichletBC(0)
    >>> grid = Grid(a=0, b=1, N=20, left_BC=left_bc, right_BC=right_bc)
    >>> solve_diffusion_implicit(grid=grid, D=D,
    ...                              u0_func=lambda x: np.zeros_like(x),
    ...                              q=lambda u, x, t: np.exp(mu * u),
    ...                              dt=0.1, steps=10)  # doctest: +ELLIPSIS
    PDEResult(t=array(...), u=array(...))
    """
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

    A = get_A_mat_from_BCs(2, grid=grid, sparse=sparse)
    b = get_b_vec_from_BCs(grid=grid)

    if sparse:
        I = scipy.sparse.identity(b.shape[0])  # noqa: E741
    else:
        I = np.eye(*A.shape)  # type: ignore  # noqa: E741

    if method == "crank-nicolson":
        lhs = I - 0.5 * C * A
    else:
        lhs = I - C * A

    # step through time
    for i in range(1, steps + 1):
        if method == "crank-nicolson":
            rhs = (I + 0.5 * C * A) @ u[i - 1, :] + C * b
        else:
            rhs = u[i - 1, :] + C * b
        # add source term
        rhs += dt * q(u[i - 1, :], grid.x_inner, i * dt)

        if sparse:
            u[i, :] = scipy.sparse.linalg.spsolve(lhs, rhs)
        else:
            u[i, :] = scipy.linalg.solve(lhs, rhs)

    return PDEResult(dt * np.arange(steps + 1), apply_BCs_to_soln(u, grid=grid))
