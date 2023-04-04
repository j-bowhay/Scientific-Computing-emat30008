from __future__ import annotations

from abc import ABC

import numpy as np
import scipy
from scipy._lib._finite_differences import _central_diff_weights


def get_central_diff_matrix(
    n: int, *, derivative: int, points: int = 3, sparse: bool = False
) -> np.ndarray | scipy.sparse.dia_matrix:
    """Generates a central difference matrix of arbitrary accuracy

    Parameters
    ----------
    n : int
        Size of the matrix (matrix will be ``n*n``)
    derivative : int
        The order of the derivate to calculate
    points : int, optional
        The number of points to use in the central difference stencil, by default 3
    sparse : bool, optional
        Return a sparse matrix instead of an array, by default False

    Returns
    -------
    np.ndarray | scipy.sparse.dia_matrix
        The central difference matrix
    """
    if n < points:
        raise ValueError(f"Matrix must be at least {points}*{points}")

    weights = _central_diff_weights(points, derivative)

    k = int(0.5 * (points - 1))
    offsets = list(range(-k, k + 1))
    diags = [weights[i] * np.ones(n - abs(offset)) for i, offset in enumerate(offsets)]

    A = scipy.sparse.diags(diags, offsets)

    return A if sparse else A.toarray()


class Grid:
    def __init__(self, a: float, b: float, N: int) -> None:
        """Grid class for representing the domain in finite difference methods

        Parameters
        ----------
        a : float
            Position of the left side of the domain
        b : float
            Position of the right side of the domain
        N : int
            Number of points to divide the domain into
        """
        if b <= a:
            raise ValueError("'b' must be greater than 'a'")
        if N < 1:
            raise ValueError("'N' must be positive")

        self.a = a
        self.b = b
        self.N = N
        self.x = np.linspace(a, b, N)

    @property
    def x_inner(self) -> np.ndarray:
        """Get the inner points of the domain

        Returns
        -------
        np.ndarray
            Location of the inner elements of the grid
        """
        return self.x[1:-1]

    @property
    def N_inner(self):
        return self.N - 2

    @property
    def dx(self):
        return (self.b - self.a) / (self.N - 1)


class BoundaryCondition(ABC):
    ...


class DirichletBC(BoundaryCondition):
    def __init__(self, gamma: float) -> None:
        r"""Dirichlet Boundary Condition ``u(b) = \gamma``

        Parameters
        ----------
        gamma : float
            Value of ``u`` at the boundary
        """
        self.gamma = gamma


class NeumannBC(BoundaryCondition):
    def __init__(self, delta: float) -> None:
        r"""Neumann Boundary Condition ``u'(b) = \delta``

        Parameters
        ----------
        value : float
            Value of ``u'`` at the boundary
        """
        self.value = delta


class RobinBC(BoundaryCondition):
    def __init__(self, delta: float, gamma: float) -> None:
        r"""Robin boundary condition ``u'(b) = delta - gamma u(b)``

        Parameters
        ----------
        delta : float
            Value of the derivative
        gamma : float
            Value of the solution
        """
        self.delta = delta
        self.gamma = gamma


def get_A_mat_from_BCs(
    derivative: int, grid: Grid, left_BC: BoundaryCondition, right_BC: BoundaryCondition
) -> np.ndarray:
    if derivative != 2:
        raise NotImplementedError

    if isinstance(left_BC, (DirichletBC, NeumannBC)) and isinstance(
        right_BC, (DirichletBC, NeumannBC)
    ):
        N = grid.N_inner
        if isinstance(left_BC, NeumannBC):
            N += 1
        if isinstance(right_BC, NeumannBC):
            N += 1
        return get_central_diff_matrix(N, derivative=2)
    elif isinstance(left_BC, RobinBC) and isinstance(right_BC, RobinBC):
        A = get_central_diff_matrix(grid.N, derivative=2)
        A[0, 0] += 2 * left_BC.delta
        A[-1, -1] -= 2 * right_BC.delta * grid.dx
    elif isinstance(left_BC, RobinBC) and isinstance(right_BC, DirichletBC):
        A = get_central_diff_matrix(grid.N, derivative=2)
        A[0, 0] += 2 * left_BC.delta
    elif isinstance(left_BC, DirichletBC) and isinstance(right_BC, (RobinBC)):
        A = get_central_diff_matrix(grid.N_inner + 1, derivative=2)
        A[-1, -1] -= right_BC.delta * grid.dx
    return A


def get_b_vec_from_BCs(
    grid: Grid, left_BC: BoundaryCondition, right_BC: BoundaryCondition
) -> np.ndarray:
    # Number of equations depends on the type of boundary condition
    if isinstance(left_BC, DirichletBC) and isinstance(right_BC, DirichletBC):
        b = np.zeros_like(grid.x_inner)
    elif (
        isinstance(left_BC, DirichletBC) and isinstance(right_BC, (NeumannBC, RobinBC))
    ) or (
        isinstance(right_BC, DirichletBC) and isinstance(left_BC, (NeumannBC, RobinBC))
    ):
        b = np.zeros((grid.N_inner + 1, 1))
    elif isinstance(left_BC, (NeumannBC, RobinBC)) and isinstance(
        right_BC, (NeumannBC, RobinBC)
    ):
        b = np.zeros_like(grid.x)

    if isinstance(left_BC, DirichletBC):
        b[0] = left_BC.gamma
    elif isinstance(left_BC, (NeumannBC, RobinBC)):
        b[0] = -2 * left_BC.delta * grid.dx
    else:
        raise NotImplementedError

    if isinstance(right_BC, DirichletBC):
        b[-1] = right_BC.gamma
    elif isinstance(right_BC, (NeumannBC, RobinBC)):
        b[-1] = 2 * right_BC.delta * grid.dx
    else:
        raise NotImplementedError

    return b


def apply_BCs_to_soln(
    inner_sol: np.ndarray, left_BC: BoundaryCondition, right_BC: BoundaryCondition
) -> np.ndarray:
    left_append = []
    right_append = []
    if isinstance(left_BC, (DirichletBC, RobinBC)):
        left_append = [left_BC.gamma]
    if isinstance(right_BC, (DirichletBC, RobinBC)):
        right_append = [right_BC.gamma]

    return np.concatenate([left_append, inner_sol, right_append])
