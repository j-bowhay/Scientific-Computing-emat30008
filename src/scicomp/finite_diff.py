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
    def __init__(self, value: float) -> None:
        """Dirichlet Boundary Condition

        Parameters
        ----------
        value : float
            Value of the boundary condition
        """
        self.value = value


class NeumannBC(BoundaryCondition):
    ...


class RobinBC(BoundaryCondition):
    ...


def get_b_vec_from_BCs(
    grid: Grid, left_BC: BoundaryCondition, right_BC: BoundaryCondition
) -> np.ndarray:
    b = np.zeros_like(grid.x_inner)

    if isinstance(left_BC, DirichletBC):
        b[0] = left_BC.value
    else:
        raise NotImplementedError

    if isinstance(right_BC, DirichletBC):
        b[-1] = right_BC.value
    else:
        raise NotImplementedError

    return b


def apply_BCs_to_soln(
    inner_sol: np.ndarray, left_BC: BoundaryCondition, right_BC: BoundaryCondition
) -> np.ndarray:
    if isinstance(left_BC, DirichletBC) and isinstance(right_BC, DirichletBC):
        return np.concatenate([[left_BC.value], inner_sol, [right_BC.value]])
    else:
        raise NotImplementedError
