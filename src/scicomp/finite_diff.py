from __future__ import annotations

from abc import ABC, abstractmethod

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

    # use central diff weights to create a finite difference matrix
    k = int(0.5 * (points - 1))
    offsets = list(range(-k, k + 1))
    diags = [weights[i] * np.ones(n - abs(offset)) for i, offset in enumerate(offsets)]

    A = scipy.sparse.diags(diags, offsets, format="csr")

    return A if sparse else A.toarray()


class Grid:
    def __init__(
        self,
        a: float,
        b: float,
        N: int,
        left_BC: BoundaryCondition,
        right_BC: BoundaryCondition,
    ) -> None:
        """Grid class for representing the domain in finite difference methods

        Parameters
        ----------
        a : float
            Position of the left side of the domain
        b : float
            Position of the right side of the domain
        N : int
            Number of points to divide the domain into
        left_BC : BoundaryCondition
            BC at the left of the domain
        right_BC : BoundaryCondition
            BC at the right of the domain
        """
        if b <= a:
            raise ValueError("'b' must be greater than 'a'")
        if N < 1:
            raise ValueError("'N' must be positive")

        self.a = a
        self.b = b
        self.N = N
        self.x = np.linspace(a, b, N)
        self.left_BC = left_BC
        self.right_BC = right_BC

    @property
    def x_inner(self) -> np.ndarray:
        """Get the points of the domain required for computation based on the BCs

        Returns
        -------
        np.ndarray
            Location of the inner elements of the grid
        """
        if isinstance(self.left_BC, DirichletBC) and isinstance(
            self.right_BC, DirichletBC
        ):
            return self.x[1:-1]
        elif isinstance(self.left_BC, DirichletBC) and isinstance(
            self.right_BC, (NeumannBC, RobinBC)
        ):
            return self.x[1:]
        elif isinstance(self.right_BC, DirichletBC) and isinstance(
            self.left_BC, (NeumannBC, RobinBC)
        ):
            return self.x[:-1]
        elif isinstance(self.left_BC, (NeumannBC, RobinBC)) and isinstance(
            self.right_BC, (NeumannBC, RobinBC)
        ):
            return self.x
        else:
            raise NotImplementedError

    @property
    def N_inner(self) -> int:
        """Returns the number of grid points required for the computation based
        on the BCs

        Returns
        -------
        int
           Number of grid points needed
        """
        N = self.N - 2
        if isinstance(self.left_BC, (NeumannBC, RobinBC)):
            N += 1
        if isinstance(self.right_BC, (NeumannBC, RobinBC)):
            N += 1
        return N

    @property
    def dx(self) -> float:
        """Distance between grid points

        Returns
        -------
        float
            Distance between grid points
        """
        return (self.b - self.a) / (self.N - 1)


class BoundaryCondition(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Abstract class for representing a Boundary Condition"""
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
        self.delta = delta


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


def get_A_mat_from_BCs(derivative: int, grid: Grid, sparse: bool = False) -> np.ndarray:
    """Generates finite difference matrix to solve PDE problems based on the boundary
    conditions.

    Parameters
    ----------
    derivative : int
        Order of the derivative
    grid : Grid
        Grid object containing boundary conditions
    sparse : bool
        Whether to store A in a sparse format. Only implemented for Dirichlet boundary
        conditions.

    Returns
    -------
    np.ndarray
        Finite difference matrix
    """
    left_BC = grid.left_BC
    right_BC = grid.right_BC

    A = get_central_diff_matrix(grid.N_inner, derivative=derivative, sparse=sparse)

    if isinstance(left_BC, DirichletBC) and isinstance(right_BC, DirichletBC):
        return A

    # higher order derivates and sparse matrices with non Dirichlet bcs aren't implemented
    if derivative != 2 or sparse:
        raise NotImplementedError

    # Changes to the finite difference matrix are required based on the
    # boundary conditions
    if isinstance(left_BC, RobinBC):
        A[0, 0] += 2 * left_BC.gamma * grid.dx
        A[0, 1] += 1
    elif isinstance(left_BC, NeumannBC):
        A[0, 1] += 1

    if isinstance(right_BC, RobinBC):
        A[-1, -1] -= 2 * right_BC.gamma * grid.dx
        A[-1, -2] += 1
    elif isinstance(right_BC, NeumannBC):
        A[-1, -2] += 1

    return A


def get_b_vec_from_BCs(grid: Grid) -> np.ndarray:
    """Get boundary condition vector based on `grid`.

    Parameters
    ----------
    grid : Grid
        The object to get boundary conditions from.

    Returns
    -------
    np.ndarray
        Boundary condition vector
    """
    left_BC = grid.left_BC
    right_BC = grid.right_BC

    # Number of equations depends on the type of boundary condition
    if isinstance(left_BC, DirichletBC) and isinstance(right_BC, DirichletBC):
        b = np.zeros_like(grid.x_inner)
    elif (
        isinstance(left_BC, DirichletBC) and isinstance(right_BC, (NeumannBC, RobinBC))
    ) or (
        isinstance(right_BC, DirichletBC) and isinstance(left_BC, (NeumannBC, RobinBC))
    ):
        b = np.zeros((grid.N_inner, 1))
    elif isinstance(left_BC, (NeumannBC, RobinBC)) and isinstance(
        right_BC, (NeumannBC, RobinBC)
    ):
        b = np.zeros_like(grid.x)

    # Value of b depends on the boundary conditions
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


def apply_BCs_to_soln(inner_sol: np.ndarray, grid: Grid) -> np.ndarray:
    """Applies the value of Dirichlet Boundary conditions to a solution

    Parameters
    ----------
    inner_sol : np.ndarray
        Solution for inner grid points
    grid : Grid
        Grid object containing boundary conditions

    Returns
    -------
    np.ndarray
        Complete solution including boundary conditions
    """
    left_BC = grid.left_BC
    right_BC = grid.right_BC
    inner_sol = np.atleast_2d(inner_sol)

    left_append = np.array([[]])
    right_append = np.array([[]])
    if isinstance(left_BC, DirichletBC):
        left_append = np.broadcast_to([left_BC.gamma], (inner_sol.shape[0], 1))
    if isinstance(right_BC, DirichletBC):
        right_append = np.broadcast_to([right_BC.gamma], (inner_sol.shape[0], 1))

    return np.hstack([left_append, inner_sol, right_append]).squeeze()
