import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scicomp.finite_diff import (
    get_central_diff_matrix,
    Grid,
    DirichletBC,
    NeumannBC,
    RobinBC,
    get_A_mat_from_BCs,
)
import scipy


class TestGetCentralDiffMatrix:
    def test_first_derive_3_point(self):
        A = get_central_diff_matrix(3, derivative=1)
        expected = np.array([[0, 0.5, 0], [-0.5, 0, 0.5], [0, -0.5, 0]])
        assert_allclose(A, expected)

        A = get_central_diff_matrix(4, derivative=1)
        expected = np.array(
            [[0, 0.5, 0, 0], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5], [0, 0, -0.5, 0]]
        )
        assert_allclose(A, expected)

    def test_second_derive_3_point(self):
        A = get_central_diff_matrix(3, derivative=2)
        expected = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
        assert_allclose(A, expected)

        A = get_central_diff_matrix(4, derivative=2)
        expected = np.array(
            [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
        )
        assert_allclose(A, expected)

    def test_invalid_size(self):
        msg = "Matrix must be at least 5*5"
        with pytest.raises(ValueError, match=msg):
            get_central_diff_matrix(2, derivative=2, points=5)


class TestGrid:
    def test_grid_dirichlet(self):
        left_bc = right_bc = DirichletBC(0)

        grid = Grid(10, 20, 100, left_BC=left_bc, right_BC=right_bc)

        assert grid.a == 10
        assert grid.b == 20
        assert grid.N == 100
        assert grid.N_inner == 98
        assert grid.dx == 10 / 99
        assert_equal(grid.x_inner, grid.x[1:-1])

    def test_grid_dirichlet_robin(self):
        left_bc = DirichletBC(0)
        right_bc = RobinBC(8, 9)

        grid = Grid(10, 20, 100, left_BC=left_bc, right_BC=right_bc)

        assert grid.a == 10
        assert grid.b == 20
        assert grid.N == 100
        assert grid.N_inner == 99
        assert grid.dx == 10 / 99
        assert_equal(grid.x_inner, grid.x[1:])


class TestGetAMat:
    def test_dirichlet(self):
        left_bc = right_bc = DirichletBC(0)
        grid = Grid(10, 20, 6, left_BC=left_bc, right_BC=right_bc)
        A = get_A_mat_from_BCs(2, grid, sparse=True)

        assert scipy.sparse.issparse(A)
        A = A.toarray()

        expected = np.array(
            [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
        )
        assert_equal(A, expected)

    def test_neumann(self):
        left_bc = right_bc = NeumannBC(0)
        grid = Grid(10, 20, 4, left_BC=left_bc, right_BC=right_bc)
        A = get_A_mat_from_BCs(2, grid, sparse=True)

        assert scipy.sparse.issparse(A)
        A = A.toarray()

        expected = np.array(
            [[-2, 2, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 2, -2]]
        )
        assert_equal(A, expected)

    def test_robin(self):
        left_bc = right_bc = RobinBC(10, 5)
        grid = Grid(10, 20, 4, left_BC=left_bc, right_BC=right_bc)
        A = get_A_mat_from_BCs(2, grid, sparse=True)

        assert scipy.sparse.issparse(A)
        A = A.toarray()

        expected = np.array(
            [
                [-2 + 2 * grid.dx * 5, 2, 0, 0],
                [1, -2, 1, 0],
                [0, 1, -2, 1],
                [0, 0, 2, -2 - 2 * grid.dx * 5],
            ]
        )
        assert_equal(A, expected)
