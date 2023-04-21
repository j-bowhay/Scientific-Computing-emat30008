import numpy as np
import pytest
from numpy.testing import assert_allclose
from scicomp.bvps import solve_linear_poisson_eq, solve_nonlinear_poisson_eq
from scicomp.finite_diff import DirichletBC, Grid, NeumannBC, RobinBC


class TestSolveLinearPoissonEquation:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_no_source(self, sparse):
        D = 1
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)
        grid = Grid(a=a, b=b, N=100, left_BC=left_BC, right_BC=right_BC)

        sol = solve_linear_poisson_eq(grid=grid, D=D, q=lambda x: 0, sparse=sparse)

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        expected = ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a) + gamma_1
        assert_allclose(sol, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_source(self, sparse):
        D = 15
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)
        grid = Grid(a=a, b=b, N=100, left_BC=left_BC, right_BC=right_BC)

        sol = solve_linear_poisson_eq(grid=grid, D=D, q=lambda x: 1, sparse=sparse)

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        x = grid.x
        expected = (
            (-1 / (2 * D)) * (x - a) * (x - b)
            + ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a)
            + gamma_1
        )
        assert_allclose(sol, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_zero_BC(self, sparse):
        left_BC = DirichletBC(0)
        right_BC = DirichletBC(0)
        grid = Grid(a=0, b=1, N=10, left_BC=left_BC, right_BC=right_BC)

        sol = solve_linear_poisson_eq(grid=grid, D=1, q=lambda x: 0, sparse=sparse)
        assert_allclose(sol, np.zeros_like(sol))

    def test_neumann(self):
        left_BC = DirichletBC(10)
        right_BC = NeumannBC(5)
        grid = Grid(a=0, b=1, N=11, left_BC=left_BC, right_BC=right_BC)
        sol = solve_linear_poisson_eq(grid=grid, D=1, q=lambda x: 0)
        x = grid.x
        assert_allclose(sol, 5 * x + 10)

        left_BC = NeumannBC(5)
        right_BC = DirichletBC(10)
        grid = Grid(a=0, b=1, N=11, left_BC=left_BC, right_BC=right_BC)
        sol = solve_linear_poisson_eq(grid=grid, D=1, q=lambda x: 0)
        x = grid.x
        assert_allclose(sol, 5 * x + 5)

    def test_robin(self):
        left_BC = DirichletBC(1)
        right_BC = RobinBC(5, -3)
        grid = Grid(a=0, b=1, N=11, left_BC=left_BC, right_BC=right_BC)
        sol = solve_linear_poisson_eq(grid=grid, D=1, q=lambda x: 0)
        x = grid.x
        assert_allclose(sol, -4 * x + 1)

        left_BC = RobinBC(5, -3)
        right_BC = DirichletBC(1)
        grid = Grid(a=0, b=1, N=11, left_BC=left_BC, right_BC=right_BC)
        sol = solve_linear_poisson_eq(grid=grid, D=1, q=lambda x: 0)
        x = grid.x
        assert_allclose(sol, 2 * x - 1, atol=1e-12)


class TestSolveNonLinearPoissonEquation:
    # Currently only has tests with linear source terms as these are easier to find
    # analytical solutions to
    @pytest.mark.parametrize("sparse", [True, False])
    def test_no_source(self, sparse):
        D = 1
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        N = 100
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)
        grid = Grid(a=a, b=b, N=N, left_BC=left_BC, right_BC=right_BC)

        sol = solve_nonlinear_poisson_eq(
            u0=np.ones((N - 2)), grid=grid, D=D, q=lambda u, x: 0, sparse=sparse
        )

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        expected = ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a) + gamma_1
        assert_allclose(sol, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_source(self, sparse):
        D = 15
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        N = 100
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)
        grid = Grid(a=a, b=b, N=N, left_BC=left_BC, right_BC=right_BC)

        sol = solve_nonlinear_poisson_eq(
            u0=np.ones((N - 2, 1)), grid=grid, D=D, q=lambda u, x: 1, sparse=sparse
        )

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        x = grid.x
        expected = (
            (-1 / (2 * D)) * (x - a) * (x - b)
            + ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a)
            + gamma_1
        )
        assert_allclose(sol, expected)
