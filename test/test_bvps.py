import numpy as np
from numpy.testing import assert_allclose
from scicomp.bvps import solve_linear_poisson_eq, solve_nonlinear_poisson_eq
from scicomp.finite_diff import DirichletBC, Grid


class TestSolveLinearPoissonEquation:
    def test_no_source(self):
        D = 1
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        grid = Grid(a=a, b=b, N=100)
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)

        sol = solve_linear_poisson_eq(
            grid=grid, left_BC=left_BC, right_BC=right_BC, D=D, q=lambda x: 0
        )

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        expected = ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a) + gamma_1
        assert_allclose(sol, expected)

    def test_source(self):
        D = 15
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        grid = Grid(a=a, b=b, N=100)
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)

        sol = solve_linear_poisson_eq(
            grid=grid, left_BC=left_BC, right_BC=right_BC, D=D, q=lambda x: 1
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


class TestSolveNonLinearPoissonEquation:
    # Currently only has tests with linear source terms as these are easier to find
    # analytical solutions to
    def test_no_source(self):
        D = 1
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        N = 100
        grid = Grid(a=a, b=b, N=N)
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)

        sol = solve_nonlinear_poisson_eq(
            u0=np.ones((N - 2)),
            grid=grid,
            left_BC=left_BC,
            right_BC=right_BC,
            D=D,
            q=lambda u, x: 0,
        )

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        expected = ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a) + gamma_1
        assert_allclose(sol, expected)

    def test_source(self):
        D = 15
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        N = 100
        grid = Grid(a=a, b=b, N=N)
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)

        sol = solve_nonlinear_poisson_eq(
            u0=np.ones((N - 2, 1)),
            grid=grid,
            left_BC=left_BC,
            right_BC=right_BC,
            D=D,
            q=lambda u, x: 1,
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
