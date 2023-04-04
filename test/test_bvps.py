from numpy.testing import assert_allclose

from scicomp.finite_diff import Grid, DirichletBC
from scicomp.bvps import solve_linear_poisson_eq


class TestSolveLinearPoissonEquation:
    def test_no_source(self):
        a = 0
        b = 10
        gamma_1 = 2
        gamma_2 = 30

        grid = Grid(a=a, b=b, N=100)
        left_BC = DirichletBC(gamma_1)
        right_BC = DirichletBC(gamma_2)

        sol = solve_linear_poisson_eq(
            grid=grid, left_BC=left_BC, right_BC=right_BC, D=1, q=lambda x: 0
        )

        assert sol.size == 100

        # check BCs
        assert sol[0] == gamma_1
        assert sol[-1] == gamma_2

        # compare to analytical solution
        expected = ((gamma_2 - gamma_1) / (b - a)) * (grid.x - a) + gamma_1
        assert_allclose(sol, expected)
