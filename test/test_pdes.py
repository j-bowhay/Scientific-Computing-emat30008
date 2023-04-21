import numpy as np
import pytest
from numpy.testing import assert_allclose
from scicomp.finite_diff import DirichletBC, Grid
from scicomp.pdes import solve_diffusion_implicit, solve_diffusion_method_lines


class TestSolveLinearDiffusionImplicit:
    def test_invalid_dt(self):
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)

        msg = "Invalid 'dt'"
        with pytest.raises(ValueError, match=msg):
            solve_diffusion_implicit(
                grid=grid,
                D=0.1,
                dt=-0.01,
                steps=1000,
                u0_func=lambda x: np.sin(np.pi * x),
            )

    def test_invalid_steps(self):
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)

        msg = "Invalid 'steps'"
        with pytest.raises(ValueError, match=msg):
            solve_diffusion_implicit(
                grid=grid,
                D=0.1,
                dt=0.01,
                steps=-1000,
                u0_func=lambda x: np.sin(np.pi * x),
            )

    @pytest.mark.parametrize("method", ["euler", "crank-nicolson"])
    @pytest.mark.parametrize("sparse", ["True", "False"])
    def test_sol_dirichlet(self, method, sparse):
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)
        sol = solve_diffusion_implicit(
            grid=grid,
            D=0.1,
            dt=0.0001,
            steps=20000,
            u0_func=lambda x: np.sin(np.pi * x),
            method=method,
            sparse=sparse,
        )

        assert_allclose(sol.u[-1, 50], np.exp(-0.2 * np.pi**2), atol=1e-4)


class TestSolveDiffusionMethodLines:
    @pytest.mark.parametrize("sparse", ["True", "False"])
    def test_sol_dirichlet(self, sparse):
        D = 1
        a = 0
        b = 1
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(a, b, 30, left_BC, right_BC)

        sol = solve_diffusion_method_lines(
            grid, D, lambda x: np.sin(np.pi * x), t_span=(0, 0.1), sparse=sparse
        )

        expected = np.exp(-D * np.pi**2 * sol.t[-1]) * np.sin(np.pi * grid.x)
        assert_allclose(sol.u[-1, :], expected, atol=1e-3)
