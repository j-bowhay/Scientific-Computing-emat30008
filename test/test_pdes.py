import numpy as np
from numpy.testing import assert_allclose
import pytest

from scicomp.pdes import solve_linear_diffusion_crank_nicolson
from scicomp.finite_diff import Grid, DirichletBC


class TestSolveLinearDiffusionCrankNicolson:
    def test_invalid_dt(self):
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)

        msg = "Invalid 'dt'"
        with pytest.raises(ValueError, match=msg):
            u = solve_linear_diffusion_crank_nicolson(
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
            u = solve_linear_diffusion_crank_nicolson(
                grid=grid,
                D=0.1,
                dt=0.01,
                steps=-1000,
                u0_func=lambda x: np.sin(np.pi * x),
            )

    def test_sol_dirichlet(self):
        left_BC = right_BC = DirichletBC(0)
        grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)
        u = solve_linear_diffusion_crank_nicolson(
            grid=grid, D=0.1, dt=0.01, steps=200, u0_func=lambda x: np.sin(np.pi * x)
        )

        assert_allclose(u[-1, 50], np.exp(-0.2 * np.pi**2), rtol=1e-4)
