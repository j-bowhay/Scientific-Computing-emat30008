import numpy as np
import pytest
from scicomp.odes import hopf_normal
from scicomp.shooting import DerivativePhaseCondition, find_limit_cycle


class TestFindLimitCycle:
    def test_invalid_time_period(self):
        beta = 1
        rho = -1
        pc = DerivativePhaseCondition(0)
        solver_args = {"method": "rkf45", "r_tol": 1e-6}
        msg = "Initial guess of period 'T' must be positive"
        with pytest.raises(ValueError, match=msg):
            find_limit_cycle(
                lambda t, y: hopf_normal(t, y, beta, rho),
                y0=[1, 0],
                T=-6.28,
                phase_condition=pc,
                ivp_solver_kwargs=solver_args,
            )

    def test_find_hopf_period(self):
        beta = 1
        rho = -1
        pc = DerivativePhaseCondition(0)
        solver_args = {"method": "rkf45", "r_tol": 1e-6}
        res = find_limit_cycle(
            lambda t, y: hopf_normal(t, y, beta, rho),
            y0=[1, 0],
            T=6.28,
            phase_condition=pc,
            ivp_solver_kwargs=solver_args,
        )
        np.testing.assert_allclose(res.T, 2 * np.pi, rtol=1e-6)
