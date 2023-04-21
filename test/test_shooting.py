import numpy as np
import pytest
from scicomp.odes import hopf_normal, modified_hopf
from scicomp.shooting import DerivativePhaseCondition, find_limit_cycle


class TestFindLimitCycle:
    def test_invalid_time_period(self):
        pc = DerivativePhaseCondition(0)
        msg = "Initial guess of period 'T' must be positive"
        with pytest.raises(ValueError, match=msg):
            find_limit_cycle(
                hopf_normal,
                y0=[1, 0],
                T=-6.28,
                phase_condition=pc,
            )

    def test_find_hopf_period(self):
        beta = 1
        rho = -1
        pc = DerivativePhaseCondition(0)
        solver_args = {"r_tol": 1e-6}
        res = find_limit_cycle(
            hopf_normal,
            y0=[1, 0],
            T=6.28,
            phase_condition=pc,
            ivp_solver_kwargs=solver_args,
            ode_params={"beta": beta, "rho": rho},
        )
        np.testing.assert_allclose(res.T, 2 * np.pi, rtol=1e-6)

    def test_find_modified_hopf_period(self):
        pc = DerivativePhaseCondition(0)

        solver_args = {"r_tol": 1e-6}
        res = find_limit_cycle(
            modified_hopf,
            y0=[1, -1],
            T=5,
            phase_condition=pc,
            ode_params={"beta": 1},
            ivp_solver_kwargs=solver_args,
        )
        np.testing.assert_allclose(res.T, 2 * np.pi, rtol=1e-6)
