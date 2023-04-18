import numpy as np
import pytest
from numpy.testing import assert_allclose
from scicomp.continuation import numerical_continuation
from scicomp.odes import modified_hopf
from scicomp.shooting import DerivativePhaseCondition, limit_cycle_shooting_func


def eq(x, c):
    return x**3 - x + c


class TestContinuation:
    def test_not_callable_equation(self):
        msg = "'equation' must be callable"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                "x^2 + c",
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
            )

    def test_param_not_in_sig(self):
        msg = "'variable_kwarg' is not a valid parameter to vary in 'equation'"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                eq,
                variable_kwarg="b",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
            )

    def test_extra_params_not_in_sig(self):
        msg = "'fixed_kwargs' are not valid inputs to 'equation'"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                fixed_kwargs={"b": 10},
            )

    def test_invalid_max_steps(self):
        msg = "'max_steps' must be positive"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=-100,
            )

    def test_invalid_discretisation(self):
        msg = "'discretisation' must be callable"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                discretisation="x",
            )

    def test_invalid_method(self):
        msg = r"cheese is not a valid method. Valid methods are: \('ps-arc', 'np'\)"
        with pytest.raises(ValueError, match=msg):
            numerical_continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                method="cheese",
            )

    def test_equation_natural(self):
        sol = numerical_continuation(
            eq,
            variable_kwarg="c",
            initial_value=-2,
            step_size=0.001,
            max_steps=4000,
            y0=[1.5],
            root_finder_kwargs={"tol": 1e-8},
            method="np",
        )

        assert sol.parameter_values[0] == -2
        assert_allclose(sol.parameter_values[-1], 0.4, atol=2e-2)
        x = np.squeeze(sol.state_values)
        assert_allclose(x**3 - x + sol.parameter_values, 0, atol=1e-12)

    def test_equation_pseudo_arc_length(self):
        sol = numerical_continuation(
            eq,
            variable_kwarg="c",
            initial_value=-2,
            step_size=0.01,
            max_steps=4000,
            y0=[1.5],
            root_finder_kwargs={"tol": 1e-8},
            method="ps-arc",
        )

        assert sol.parameter_values[0] == -2
        # check pseudo arc length was able to get around the corner
        assert sol.parameter_values[-1] > 30
        x = np.squeeze(sol.state_values)
        assert_allclose(x**3 - x + sol.parameter_values, 0, atol=1e-12)

    def test_ode_limit_cycle_continuation_pseudo_arc(self):
        res = numerical_continuation(
            equation=modified_hopf,
            variable_kwarg="beta",
            initial_value=2,
            y0=[1, 1, 6],
            step_size=-0.1,
            max_steps=50,
            discretisation=limit_cycle_shooting_func,
            discretisation_kwargs={
                "phase_condition": DerivativePhaseCondition(0),
                "ivp_solver_kwargs": {"r_tol": 1e-6},
            },
        )

        assert res.parameter_values.shape[0] == 50
        assert res.parameter_values[0] == 2
        assert res.parameter_values[22] < 0
        assert res.parameter_values[-1] > 2

    def test_ode_limit_cycle_continuation_natural_parameter(self):
        res = numerical_continuation(
            equation=modified_hopf,
            variable_kwarg="beta",
            initial_value=2,
            y0=[1, 1, 6],
            step_size=-0.1,
            max_steps=50,
            discretisation=limit_cycle_shooting_func,
            discretisation_kwargs={
                "phase_condition": DerivativePhaseCondition(0),
                "ivp_solver_kwargs": {"r_tol": 1e-6},
            },
            method="np",
        )

        assert res.parameter_values.shape[0] == 23
        assert res.parameter_values[0] == 2
        assert res.parameter_values[-1] < 0
