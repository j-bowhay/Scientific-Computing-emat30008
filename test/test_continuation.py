import numpy as np
import pytest

from scicomp.continuation import continuation


def eq(x, c):
    return x**3 - x + c


class TestContinuation:
    def test_not_callable_equation(self):
        msg = "'equation' must be callable"
        with pytest.raises(ValueError, match=msg):
            continuation(
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
            continuation(
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
            continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                fixed_kwargs={"b": 10},
            )

    def test_invalid_step_size(self):
        msg = "'step_size' must be positive"
        with pytest.raises(ValueError, match=msg):
            continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=-0.1,
                max_steps=100,
            )

    def test_invalid_max_steps(self):
        msg = "'max_steps' must be positive"
        with pytest.raises(ValueError, match=msg):
            continuation(
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
            continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                discretisation="x",
            )

    def test_invalid_method(self):
        msg = "cheese is not a valid method. Valid methods are: \('ps-arc', 'np'\)"
        with pytest.raises(ValueError, match=msg):
            continuation(
                eq,
                variable_kwarg="c",
                initial_value=0,
                y0=0,
                step_size=0.1,
                max_steps=100,
                method="cheese",
            )
