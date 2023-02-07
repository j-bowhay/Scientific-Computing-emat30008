from scicomp import integrate

import pytest


def dummy_ode(t, y):
    return y

class TestSolveOde:
    def test_invalid_signature(self):
        with pytest.raises(ValueError, match="'f' must be callable"):
            integrate.solve_ode("ode", [0], [0, 1], "euler", 0.1)
        with pytest.raises(ValueError, match="'f' has an invalid signature"):
            integrate.solve_ode(lambda t, y, z: y, [0], [0, 1], "euler", 0.1)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="cheddar cheese is not a valid option for 'method'"):
            integrate.solve_ode(dummy_ode, [0], (0, 1), h=0.1, method="cheddar cheese")

    def test_invalid_t_span(self):
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ode(dummy_ode, [0], t_span=(1, 0), h=0.1, method="euler")
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ode(dummy_ode, [0], t_span=(0, 1, 2), h=0.1, method="euler")
