from scicomp import integrate
from scicomp.odes import zero_ode, exponential_ode

import pytest
import numpy as np

ALL_METHODS = pytest.mark.parametrize("method", integrate._fixed_step_methods.keys())


class TestSolveOde:
    @ALL_METHODS
    def test_invalid_signature(self, method):
        with pytest.raises(ValueError, match="'f' must be callable"):
            integrate.solve_ivp("ode", [0], [0, 1], method, 0.1)
        with pytest.raises(ValueError, match="'f' has an invalid signature"):
            integrate.solve_ivp(lambda t, y, z: y, [0], [0, 1], method, 0.1)

    def test_invalid_method(self):
        with pytest.raises(ValueError,
                           match="cheddar cheese is not a valid option for 'method'"):
            integrate.solve_ivp(exponential_ode, [0], (0, 1), h=0.1, method="cheddar cheese")

    @ALL_METHODS
    def test_invalid_t_span(self, method):
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(exponential_ode, [0], t_span=(1, 0), h=0.1, method=method)
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(exponential_ode, [0], t_span=(0, 1, 2), h=0.1, method=method)

    @ALL_METHODS
    def test_zero_ode(self, method):
        res = integrate.solve_ivp(zero_ode, np.ones(10), [0, 5], method=method, h=1e-1)
        assert np.array_equal(res.y, np.ones((10, res.t.size)))
