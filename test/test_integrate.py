import numpy as np
import pytest
from scicomp import integrate
from scicomp.odes import exponential_ode, zero_ode

ALL_METHODS = pytest.mark.parametrize("method", integrate._fixed_step_methods.keys())
FIXED_STEP_METHODS = pytest.mark.parametrize("method",
                                             integrate._fixed_step_methods.keys())


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
            integrate.solve_ivp(exponential_ode, [0], (0, 1), h=0.1,
                                method="cheddar cheese")

    @ALL_METHODS
    def test_invalid_t_span(self, method):
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(exponential_ode, [0], t_span=(1, 0), h=0.1,
                                method=method)
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(exponential_ode, [0], t_span=(0, 1, 2), h=0.1,
                                method=method)

    @ALL_METHODS
    def test_zero_ode(self, method):
        res = integrate.solve_ivp(zero_ode, np.ones(10), [0, 5], method=method, h=1e-1)
        assert np.array_equal(res.y, np.ones((10, res.t.size)))

    @FIXED_STEP_METHODS
    def test_t_span_obeyed_fixed_step(self, method):
        t_span = (2, 5.432)
        res = integrate.solve_ivp(zero_ode, np.ones(10), t_span, method=method, h=1e-1)
        np.testing.assert_allclose(t_span, (res.t[0], res.t[-1]))
    
    @FIXED_STEP_METHODS
    def test_fixed_steps_taken(self, method):
        res = integrate.solve_ivp(zero_ode, np.ones(10), (0, 5), method=method, h=1e-1)
        diff = np.diff(res.t)[:-1]
        np.testing.assert_allclose(diff, np.broadcast_to(1e-1, diff.shape))
