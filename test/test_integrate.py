import numpy as np
import pytest
from scicomp import integrate
from scicomp.odes import exponential_ode, shm_ode, zero_ode

ALL_METHODS = pytest.mark.parametrize("method", integrate._all_methods.keys())
FIXED_STEP_METHODS = pytest.mark.parametrize(
    "method", integrate._fixed_step_methods.keys()
)
EMBEDDED_METHODS = pytest.mark.parametrize("method", integrate._embedded_methods.keys())


class TestSolveOde:
    def test_invalid_signature(self):
        with pytest.raises(ValueError, match="'f' must be callable"):
            integrate.solve_ivp("ode", y0=[0], t_span=[0, 1], method="rk4", h=0.1)
        with pytest.raises(ValueError, match="'f' has an invalid signature"):
            integrate.solve_ivp(
                lambda t, y, z: y, y0=[0], t_span=[0, 1], method="rk4", h=0.1
            )

    def test_invalid_method(self):
        with pytest.raises(
            ValueError, match="cheddar cheese is not a valid option for 'method'"
        ):
            integrate.solve_ivp(
                exponential_ode, y0=[0], t_span=(0, 1), h=0.1, method="cheddar cheese"
            )

    def test_invalid_mode(self):
        with pytest.raises(
            ValueError, match="cheddar cheese is not a valid option for 'mode'"
        ):
            integrate.solve_ivp(
                exponential_ode, y0=[0], t_span=(0, 1), h=0.1, mode="cheddar cheese"
            )

    def test_invalid_t_span(self):
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(
                exponential_ode, y0=[0], t_span=(1, 0), h=0.1, method="rk4"
            )
        with pytest.raises(ValueError, match="Invalid values for 't_span'"):
            integrate.solve_ivp(
                exponential_ode, y0=[0], t_span=(0, 1, 2), h=0.1, method="rk4"
            )

    def test_invalid_ics(self):
        with pytest.raises(
            ValueError, match="Initial conditions must be 1 dimensional."
        ):
            integrate.solve_ivp(
                exponential_ode,
                y0=np.array([[1, 1], [1, 2]]),
                t_span=(0, 1),
                method="rk4",
                h=1,
            )

    def test_not_step_size_fixed(self):
        with pytest.raises(ValueError, match="size must be provided"):
            integrate.solve_ivp(
                exponential_ode, y0=[1], t_span=[0, 1], method="rk4", mode="fixed"
            )

    def test_negative_arg(self):
        with pytest.raises(ValueError, match="Invalid negative option."):
            integrate.solve_ivp(
                exponential_ode, y0=[1], t_span=[0, 1], method="rk4", h=-1
            )
        with pytest.raises(ValueError, match="Invalid negative option."):
            integrate.solve_ivp(
                exponential_ode, y0=[1], t_span=[0, 1], method="rk4", r_tol=-1
            )
        with pytest.raises(ValueError, match="Invalid negative option."):
            integrate.solve_ivp(
                exponential_ode, y0=[1], t_span=[0, 1], method="rk4", a_tol=-1
            )
        with pytest.raises(ValueError, match="Invalid negative option."):
            integrate.solve_ivp(
                exponential_ode,
                y0=[1],
                t_span=[0, 1],
                method="rk4",
                r_tol=0.1,
                max_step=-1,
            )

    @ALL_METHODS
    def test_zero_ode(self, method):
        res = integrate.solve_ivp(
            zero_ode, y0=np.ones(10), t_span=[0, 5], method=method, mode="fixed", h=1e-1
        )
        assert np.array_equal(res.y, np.ones((10, res.t.size)))

    @ALL_METHODS
    def test_zero_adaptive(self, method):
        res = integrate.solve_ivp(
            zero_ode,
            y0=np.ones(10),
            t_span=[0, 5],
            method=method,
            r_tol=1e-3,
            a_tol=1e-3,
        )
        assert np.array_equal(res.y, np.ones((10, res.t.size)))

    def test_t_span_obeyed_fixed_step(self):
        t_span = (2, 5.432)
        res = integrate.solve_ivp(
            zero_ode, y0=np.ones(10), t_span=t_span, method="rk4", mode="fixed", h=1e-1
        )
        np.testing.assert_allclose(t_span, (res.t[0], res.t[-1]))

    def test_t_span_obeyed_adaptive(self):
        t_span = (2, 5.432)
        res = integrate.solve_ivp(
            lambda t, y: shm_ode(t, y, 1),
            y0=[1, 0],
            t_span=t_span,
            method="rkf45",
            h=1e-1,
            r_tol=1e-2,
        )
        np.testing.assert_allclose(t_span, (res.t[0], res.t[-1]))

    def test_fixed_steps_taken(self):
        res = integrate.solve_ivp(
            zero_ode, y0=np.ones(10), t_span=(0, 5), method="rk4", mode="fixed", h=1e-1
        )
        diff = np.diff(res.t)[:-1]
        np.testing.assert_allclose(diff, np.broadcast_to(1e-1, diff.shape))

    @ALL_METHODS
    def test_fixed_step_shm(self, method):
        res = integrate.solve_ivp(
            lambda t, y: shm_ode(t, y, 1),
            y0=[1, 0],
            t_span=(0, 0.1),
            method=method,
            mode="fixed",
            h=1e-6,
        )
        np.testing.assert_allclose(res.y[:, -1], [np.cos(0.1), -np.sin(0.1)], rtol=1e-6)

    @FIXED_STEP_METHODS
    def test_richardson_adaptive_shm(self, method):
        res = integrate.solve_ivp(
            lambda t, y: shm_ode(t, y, 1),
            y0=[1, 0.5],
            t_span=(0, 0.5),
            method=method,
        )
        if method == "euler":
            tol = 1e-2
        else:
            tol = 1e-3
        np.testing.assert_allclose(
            res.y[:, -1],
            [np.cos(0.5) + 0.5 * np.sin(0.5), -np.sin(0.5) + 0.5 * np.cos(0.5)],
            atol=tol,
        )

    @EMBEDDED_METHODS
    def test_adaptive_shm(self, method):
        res = integrate.solve_ivp(
            lambda t, y: shm_ode(t, y, 1),
            y0=[1, 0.5],
            t_span=(0, 0.5),
            method=method,
        )
        np.testing.assert_allclose(
            res.y[:, -1],
            [np.cos(0.5) + 0.5 * np.sin(0.5), -np.sin(0.5) + 0.5 * np.cos(0.5)],
            atol=1e-3,
        )
