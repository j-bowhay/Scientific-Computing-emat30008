import numpy as np
from scicomp.odes import exponential_ode, shm_ode, zero_ode


def test_zero_ode():
    rhs = zero_ode(np.nan, np.random.random((10, 1)))
    assert rhs.shape == (10, 1)
    assert np.all(rhs == 0)


def test_exponential_ode():
    y = np.random.random((10, 1))
    rhs = exponential_ode(np.nan, y)
    assert rhs.shape == (10, 1)
    np.testing.assert_array_equal(rhs, y)


def test_shm_ode():
    y = np.random.random((10, 1))
    rhs = shm_ode(np.nan, y, 5)
    np.testing.assert_array_equal(rhs, [y[1], -25 * y[0]])
