from scicomp.odes import zero_ode

import numpy as np

def test_zero_ode():
    rsh = zero_ode(20, np.random.random((10,1)))
    assert rsh.shape == (10,1)
    assert np.all(rsh == 0)
