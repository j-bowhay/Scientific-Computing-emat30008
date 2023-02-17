import numpy as np
from scicomp.odes import hopf_normal
from scicomp.shooting import find_limit_cycle 


class TestFindLimitCycle:
    def test_find_hopf_period(self):
        beta = 1
        rho = -1
        np.testing.assert_allclose(
            find_limit_cycle(lambda t, y: hopf_normal(t, y, beta, rho), [1, 1, 6])[-1],
            2*np.pi, rtol=1e-4)