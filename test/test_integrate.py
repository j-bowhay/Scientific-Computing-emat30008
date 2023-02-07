from scicomp import integrate

import pytest

class TestSolveOde:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="cheddar cheese is not a valid option for 'method'"):
            integrate.solve_ode(lambda t, y: y, [0], (0, 1), 0.1, method="cheddar cheese")