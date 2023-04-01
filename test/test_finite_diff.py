import numpy as np
from numpy.testing import assert_allclose
import pytest

from scicomp.finite_diff import get_central_diff_matrix


class TestGetCentralDiffMatrix:
    def test_first_derive_3_point(self):
        A = get_central_diff_matrix(3, derivative=1)
        expected = np.array([[0, 0.5, 0], [-0.5, 0, 0.5], [0, -0.5, 0]])
        assert_allclose(A, expected)

        A = get_central_diff_matrix(4, derivative=1)
        expected = np.array(
            [[0, 0.5, 0, 0], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5], [0, 0, -0.5, 0]]
        )
        assert_allclose(A, expected)

    def test_second_derive_3_point(self):
        A = get_central_diff_matrix(3, derivative=2)
        expected = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
        assert_allclose(A, expected)

        A = get_central_diff_matrix(4, derivative=2)
        expected = np.array(
            [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
        )
        assert_allclose(A, expected)

    def test_invalid_size(self):
        msg = "Matrix must be at least 5*5"
        with pytest.raises(ValueError, match=msg):
            get_central_diff_matrix(2, derivative=2, points=5)
