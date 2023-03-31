import numpy as np
import scipy
from scipy._lib._finite_differences import _central_diff_weights


def get_central_diff_matrix(
    n: int, *, derivative: int, points: int = 3, sparse: bool = False
) -> np.ndarray | scipy.sparse.dia_matrix:
    """Generates a central difference matrix of arbitrary accuracy

    Parameters
    ----------
    n : int
        Size of the matrix (matrix will be ``n*n``)
    derivative : int
        The order of the derivate to calculate
    points : int, optional
        The number of points to use in the central difference stencil, by default 3
    sparse : bool, optional
        Return a sparse matrix instead of an array, by default False

    Returns
    -------
    np.ndarray | scipy.sparse.dia_matrix
        The central difference matrix
    """
    weights = _central_diff_weights(points, derivative)

    k = int(0.5 * (points - 1))
    offsets = list(range(-k, k + 1))
    diags = [weights[i] * np.ones(n - abs(offset)) for i, offset in enumerate(offsets)]

    A = scipy.sparse.diags(diags, offsets)

    return A if sparse else A.toarray()
