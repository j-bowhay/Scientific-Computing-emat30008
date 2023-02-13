import numpy as np

def zero_ode(t: float, y: np.ndarray) -> np.ndarray:
    """An ODE RHS that always returns zeros.

    Parameters
    ----------
    t : float
        Time to evaluate at.
    y : np.ndarray
        State to evaluate at.

    Returns
    -------
    np.ndarray
        Array of zeros that is the same shape as `y`.
    """
    return np.zeros_like(y)