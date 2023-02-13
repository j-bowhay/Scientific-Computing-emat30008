import numpy as np


def zero_ode(t: float, y: np.ndarray) -> np.ndarray:
    """An ODE RHS that always returns zeros.
    
    ``y' = 0``

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


def exponential_ode(t: float, y: np.ndarray) -> np.ndarray:
    """An ode rsh which is equal to the current state.
    
    `` y' = y``

    Parameters
    ----------
    t : float
        Time to evaluate at.
    y : np.ndarray
        State to evaluate at.

    Returns
    -------
    np.ndarray
        Returns `y`
    """
    return y
