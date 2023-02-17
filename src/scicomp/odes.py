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
    """An ODE RHS which is equal to the current state.
    
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

def shm_ode(t: float, y: np.ndarray, omega: float) -> np.ndarray:
    """An ODE RSH for simple harmonic motion with period `omega`.

    Parameters
    ----------
    t : float
        _description_
    y : np.ndarray
        _description_
    omega : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    return [y[1], -omega**2 * y[0]]

def hopf_normal(t: float, y: np.ndarray, beta: float, rho: float) -> np.ndarray:
    return [beta*y[0] - y[1] + rho*y[0]*(y[0]**2 + y[1]**2),
            y[0] + beta*y[1] + rho*y[1]*(y[0]**2 + y[1]**2)]

def predator_prey(t: float, y: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return [y[0] * (1 - y[0]) - (a * y[0] * y[1])/(d + y[0]),
            b*y[1]*(1 - y[1]/y[0])]
