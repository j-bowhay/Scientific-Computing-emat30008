from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
        Time
    y : np.ndarray
        Current state

    Returns
    -------
    np.ndarray
        Returns `y`
    """
    return y


def shm_ode(t: float, y: np.ndarray, omega: float) -> npt.ArrayLike:
    """An ODE RSH for simple harmonic motion with period `omega`.

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        Current state
    omega : float
        Frequency

    Returns
    -------
    np.ndarray
        RHS of the ODE
    """
    return [y[1], -(omega**2) * y[0]]


def hopf_normal(t: float, y: np.ndarray, beta: float, rho: float) -> npt.ArrayLike:
    r"""RHS function for the hopf normal form.
    
    ..math::

        \newcommand{\diff}[2]{\frac{\mathrm{d}#1}{\mathrm{d}#2}}
        \begin{align}
        \diff{u_1}{t} &= \beta u_1 - u_2 + \rho u_1\left(u_1^2 + u_2^2\right),\\
        \diff{u_2}{t} &= u_1 + \beta u_2 + \rho u_2\left(u_1^2 + u_2^2\right),
        \end{align}

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        Current state
    beta : float
        magnitude parameter
    rho : float
        criticality parameter

    Returns
    -------
    npt.ArrayLike
        RHS of the ODE
    """
    return [
        beta * y[0] - y[1] + rho * y[0] * (y[0] ** 2 + y[1] ** 2),
        y[0] + beta * y[1] + rho * y[1] * (y[0] ** 2 + y[1] ** 2),
    ]


def hopf_3D(t: float, y: np.ndarray, beta: float, rho: float) -> npt.ArrayLike:
    r"""RHS function for a 3D hopf bifurcation.
    
    ..math::

        \newcommand{\diff}[2]{\frac{\mathrm{d}#1}{\mathrm{d}#2}}
        \begin{align}
        \diff{u_1}{t} &= \beta u_1 - u_2 + \sigma u_1\left(u_1^2 + u_2^2\right)\\
        \diff{u_2}{t} &= u_1 + \beta u_2 + \sigma u_2\left(u_1^2 + u_2^2\right)\\
        \diff{u_3}{t} &= -u_3
        \end{align}

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        Current state
    beta : float
        magnitude parameter
    rho : float
        criticality parameter

    Returns
    -------
    npt.ArrayLike
        RHS of the ODE
    """
    return [
        beta * y[0] - y[1] + rho * y[0] * (y[0] ** 2 + y[1] ** 2),
        y[0] + beta * y[1] + rho * y[1] * (y[0] ** 2 + y[1] ** 2),
        -y[2],
    ]


def predator_prey(
    t: float, y: np.ndarray, a: float, b: float, d: float
) -> npt.ArrayLike:
    r"""RHS function for the predator prey equation
    
    .. math::
    
        \begin{aligned}
        \frac{\text{d}x}{\text{d}t} &= x(1-x) - \frac{axy}{d+x}\\
        \frac{\text{d}y}{\text{d}t} &= by\left(1-\frac{y}{x}\right)
        \end{aligned}

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        Current state
    a : float
        ODE parameter
    b : float
        ODE parameter
    d : float
       ODE parameter

    Returns
    -------
    npt.ArrayLike
        RHS of ODE
    """
    return [
        y[0] * (1 - y[0]) - (a * y[0] * y[1]) / (d + y[0]),
        b * y[1] * (1 - y[1] / y[0]),
    ]


def modified_hopf(t, y, beta):
    r"""Ode for the modified hopf normal form

    .. math::

        \begin{align}
        \diff{u_1}{t} &= \beta u_1 - u_2 + u_1\left(u_1^2 + u_2^2\right)
        - u_1\left(u_1^2 + u_2^2\right)^2,\\
        \diff{u_2}{t} &= u_1 + \beta u_2 + u_2\left(u_1^2 + u_2^2\right)
        - u_2\left(u_1^2 + u_2^2\right)^2
        \end{align}

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        Current state
    beta : float
        magnitude parameter

    Returns
    -------
    npt.ArrayLike
        RHS of the ODE
    """
    return [
        beta * y[0]
        - y[1]
        + y[0] * (y[0] ** 2 + y[1] ** 2)
        - y[0] * (y[0] ** 2 + y[1] ** 2) ** 2,
        y[0]
        + beta * y[1]
        + y[1] * (y[0] ** 2 + y[1] ** 2)
        - y[1] * (y[0] ** 2 + y[1] ** 2) ** 2,
    ]
