from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import root


class ContinuationError(Exception):
    ...


@dataclass
class ContinuationResult:
    """Results object containing output of numerical continuation.
    
    Has the following attributes:
        state_value : np.ndarray
            Value of the state vector at each corresponding value of `parameter_values`
        parameter_values : np.ndarray
            Value of the parameter that was varied.
    """
    state_values: np.ndarray
    parameter_values: np.ndarray


_valid_methods = ["ps-arc", "np"]


def numerical_continuation(
    equation: Callable[[np.ndarray], np.ndarray],
    *,
    variable_kwarg: str,
    initial_value: float,
    y0: npt.ArrayLike,
    step_size: float,
    max_steps: int,
    fixed_kwargs: Optional[dict] = None,
    discretisation: Callable[
        [np.ndarray, Callable], np.ndarray
    ] = lambda x, func, **kwargs: func(x, **kwargs),
    method: str = "ps-arc",
    root_finder_kwargs: Optional[dict] = None,
    discretisation_kwargs: Optional[dict] = None,
) -> ContinuationResult:
    """Numerical continuation function for tracking roots, steady states and
    limit cycles.

    Parameters
    ----------
    equation : Callable
        Equation to track roots/limit cycles of. Must have signature
        ``equation(y, **kwargs)`` where `x` is the state vector and any parameters are
        passed as keyword arguments.
    variable_kwarg : str
        Name of the keyword argument corresponding the parameter that is meant to be
        varied.
    initial_value : float
        Initial value of the parameter to be varied.
    y0 : npt.ArrayLike
        Initial guess at the state vector correspond to the initial value of the
        parameter. Note for limit cycles this is augmented with period of the limit
        cycle as the last element.
    step_size : float
        Step size to be taken when varying of the parameter. Note for pseudo arc length
        continuation this is only the initial step size.
    max_steps : int
        Maximum number of steps to take in varying the parameter.
    fixed_kwargs : dict, optional
        Additional extra parameters to be passed to the equation that are not varied,
        by default None.
    discretisation : Callable, optional
        Optional discretisation to be applied to the equation, for example to find limit
        cycles. Must have the signature
        ``discretisation(y, equation, **equation parameters)``. By default no
        discretisation is applied.
    method : str, optional
        Method of numerical continuation to use. Either "ps-arc" for pseudo arc
        length continuation or "np" for natural parameter continuation,
        by default "ps-arc"
    root_finder_kwargs : dict, optional
        Optional arguments that are passed to the root finder, by default None.
    discretisation_kwargs : dict, optional
        _description_, by default None

    Returns
    -------
    ContinuationResult
        Result object with the following attributes:
            state_value : np.ndarray
                Value of the state vector at each corresponding value of `parameter_values`
            parameter_values : np.ndarray


    Raises
    ------
    ContinuationError
        Raised if initial value of `y0` is bad and the root finder cannot converge.
    """
    fixed_kwargs = {} if fixed_kwargs is None else fixed_kwargs
    root_finder_kwargs = {} if root_finder_kwargs is None else root_finder_kwargs
    discretisation_kwargs = (
        {} if discretisation_kwargs is None else discretisation_kwargs
    )

    # Input validation
    if not callable(equation):
        raise ValueError("'equation' must be callable")
    else:
        equation_sig = set(inspect.signature(equation).parameters)

        if variable_kwarg not in equation_sig:
            raise ValueError(
                "'variable_kwarg' is not a valid parameter to vary in 'equation'"
            )
        elif len(fixed_kwargs) > 0 and not set(fixed_kwargs.keys()) <= equation_sig:
            raise ValueError("'fixed_kwargs' are not valid inputs to 'equation'")

    if max_steps <= 0:
        raise ValueError("'max_steps' must be positive")

    if not callable(discretisation):
        raise ValueError("'discretisation' must be callable")

    if method not in _valid_methods:
        raise ValueError(
            f"{method} is not a valid method. Valid methods are: {*_valid_methods,}"
        )

    # Initial starting point
    initial_sol = root(
        lambda x: discretisation(
            x,
            equation,
            **{variable_kwarg: initial_value},
            **fixed_kwargs,
            **discretisation_kwargs,
        ),
        x0=[y0],
        **root_finder_kwargs,
    )

    if initial_sol.success:
        augmented_param = [np.array([initial_value, *initial_sol.x])]
    else:
        raise ContinuationError("Bad initial guess; failed to converge")

    if method == "np":
        steps = max_steps - 1  # first step already taken
    elif method == "ps-arc":
        # need to do an iteration of natural parameter continuation
        # to find initial secant
        steps = 1

    # natural parameter continuation
    for _ in range(steps):
        param = augmented_param[-1][0] + step_size
        sol = root(
            lambda x: discretisation(
                x,
                equation,
                **{variable_kwarg: param},
                **fixed_kwargs,
                **discretisation_kwargs,
            ),
            x0=[augmented_param[-1][1:]],
            **root_finder_kwargs,
        )
        if sol.success:
            augmented_param.append(np.array([param, *sol.x]))
        else:
            break

    if method == "ps-arc":
        for _ in range(max_steps - 2):
            secant = augmented_param[-1] - augmented_param[-2]
            predicted = augmented_param[-1] + secant

            sol = root(
                lambda x: [
                    *discretisation(
                        x[1:],
                        equation,
                        **{variable_kwarg: x[0]},
                        **fixed_kwargs,
                        **discretisation_kwargs,
                    ),
                    np.dot(secant, x - predicted),
                ],
                x0=[augmented_param[-1]],
                **root_finder_kwargs,
            )

            if sol.success:
                augmented_param.append(sol.x)
            else:
                break

    augmented_param_array = np.asarray(augmented_param)
    return ContinuationResult(
        augmented_param_array[:, 1:].T, augmented_param_array[:, 0]
    )
