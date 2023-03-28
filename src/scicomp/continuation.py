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
    state_values: np.ndarray
    parameter_values: np.ndarray


_valid_methods = ["ps-arc", "np"]


def continuation(
    equation: Callable,
    *,
    variable_kwarg: str,
    initial_value: float,
    y0: npt.ArrayLike,
    step_size: float,
    max_steps: int,
    fixed_kwargs: Optional[dict] = None,
    discretisation: Callable = lambda x: x,
    method: str = "ps-arc",
    root_finder_kwargs: Optional[dict] = None,
) -> ContinuationResult:
    fixed_kwargs = {} if fixed_kwargs is None else fixed_kwargs
    root_finder_kwargs = {} if root_finder_kwargs is None else root_finder_kwargs

    if not callable(equation):
        raise ValueError("'equation' must be callable")
    else:
        equation_sig = set(inspect.signature(equation).parameters)

        if variable_kwarg not in equation_sig:
            raise ValueError(
                "'variable_kwarg' is not a valid parameter to vary in 'equation'"
            )
        elif len(fixed_kwargs) > 0 and set(fixed_kwargs.keys()) <= equation_sig:
            raise ValueError("'fixed_kwargs' are not valid inputs to 'equation'")

    if step_size <= 0:
        raise ValueError("'step_size' must be positive")

    if max_steps <= 0:
        raise ValueError("'max_steps' must be positive")

    if not callable(discretisation):
        raise ValueError("'discretisation' must be callable")

    if method not in _valid_methods:
        raise ValueError(
            f"{method} is not a valid method. Valid methods are: {*_valid_methods,}"
        )

    initial_sol = root(
        lambda x: discretisation(
            equation(x, **{variable_kwarg: initial_value}, **fixed_kwargs)
        ),
        x0=[y0],
        **root_finder_kwargs,
    )
    if initial_sol.success:
        augmented_param = [np.array([initial_value, *initial_sol.x])]
    else:
        raise ContinuationError("Bad initial guess; failed to converge")

    if method == "np":
        steps = max_steps
    elif method == "ps-arc":
        # need to do an iteration of natural parameter continuation to find initial secant
        steps = 1

    # natural parameter continuation
    for _ in range(steps):
        param = augmented_param[-1][0] + step_size
        sol = root(
            lambda x: discretisation(
                equation(x, **{variable_kwarg: param}, **fixed_kwargs)
            ),
            x0=[augmented_param[-1][1:]],
            **root_finder_kwargs,
        )
        if sol.success:
            augmented_param.append(np.array([param, *sol.x]))
        else:
            break

    if method == "ps-arc":
        for _ in range(max_steps):
            secant = augmented_param[-1] - augmented_param[-2]
            predicted = augmented_param[-1] + secant

            sol = root(
                lambda x: [
                    *discretisation(
                        equation(x[1:], **{variable_kwarg: x[0]}, **fixed_kwargs)
                    ),
                    np.dot(secant, x - predicted),
                ],
                x0=predicted,
                **root_finder_kwargs,
            )

            if sol.success:
                augmented_param.append(sol.x)
            else:
                break

    augmented_param = np.asarray(augmented_param)
    return ContinuationResult(augmented_param[:, 1:], augmented_param[:, 0])
