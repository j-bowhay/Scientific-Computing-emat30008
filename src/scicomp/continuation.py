from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import inspect

import numpy as np
import numpy.typing as npt
from scipy.optimize import root


@dataclass
class ContinuationResult:
    state_values: np.ndarray
    parameter_values: np.ndarray


_valid_methods = ["ps-arc", "np"]


def continuation(
    equation: Callable,
    *,
    variable_kwarg: str,
    y0: npt.ArrayLike,
    step_size: float,
    max_steps: int,
    fixed_kwargs: Optional[dict] = None,
    discretisation: Callable = lambda x: x,
    method: str = "ps-arc",
) -> ContinuationResult:
    fixed_kwargs = {} if fixed_kwargs is None else fixed_kwargs

    if not callable(equation):
        raise ValueError("'equation' must be callable")
    else:
        equation_sig = set(inspect.signature(equation).parameters)

        if variable_kwarg not in equation_sig:
            raise ValueError(
                "'variable_kwarg' is not a valid parameter to vary in 'equation'"
            )
        elif set(fixed_kwargs.keys()) <= equation_sig:
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
