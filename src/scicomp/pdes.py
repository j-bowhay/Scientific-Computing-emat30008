from __future__ import annotations

from typing import Callable

import numpy as np
import scipy

from scicomp.finite_diff import Grid, get_A_mat_from_BCs, get_b_vec_from_BCs


def solve_linear_diffusion_method_lines():
    ...


def solve_linear_diffusion_crank_nicolson(
    grid: Grid,
    D: float,
    dt: float,
    steps: int,
    u0_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("Invalid 'dt'")
    if steps <= 0:
        raise ValueError("Invalid 'steps'")
    
    C = (dt * D)/(grid.dx**2)
    
    # preallocate and apply ICs
    u = np.empty((steps, grid.N_inner))
    u[0, :] = u0_func(grid.x_inner)
    
    A = get_A_mat_from_BCs(2, grid=grid)
    b = get_b_vec_from_BCs(grid)
    
    lhs = (np.eye(*A.shape) - 0.5*C*A)
    for i in range(1, steps):
        rhs = (np.eye(*A.shape) + 0.5*C*A) @ u[i-1, :] + C*b
        u[i, :] = scipy.linalg.solve(lhs, rhs)
    
    return u
