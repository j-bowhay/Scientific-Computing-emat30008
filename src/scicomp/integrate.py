"""
TODO List:
- Adaptive stepper
- Dense Interpolation
- Richardson Extrapolation
- Embedded Runge-Kutta Formula
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

def euler_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h*f(t,y)

def rk4_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t,y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    
    return y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)*h

@dataclass
class ODEResult:
    y: np.ndarray
    t: np.ndarray

def solve_to(f: callable, y0: np.ndarray, tspan: tuple[float, float], h: float, method: callable) -> ODEResult:
    t = [tspan[0]]
    y = [np.asarray(y0)]
    
    while t[-1] < tspan[-1]:
        t.append(t[-1] + h)
        y.append(method(f, t[-1], y[-1], h))
    
    return ODEResult(np.asarray(y), np.asarray(t))

if __name__ == "__main__":
    ode1 = lambda t, y: y
    res = solve_to(ode1, [1], (0, 1), 1e-3, euler_step)
    plt.plot(res.t, res.y)
    res = solve_to(ode1, [1], (0, 1), 1e-3, rk4_step)
    plt.plot(res.t, res.y)
    plt.show()