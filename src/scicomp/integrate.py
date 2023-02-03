"""
TODO List:
- Adaptive stepper
- Dense Interpolation
- Richardson Extrapolation
- Embedded Runge-Kutta Formula
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

def euler_step(f, t, y, h):
    return y + h*f(t,y)

@dataclass
class ODEResult:
    y: np.ndarray
    t: np.ndarray

def solve_to(f, y0, tspan, h, method) -> ODEResult:
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
    plt.show()