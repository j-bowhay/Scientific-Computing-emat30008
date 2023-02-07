from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

def euler_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h*np.asarray(f(t,y))

def rk4_step(f: callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = np.asarray(f(t,y))
    k2 = np.asarray(f(t + h/2, y + h*k1/2))
    k3 = np.asarray(f(t + h/2, y + h*k2/2))
    k4 = np.asarray(f(t + h, y + h*k3))
    
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
    ...