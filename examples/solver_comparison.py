import matplotlib.pyplot as plt
import scipy
import numpy as np
from scicomp.odes import predator_prey
from scicomp.integrate import solve_ivp

a = 1
d = 0.1
b = 0.1

res = solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d),
    t_span=(0, 100),
    y0=[1, 1],
    r_tol=1e-1,
    method="rk4",
    h=0.01,
    max_step=0.01,
)
res2 = scipy.integrate.solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d), (0, 100), [1, 1], rtol=1e-6
)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.plot(res2.t, res2.y[0, :])
plt.plot(res2.t, res2.y[1, :])
plt.show()
