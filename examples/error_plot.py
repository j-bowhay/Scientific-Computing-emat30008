import matplotlib.pyplot as plt
import numpy as np

from scicomp.integrate import _solve_to_fixed_step, rk4_step, euler_step

ode = lambda t, y: [y[1], -y[0]]
res = _solve_to_fixed_step(ode, [1, 0], (0, 10), 1e-3, rk4_step)
plt.plot(res.t, res.y[:, 0])
plt.show()

x_true = lambda t : np.cos(t)
rk4_err = []
euler_err = []
hs = np.logspace(-1, -12, base=2)
for h in hs:
    res = _solve_to_fixed_step(ode, [1, 0], (0, 1), h, rk4_step)
    rk4_err.append(np.abs(x_true(res.t[-1]) - res.y[-1,0]))
    res = _solve_to_fixed_step(ode, [1, 0], (0, 1), h, euler_step)
    euler_err.append(np.abs(x_true(res.t[-1]) - res.y[-1,0]))
plt.loglog(hs, rk4_err)
plt.loglog(hs, euler_err)
plt.show()