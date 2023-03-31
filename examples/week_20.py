import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.animation import FuncAnimation

from scicomp.finite_diff import get_central_diff_matrix

a = 0
b = 1
t_span = (0, 1)
alpha = 0
beta = 0
D = 1
mu = 4

N = 20

x = np.linspace(a, b, N + 1)
x_inner = x[1:-1]
dx = (b - a) / N

A = get_central_diff_matrix(N-1, derivative=2)
b_DD = np.zeros((N - 1, 1)).squeeze()
b_DD[0] = alpha
b_DD[-1] = beta


def rhs(t, y):
    return (D / (dx) ** 2) * (A @ y + b_DD) + np.exp(mu * y)


sol = scipy.integrate.solve_ivp(
    rhs, t_span, np.zeros_like(b_DD), rtol=1e-12, atol=1e-12
)
# sol = scicomp.integrate.solve_ivp(rhs, t_span=t_span, y0=np.zeros_like(b_DD), r_tol=1e-9, method="dopri45")
U = sol.y.T

fig, ax = plt.subplots()
ax.set_ylim([0, np.amax(U)])
(line1,) = ax.plot(x_inner, U[0, :])


def update(frame):
    line1.set_data(x_inner, U[frame, :])
    return (line1,)


ani = FuncAnimation(fig, update, frames=range(U.shape[0]), interval=1)

plt.show()
