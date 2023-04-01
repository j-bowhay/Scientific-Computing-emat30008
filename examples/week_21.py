import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.animation import FuncAnimation

from scicomp.finite_diff import get_central_diff_matrix

alpha = 0
beta = 0
L = 1
D = 0.1
N = 100
dt = 0.1
Nt = 1000
C = (dt * D) / (dt**2)

x = np.linspace(0, L, N + 1)
u = np.empty((Nt, N - 1))
u[0, :] = np.sin(np.pi * x[1:-1])

A_DD = get_central_diff_matrix(N - 1, derivative=2)
b_DD = np.zeros((N - 1, 1)).squeeze()
b_DD[0] = alpha
b_DD[-1] = beta

for i in range(1, Nt):
    A = np.eye(*A_DD.shape) - (C / 2) * A_DD
    b = (np.eye(*A_DD.shape) + (C / 2) * A_DD) @ u[i - 1, :] + C * b_DD
    u[i, :] = np.linalg.solve(A, b)


fig, ax = plt.subplots()
ax.set_ylim([0, np.amax(u)])
(line1,) = ax.plot(x[1:-1], u[0, :])


def update(frame):
    line1.set_data(x[1:-1], u[frame, :])
    return (line1,)


ani = FuncAnimation(fig, update, frames=range(u.shape[0]), interval=1)

plt.show()
