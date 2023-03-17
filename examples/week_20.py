import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = 0
b = 1
t_span = (0,1)
alpha = 0
beta = 0
D = 1
mu = 2

N = 20

x = np.linspace(a, b, N + 1)
x_inner = x[1:-1]
dx = (b - a) / N

k = [np.ones(N - 2), -2 * np.ones(N - 1), np.ones(N - 2)]
offset = [-1, 0, 1]
A = scipy.sparse.diags(k, offset).toarray()
b_DD = np.zeros((N - 1, 1)).squeeze()
b_DD[0] = alpha
b_DD[-1] = beta

def rhs(t,u):
    return (D/(dx)**2) * (A@u + b_DD) + np.exp(mu*u)

sol = scipy.integrate.solve_ivp(rhs, t_span, np.zeros_like(b_DD), r_tol=1e-6)
U = sol.y.T

fig, ax = plt.subplots()
ax.set_ylim([0,np.amax(U)])
line1, = ax.plot(x_inner, U[0,:])

def update(frame):
    line1.set_data(x_inner, U[frame,:])
    return line1,

ani = FuncAnimation(
    fig, update,
    frames=range(U.shape[0]),
    interval=1)

plt.show()