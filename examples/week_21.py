import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scicomp.finite_diff import DirichletBC, Grid
from scicomp.pdes import solve_linear_diffusion_implicit

left_BC = right_BC = DirichletBC(0)
grid = Grid(0, 1, 100, left_BC=left_BC, right_BC=right_BC)
u = solve_linear_diffusion_implicit(
    grid=grid, D=0.1, dt=0.01, steps=1000, u0_func=lambda x: np.sin(np.pi * x)
)

fig, ax = plt.subplots()
ax.set_ylim([0, np.amax(u)])
(line1,) = ax.plot(grid.x, u[0, :])


def update(frame):
    line1.set_data(grid.x, u[frame, :])
    return (line1,)


ani = FuncAnimation(fig, update, frames=range(u.shape[0]), interval=10)

plt.show()
