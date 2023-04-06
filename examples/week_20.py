import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scicomp.finite_diff import DirichletBC, Grid
from scicomp.pdes import solve_diffusion_method_lines

left_BC = right_BC = DirichletBC(0)
grid = Grid(0, 1, 30, left_BC, right_BC)

sol = solve_diffusion_method_lines(grid, 1, lambda x: np.sin(np.pi * x), t_span=(0, 10))

fig, ax = plt.subplots()
ax.set_ylim([0, np.amax(sol.u)])
(line1,) = ax.plot(grid.x, sol.u[0, :])


def update(frame):
    line1.set_data(grid.x, sol.u[frame, :])
    return (line1,)


ani = FuncAnimation(fig, update, frames=range(sol.t.size), interval=1)

plt.show()
