import matplotlib.pyplot as plt
import numpy as np
import scipy

from scicomp.finite_diff import (
    get_central_diff_matrix,
    Grid,
    DirichletBC,
    get_b_vec_from_BCs,
    apply_BCs_soln,
)


def solve_poisson_linear(a, b, N, alpha, beta, q):
    grid = Grid(a, b, N)

    A = get_central_diff_matrix(grid.N_inner, derivative=2)
    left_BC = DirichletBC(alpha)
    right_BC = DirichletBC(beta)
    b_DD = get_b_vec_from_BCs(grid, left_BC, right_BC)

    rhs = -b_DD - (grid.dx**2) * q(grid.x_inner)

    u_inner = np.linalg.solve(A, rhs).squeeze()
    return grid.x, apply_BCs_soln(u_inner, left_BC, right_BC)


def solve_bratu(N, mu):
    dx = 1 / N
    A = get_central_diff_matrix(N - 1, derivative=2)

    def inner(u):
        return A @ u + (dx) ** 2 * np.exp(0.1 * u)

    x0 = np.ones((N - 1, 1))
    res = scipy.optimize.root(inner, x0)
    print(res.success)
    return res.x


if __name__ == "__main__":
    x, u = solve_poisson_linear(a=0, b=1, N=10, alpha=0, beta=2, q=lambda x: 1)
    plt.plot(
        x,
        u,
    )
    plt.show()
    plt.plot(solve_bratu(20, mu=4))
    plt.show()
