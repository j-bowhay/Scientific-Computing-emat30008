import numpy as np
import scipy
import matplotlib.pyplot as plt


def solve_poisson_linear(a, b, N, alpha, beta, q):
    x = np.linspace(a, b, N + 1)
    x_inner = x[1:-1]
    dx = (b - a) / N
    k = [np.ones(N - 2), -2 * np.ones(N - 1), np.ones(N - 2)]
    offset = [-1, 0, 1]
    A = scipy.sparse.diags(k, offset).toarray()
    b_DD = np.zeros((N - 1, 1))
    b_DD[0] = alpha
    b_DD[-1] = beta

    rhs = -b_DD - (dx**2) * q(x_inner)

    u_inner = np.linalg.solve(A, rhs).squeeze()
    return x, np.concatenate([[alpha], u_inner, [beta]])


def solve_bratu(N, mu):
    dx = 1 / N

    def inner(u):
        k = [np.ones(N - 2), -2 * np.ones(N - 1), np.ones(N - 2)]
        offset = [-1, 0, 1]
        A = scipy.sparse.diags(k, offset).toarray()

        return A @ u + (dx) ** 2 * np.exp(0.1 * u)

    x0 = np.ones((N - 1, 1))
    res = scipy.optimize.root(inner, x0)
    print(res.success)
    return res.x


if __name__ == "__main__":
    # x, u = solve_poisson_linear(a=0, b=1, N=10, alpha=0, beta=2, q=lambda x: 1)
    # plt.plot(x, u,)
    # plt.show()
    plt.plot(solve_bratu(20, mu=4))
    plt.show()
