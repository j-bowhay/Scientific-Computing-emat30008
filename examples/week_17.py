import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


def eq(x, c):
    return x**3 - x + c


# natural parameter continuation

c = [-2]
x = [root(lambda x: eq(x, c), [1.5], tol=1e-6).x]
h = 0.01

for _ in range(int(4 / h)):
    c_new = c[-1] + h
    sol = root(lambda x: eq(x, c_new), [x[-1]], tol=1e-6)
    if sol.success:
        c.append(c_new)
        x.append(sol.x)
    else:
        break

# plt.plot(c, x)
# plt.xlabel("c")
# plt.ylabel("x")
# plt.show()

# pseudo-arclength

c = [-2]

nu = [np.array([-2, *root(lambda x: eq(x, c), [1.5], tol=1e-6).x])]
h = 0.1
c_new = nu[-1][0] + h
sol = root(lambda x: eq(x, c_new), [x[-1]], tol=1e-6)
if sol.success:
    nu.append(np.array([c_new, *sol.x]))

print(nu)

for _ in range(50):
    secant = nu[-1] - nu[-2]
    nu_predicted = nu[-1] + secant

    sol = root(
        lambda x: [*eq(x[1:], x[0]), np.dot(secant, x - nu_predicted)],
        nu_predicted,
        tol=1e-6,
    )
    if sol.success:
        nu.append(sol.x)
    else:
        break

nu = np.asarray(nu)

plt.plot(nu[:, 0], nu[:, 1])
plt.show()
