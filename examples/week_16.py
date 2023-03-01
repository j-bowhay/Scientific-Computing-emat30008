import matplotlib.pyplot as plt
import scipy
from scicomp.odes import hopf_normal
from scicomp.shooting import _find_limit_cycle

beta = 1
rho = -1

res = scipy.integrate.solve_ivp(
    lambda t, y: hopf_normal(t, y, beta, rho), (0, 20), [1, 1], rtol=1e-6
)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()


print(_find_limit_cycle(lambda t, y: hopf_normal(t, y, beta, rho), [1, 1, 6]))
