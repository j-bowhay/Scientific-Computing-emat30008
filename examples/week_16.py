from scicomp.odes import hopf_normal
from scicomp.shooting import find_limit_cycle

import scipy
import matplotlib.pyplot as plt

beta = 1
rho = -1

res = scipy.integrate.solve_ivp(lambda t, y: hopf_normal(t, y, beta, rho), (0, 20), [1,1], rtol=1e-6)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()


print(find_limit_cycle(lambda t, y: hopf_normal(t, y, beta, rho), [1, 1, 6]))