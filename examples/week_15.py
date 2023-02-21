import matplotlib.pyplot as plt
import scipy
from scicomp.odes import predator_prey
from scicomp.shooting import find_limit_cycle

a = 1
d = 0.1
b = 0.1

res = scipy.integrate.solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d), (0, 100), [1, 1], rtol=1e-6
)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()

print(find_limit_cycle(lambda t, y: predator_prey(t, y, a, b, d), [0.8, 0.2, 30]))
