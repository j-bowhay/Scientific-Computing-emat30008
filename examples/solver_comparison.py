import matplotlib.pyplot as plt
import scipy
from scicomp.integrate import solve_ivp
from scicomp.odes import predator_prey

a = 1
d = 0.1
b = 0.1

y0 = [1, 0.5]

res = solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d),
    t_span=(0, 100),
    y0=y0,
    r_tol=1e-6,
    method="ck45",
)
res2 = scipy.integrate.solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d), (0, 100), y0, rtol=1e-6
)

print(len(res.t))
print(len(res2.t))
plt.plot(res.t, res.y[0, :], label="scicomp")
plt.plot(res.t, res.y[1, :], label="scicomp")
plt.plot(res2.t, res2.y[0, :], label="scipy")
plt.plot(res2.t, res2.y[1, :], label="scipy")
plt.legend()
plt.show()
