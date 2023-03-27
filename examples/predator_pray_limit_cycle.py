import matplotlib.pyplot as plt
import scicomp
from scicomp.odes import predator_prey
from scicomp.shooting import DerivativePhaseCondition, find_limit_cycle

a = 1
d = 0.1
b = 0.1

res = scicomp.integrate.solve_ivp(
    lambda t, y: predator_prey(t, y, a, b, d),
    t_span=(0, 100),
    y0=[1, 1],
    r_tol=1e-6,
    method="rkf45",
)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()

pc = DerivativePhaseCondition(0)
solver_args = {"method": "rkf45", "r_tol": 1e-5}
print(
    find_limit_cycle(
        lambda t, y: predator_prey(t, y, a, b, d),
        y0=[0.8, 0.2],
        T=30,
        phase_condition=pc,
        ivp_solver_kwargs=solver_args,
    )
)
