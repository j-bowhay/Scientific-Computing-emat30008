from scicomp.shooting import DerivativePhaseCondition, find_limit_cycle
from scicomp.integrate import solve_ivp
from scicomp.odes import hopf_normal
import scipy
import matplotlib.pyplot as plt

beta = 1
rho = -1

res = solve_ivp(
    lambda t, y: hopf_normal(t, y, beta, rho),
    y0=[1, 1],
    t_span=[0, 10],
    method="rkf45",
    r_tol=1e-5,
)

plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()


# pc = DerivativePhaseCondition(0)
# solver_args = {"method": "rkf45", "r_tol": 1e-5}
# res = find_limit_cycle(
#     lambda t, y: hopf_normal(t, y, beta, rho),
#     y0=[1, 1],
#     T=6.28,
#     phase_condition=pc,
#     ivp_solver_kwargs=solver_args,)
