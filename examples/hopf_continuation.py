import matplotlib.pyplot as plt
from scicomp.continuation import numerical_continuation
from scicomp.integrate import solve_ivp
from scicomp.odes import modified_hopf
from scicomp.shooting import DerivativePhaseCondition, limit_cycle_shooting_func

res = solve_ivp(
    modified_hopf,
    y0=[1, 1],
    t_span=(0, 100),
    ode_params={
        "beta": 2,
    },
)
plt.plot(res.t, res.y[0, :])
plt.plot(res.t, res.y[1, :])
plt.show()

res = numerical_continuation(
    equation=modified_hopf,
    variable_kwarg="beta",
    initial_value=2,
    y0=[1, 1, 6],
    step_size=-0.1,
    max_steps=45,
    discretisation=limit_cycle_shooting_func,
    discretisation_kwargs={
        "phase_condition": DerivativePhaseCondition(0),
        "ivp_solver_kwargs": {"r_tol": 1e-6},
    },
)
print(res.parameter_values.shape)
plt.plot(res.parameter_values, res.state_values[0, :])
plt.plot(res.parameter_values, res.state_values[1, :])
plt.show()
