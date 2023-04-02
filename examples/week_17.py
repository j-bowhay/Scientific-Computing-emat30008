import matplotlib.pyplot as plt
from scicomp.continuation import continuation


def eq(x, c):
    return x**3 - x + c


# natural parameter continuation

sol = continuation(
    eq,
    variable_kwarg="c",
    initial_value=-2,
    step_size=0.001,
    max_steps=4000,
    y0=[1.5],
    root_finder_kwargs={"tol": 1e-9},
    method="np",
)

plt.plot(sol.parameter_values, sol.state_values)
plt.xlabel("c")
plt.ylabel("x")
plt.show()

# pseudo-arclength

sol = continuation(
    eq,
    variable_kwarg="c",
    initial_value=-2,
    step_size=0.1,
    max_steps=50,
    y0=[1.5],
    root_finder_kwargs={"tol": 1e-6},
)
plt.plot(sol.parameter_values, sol.state_values)
plt.show()
