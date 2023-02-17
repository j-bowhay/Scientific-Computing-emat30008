import numpy as np
import scipy


def find_limit_cycle(ode: callable, x0: np.ndarray) -> np.ndarray:
    def condition(x):
        condition_1 = x[:2] - scipy.integrate.solve_ivp(ode, (0, x[2]), x[:2],
                                                        rtol=1e-5).y[:, -1]
        # make this generic
        condition_2 = ode(np.nan, x[:2])[0]
        return [*condition_1, condition_2]
    
    return scipy.optimize.root(condition, x0).x