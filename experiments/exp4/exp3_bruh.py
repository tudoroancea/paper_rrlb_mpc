import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

from rrlb import find_cstr_steady_state, run_closed_loop_simulation

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def bruh(n: int, epsilon: float):
    initial_states = np.load("exp3_initial_states_{}_{}.npy".format(n, int(epsilon)))
    constraint_violations = np.load(
        "exp3_constraint_violations_{}_{}.npy".format(n, int(epsilon))
    )
    print(
        "proportion of NaNs in constraint violations: {}".format(
            np.count_nonzero(np.isnan(constraint_violations))
            / constraint_violations.size
        )
    )
    # find the initial states with NaN in constraint violations
    nan_indices = np.random.permutation(np.argwhere(np.isnan(constraint_violations)))[
        :10
    ]
    print("nan indices: {}".format(nan_indices))

    x_ref, u_ref = find_cstr_steady_state(1)
    T = 200 / 3600
    N = 10
    params = {
        "N": N,
        "Nsim": 100,
        "dt": T / N,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": initial_states[
            nan_indices[0, 0],
            nan_indices[0, 1],
            nan_indices[0, 2],
            nan_indices[0, 3],
            :,
        ],
    }
    rrlb_params = {
        "epsilon_0": epsilon,
        "epsilon_rate": 1.0,
    }
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=False,
        rrlb_params=rrlb_params,
        plot=True,
        show_plot=True,
        verbose=True,
        generate_code=True,
        build_solver=True,
    )


if __name__ == "__main__":
    bruh(20, 30.0)
