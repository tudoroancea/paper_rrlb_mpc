# The goal of the fourth experiment is to investigate how where the constraint
# violations appear.
import itertools
import os.path
from time import perf_counter

import numpy as np
from rrlb import run_closed_loop_simulation
from rrlb.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt

np.random.seed(127)


def subexp4(
    xinit: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    epsilon: float = 30.0,
    build: bool = True,
):
    x_ref, u_ref = find_cstr_steady_state(1)
    T = 200 / 3600
    N = 10
    params = {
        "N": N,
        "Nsim": 100,
        "dt": T / N,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": xinit,
    }
    rrlb_params = {
        "epsilon_0": epsilon,
        "epsilon_rate": 1.0,
    }
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        rrlb_params=rrlb_params,
        plot=False,
        show_plot=False,
        verbose=False,
        generate_code=build,
        build_solver=build,
    )
    # print info on the run
    # print(
    #     "Average runtime: {} ± {} ms".format(
    #         1000 * np.mean(results["time_tot"]), 1000 * np.std(results["time_tot"])
    #     )
    # )
    # print("Performance measure: {}".format(results["performance_measure"]))

    return np.sum(results["constraint_violations"])


def exp4():
    n = 10
    epsilon = 30.0

    try:
        start = perf_counter()
        initial_states = np.load(
            "exp4_initial_states_{}_{}.npy".format(n, int(epsilon))
        )
        assert initial_states.shape == (n, n, n, n, 4)
        stop = perf_counter()
        print("Imported initial states in {} ms".format(stop - start))
    except:
        start = perf_counter()
        c_A_init = np.linspace(0.0, 10.0, n)
        c_B_init = np.linspace(0.0, 10.0, n)
        theta_init = np.linspace(98.0, 150.0, n)
        theta_K_init = np.linspace(92.0, 150.0, n)
        initial_states = np.zeros((n, n, n, n, 4))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        initial_states[i, j, k, l, :] = np.array(
                            [c_A_init[i], c_B_init[j], theta_init[k], theta_K_init[l]]
                        )

        np.save("exp4_initial_states_{}_{}.npy".format(n, int(epsilon)), initial_states)
        stop = perf_counter()
        print(
            "Created and dumped initial states in {} ms".format(1000 * (stop - start))
        )

    # run simulations
    start = perf_counter()
    results = np.zeros((n, n, n, n))
    subexp4()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    try:
                        print(
                            "Running exp4 on initial state {},{},{},{}".format(
                                i, j, k, l
                            )
                        )
                        results[i, j, k, l] = subexp4(
                            initial_states[i, j, k, l, :],
                            build=False,
                        )
                    except ValueError as e:
                        print("Error: {}\n putting None".format(e))
                        results[i, j, k, l] = np.nan

    stop = perf_counter()
    print("ran all experiments in {} s".format(stop - start))

    # plot constraint violations
    np.save("exp4_constraint_violations_{}_{}.npy".format(n, int(epsilon)), results)


if __name__ == "__main__":
    exp4()
