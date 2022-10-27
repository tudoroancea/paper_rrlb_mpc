# The goal of the third experiment is to investigate how where the constraint
# violations appear.
import sys
from time import perf_counter

import numpy as np

from rrlb import run_closed_loop_simulation
from rrlb.cstr import find_cstr_steady_state

np.random.seed(127)


def subexp3(
    xinit: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    epsilon: float = 30.0,
    build: bool = True,
):
    x_ref, u_ref = find_cstr_steady_state(1)
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params={
            "N": 10,
            "Nsim": 100,
            "dt": 20 / 3600,
            "x_ref": x_ref,
            "u_ref": u_ref,
            "xinit": xinit,
        },
        rrlb=True,
        rrlb_params={"epsilon_0": epsilon, "epsilon_rate": 1.0},
        plot=False,
        show_plot=False,
        verbose=False,
        generate_code=build,
        build_solver=build,
    )
    res = np.sum(results["constraint_violations"])
    if res >= 100.0:
        print("constraints violated for xinit={}".format(xinit))
        print(res)
        pass


def exp3(n: int, epsilon: float):
    start = perf_counter()
    c_A_init = np.linspace(0.0, 10.0, n)
    c_B_init = np.linspace(0.0, 10.0, n)
    theta_init = np.linspace(98.0, 150.0, n)
    theta_K_init = np.linspace(92.0, 150.0, n)

    xinit = np.array([4.0, 5.0, 130.0, 95.0])

    theta_id = np.argmin(np.abs(theta_init - xinit[2]))
    theta_K_id = np.argmin(np.abs(theta_K_init - xinit[3]))
    print("theta={}, theta_K={}".format(theta_init[theta_id], theta_K_init[theta_K_id]))

    initial_states = np.zeros((n, n, 4))
    for i in range(n):
        for j in range(n):
            initial_states[i, j, :] = np.array(
                [
                    c_A_init[i],
                    c_B_init[j],
                    theta_init[theta_id],
                    theta_K_init[theta_K_id],
                ]
            )

    np.save("exp3_initial_states_{}_{}.npy".format(n, int(epsilon)), initial_states)
    stop = perf_counter()
    print("Created and dumped initial states in {} ms".format(1000 * (stop - start)))

    # run simulations =================================================================
    start = perf_counter()
    results = np.zeros((n, n))
    subexp3()
    for i in range(n):
        for j in range(n):
            try:
                sys.stdout.write("Running exp3 on initial state {},{}\r".format(i, j))
                results[i, j] = subexp3(
                    initial_states[i, j, :],
                    build=False,
                )
                if results[i, j] >= 100.0:
                    print(i, j)
                    # print(initial_states[i, j, :])
                    # print(results[i, j])
                    results[i, j] = np.nan
            except ValueError as e:
                results[i, j] = np.nan

    stop = perf_counter()
    print("ran all experiments in {} s".format(stop - start))

    # plot constraint violations
    np.save("exp3_constraint_violations_{}_{}.npy".format(n, int(epsilon)), results)


if __name__ == "__main__":
    exp3(n=30, epsilon=0.0)
