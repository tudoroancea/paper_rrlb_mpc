# The goal of the first experiment is to show that the scheme yields a local asymptotically stable system as proved theoretically.

import numpy as np
from work import run_closed_loop_simulation
from work.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt

np.random.seed(127)


def subexp1(xinit: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]), gen: bool = True):
    x_ref, u_ref = find_cstr_steady_state(1)
    params = {
        "N": 10,
        "Nsim": 100,
        "dt": 50 / 3600,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": xinit,
    }
    rrlb_params = {
        "epsilon_0": 50.0,
        "epsilon_rate": 1.0,
    }
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        rrlb_params=rrlb_params,
        show_plot=False,
        verbose=True,
        generate_code=gen,
        build_solver=gen,
    )
    # print average runtime
    print("Average runtime: {} ms".format(1000 * np.mean(results["time_tot"])))

    return results


def exp1():
    # generate initial states in the following
    # 0.0 <= c_A, c_B <= 5.0 ; 98.0 <= theta <= 120.0 ; 92.0 <= theta_K <= 110.0
    nbr_initial_states = 5
    initial_states = np.zeros((nbr_initial_states, 4))
    initial_states[:, 0] = np.random.random_sample(nbr_initial_states) * 6.0 + 0.1
    initial_states[:, 1] = np.random.random_sample(nbr_initial_states) * 6.0 + 0.1
    initial_states[:, 2] = (
        np.random.random_sample(nbr_initial_states) * (140.0 - 100.0) + 100.0
    )
    initial_states[:, 3] = (
        np.random.random_sample(nbr_initial_states) * (140.0 - 95.0) + 95.0
    )
    # noinspection PyTypeChecker
    np.savetxt(
        "exp1_initial_states.csv",
        initial_states,
        delimiter=",",
        header="c_A,c_B,theta,theta_K",
    )
    # run simulations
    results = []
    print("Running exp1 on initial state n°0")
    results.append(subexp1())
    plt.suptitle("RRLB MPC - xinit n°0")
    for i in range(nbr_initial_states):
        print("Running exp1 on initial state n°" + str(i + 1))
        results.append(subexp1(initial_states[i], gen=False))
        plt.suptitle("RRLB MPC - xinit n°" + str(i + 1))

    # plot results
    plt.figure()
    for i in range(nbr_initial_states + 1):
        plt.plot(results[i]["discrepancies"], label=f"initial state {i}")
    plt.legend()
    plt.ylabel("discrepancy")
    plt.xlabel("iteration")
    plt.show()


if __name__ == "__main__":
    exp1()
