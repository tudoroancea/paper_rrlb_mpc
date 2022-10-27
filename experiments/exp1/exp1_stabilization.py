# The goal of the first experiment is to show that the scheme yields a local
# asymptotically stable system as proved theoretically.

import numpy as np
from rrlb import run_closed_loop_simulation
from rrlb.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(127)


def subexp1(xinit: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]), gen: bool = True):
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
        "epsilon_0": 30.0,
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
        generate_code=gen,
        build_solver=gen,
    )

    return results


def exp1():
    # generate initial states in the following bounding box:
    # 0.1 <= c_A, c_B <= 6.0 ; 100.0 <= theta <= 140.0 ; 95.0 <= theta_K <= 140.0
    nbr_initial_states = 5
    initial_states = np.zeros((nbr_initial_states, 4))
    initial_states[:, 0] = (
        np.random.random_sample(nbr_initial_states) * (6.0 - 0.1) + 0.1
    )
    initial_states[:, 1] = (
        np.random.random_sample(nbr_initial_states) * (6.0 - 0.1) + 0.1
    )
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
    for i in range(1, nbr_initial_states + 1):
        if i != 11:
            print("Running exp1 on initial state n°" + str(i))
            results.append(subexp1(initial_states[i - 1], gen=False))
        else:
            print("Skipping exp1 on initial state n°" + str(i))
            results.append(None)

    # plot discrepancies
    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"figure.dpi": "100"})
    plt.figure(figsize=(6.4, 4.8))
    for i in range(nbr_initial_states + 1):
        if results[i] is not None:
            plt.plot(results[i]["discrepancies"], label=f"initial state {i}")

    plt.ylabel(r"distance to $x^*$")
    plt.xlabel("iteration")
    plt.savefig("exp1_discrepancies.png", bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    exp1()
