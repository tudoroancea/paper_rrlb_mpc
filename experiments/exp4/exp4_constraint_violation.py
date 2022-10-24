# The goal of the fourth experiment is to investigate how where the constraint
# violations appear.

import numpy as np
from work import run_closed_loop_simulation
from work.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt

np.random.seed(127)


def subexp4(epsilon_rate: float = 1.0):
    x_ref, u_ref = find_cstr_steady_state(1)
    T = 200 / 3600
    N = 10
    params = {
        "N": N,
        "Nsim": 100,
        "dt": T / N,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": np.array([1.0, 0.5, 100.0, 100.0]),
    }
    rrlb_params = {
        "epsilon_0": 30.0,
        "epsilon_rate": epsilon_rate,
    }
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        rrlb_params=rrlb_params,
        plot=False,
        show_plot=False,
        verbose=False,
    )
    # print info on the run
    print(
        "Average runtime: {} ± {} ms".format(
            1000 * np.mean(results["time_tot"]), 1000 * np.std(results["time_tot"])
        )
    )
    print("Performance measure: {}".format(results["performance_measure"]))

    return results


def exp4():
    res_fixed_eps = subexp4(epsilon_rate=1.0)
    # plt.suptitle("Running RRLB NMPC - fixed epsilon")
    res_decreasing_eps = subexp4(epsilon_rate=0.4)
    # plt.suptitle("Running RRLB NMPC - decreasing epsilon")

    # plot discrepancies
    plt.figure()
    plt.plot(res_fixed_eps["discrepancies"], label="fixed $\epsilon$")
    plt.plot(res_decreasing_eps["discrepancies"], label="decreasing $\epsilon$")
    plt.legend()
    plt.ylabel("discrepancy")
    plt.xlabel("iteration")

    # plot constraint violations
    plt.figure()
    plt.plot(res_fixed_eps["constraint_violations"], label=f"fixed $\epsilon$")
    plt.plot(
        res_decreasing_eps["constraint_violations"], label=f"decreasing $\epsilon$"
    )
    plt.legend()
    plt.ylabel("constraint violation")
    plt.xlabel("iteration")

    # plot runtimes
    plt.figure()
    plt.boxplot(
        1000 * np.array([res_fixed_eps["time_tot"], res_decreasing_eps["time_tot"]]).T,
        labels=["fixed $\epsilon$", "decreasing $\epsilon$"],
    )
    plt.legend()
    plt.ylabel("runtime [ms]")
    plt.xlabel("iteration")

    plt.show()


if __name__ == "__main__":
    exp4()
