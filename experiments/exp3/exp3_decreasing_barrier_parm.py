# The goal of the third experiment is to compare the behaviors of RRLB NMPC with fixed
# and decreasing barrier parameter epsilon.

import numpy as np
from work import run_closed_loop_simulation
from work.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt

np.random.seed(127)


def subexp3(epsilon_rate: float = 1.0, build: bool = True):
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

    return results


def exp3():
    print("Run 0")
    res_fixed_eps = subexp3(epsilon_rate=1.0, build=True)
    fixed_eps_runtimes = [res_fixed_eps["time_tot"]]
    fixed_eps_perf = res_fixed_eps["performance_measure"]
    print("Fixed epsilon performance: {}".format(fixed_eps_perf))
    for i in range(100):
        print("Run {}".format(i + 1))
        res_fixed_eps = subexp3(epsilon_rate=1.0, build=False)
        fixed_eps_runtimes.append(res_fixed_eps["time_tot"])

    fixed_eps_runtimes = np.concatenate(fixed_eps_runtimes)
    print(
        "Fixed epsilon average runtime: {} ± {} ms".format(
            np.mean(fixed_eps_runtimes) * 1000, np.std(fixed_eps_runtimes) * 1000
        )
    )

    print("Run 0")
    res_decreasing_eps = subexp3(epsilon_rate=0.4, build=True)
    decreasing_eps_runtimes = [res_decreasing_eps["time_tot"]]
    decreasing_eps_perf = res_decreasing_eps["performance_measure"]
    print("Decreasing epsilon performance: {}".format(decreasing_eps_perf))
    for i in range(100):
        print("Run {}".format(i + 1))
        res_decreasing_eps = subexp3(epsilon_rate=0.4, build=False)
        decreasing_eps_runtimes.append(res_decreasing_eps["time_tot"])

    decreasing_eps_runtimes = np.concatenate(decreasing_eps_runtimes)
    print(
        "Decreasing epsilon average runtime: {} ± {} ms".format(
            np.mean(decreasing_eps_runtimes) * 1000,
            np.std(decreasing_eps_runtimes) * 1000,
        )
    )

    fig, ax = plt.subplots()
    ax.boxplot([fixed_eps_runtimes, decreasing_eps_runtimes])
    ax.set_xticklabels(["Fixed epsilon", "Decreasing epsilon"])
    ax.set_ylabel("Runtime (s)")
    plt.show()

    return fixed_eps_perf, decreasing_eps_perf

    # # plot discrepancies
    # plt.figure()
    # plt.plot(res_fixed_eps["discrepancies"], label="fixed $\epsilon$")
    # plt.plot(res_decreasing_eps["discrepancies"], label="decreasing $\epsilon$")
    # plt.legend()
    # plt.ylabel("discrepancy")
    # plt.xlabel("iteration")
    #
    # # plot runtimes
    # plt.figure()
    # plt.boxplot(
    #     1000 * np.array([res_fixed_eps["time_tot"], res_decreasing_eps["time_tot"]]).T,
    #     labels=["fixed $\epsilon$", "decreasing $\epsilon$"],
    # )
    # plt.legend()
    # plt.ylabel("runtime [ms]")
    # plt.xlabel("iteration")
    #
    # plt.show()


if __name__ == "__main__":
    exp3()
