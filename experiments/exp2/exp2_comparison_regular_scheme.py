# The goal of the second experiment is to compare the behaviors of regular NMPC and RRLB
# NMPC to show the advantages our new scheme has.
import sys

import numpy as np
from rrlb import run_closed_loop_simulation
from rrlb.cstr import find_cstr_steady_state
import matplotlib.pyplot as plt

np.random.seed(127)


def subexp2(rrlb: bool = True, build: bool = True):
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
        "epsilon_rate": 1.0,
    }
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=rrlb,
        rrlb_params=rrlb_params,
        plot=build,
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


def exp2():
    # run RRLB NMPC and regular NMPC
    print("RRLB NMPC Run 0")
    res_rrlb = subexp2(rrlb=True, build=True)
    plt.suptitle("RRLB NMPC")
    rrlb_runtimes = [res_rrlb["time_tot"]]
    rrlb_perf = res_rrlb["performance_measure"]
    print("RRLB performance: {}".format(rrlb_perf))
    for i in range(100):
        sys.stdout.write("Run {}\r".format(i + 1))
        res_rrlb = subexp2(rrlb=True, build=False)
        rrlb_runtimes.append(res_rrlb["time_tot"])

    rrlb_runtimes = np.concatenate(rrlb_runtimes)
    print(
        "Average runtime RRLB MPC: {} ± {} ms".format(
            1000 * np.mean(rrlb_runtimes), 1000 * np.std(rrlb_runtimes)
        )
    )

    print("Regular NMPC Run 0")
    res_reg = subexp2(rrlb=False, build=True)
    plt.suptitle("Regular NMPC")
    reg_runtimes = [res_reg["time_tot"]]
    reg_perf = res_reg["performance_measure"]
    print("Regular performance: {}".format(reg_perf))
    for i in range(100):
        sys.stdout.write("Run {}\r".format(i + 1))
        res_reg = subexp2(rrlb=False, build=False)
        reg_runtimes.append(res_reg["time_tot"])

    reg_runtimes = np.concatenate(reg_runtimes)
    print(
        "Average runtime Regular MPC: {} ± {} ms".format(
            1000 * np.mean(reg_runtimes), 1000 * np.std(reg_runtimes)
        )
    )

    # plot the results
    plt.figure()
    plt.boxplot([rrlb_runtimes, reg_runtimes])
    plt.xticks([1, 2], ["RRLB", "Regular"])
    plt.ylabel("Runtime [s]")
    plt.title("Runtime Comparison")

    # plot discrepancies
    plt.figure()
    plt.plot(res_rrlb["discrepancies"], label="RRLB NMPC")
    plt.plot(res_reg["discrepancies"], label="Regular NMPC")
    plt.legend()
    plt.ylabel("discrepancy")
    plt.xlabel("iteration")

    # plot constraint violations
    # plt.figure()
    # plt.plot(res_rrlb["constraint_violations"], label=f"RRLB NMPC")
    # plt.plot(res_reg["constraint_violations"], label=f"Regular NMPC")
    # plt.legend()
    # plt.ylabel("constraint violation")
    # plt.xlabel("iteration")

    # plot runtimes
    # plt.figure()
    # plt.boxplot(
    #     1000 * np.array([res_rrlb["time_tot"], res_reg["time_tot"]]).T,
    #     labels=["RRLB NMPC", "Regular NMPC"],
    # )
    # plt.legend()
    # plt.ylabel("runtime [ms]")
    # plt.xlabel("iteration")

    plt.show()


if __name__ == "__main__":
    exp2()
