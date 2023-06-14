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
    results = run_closed_loop_simulation(
        problem="cstr",
        problem_params={
            "N": 10,
            "Nsim": 100,
            "dt": 20 / 3600,
            "x_ref": x_ref,
            "u_ref": u_ref,
            "xinit": np.array([1.0, 0.5, 100.0, 100.0]),
        },
        rrlb=rrlb,
        rrlb_params={
            "epsilon_0": 30.0,
            "epsilon_rate": 1.0,
        },
        plot=False,
        show_plot=False,
        verbose=False,
        generate_code=build,
        build_solver=build,
        plot_filename="exp2_{}_trajectory.png".format("rrlb" if rrlb else "regular"),
    )

    return results


def exp2():
    # run RRLB NMPC and regular NMPC ==================================================
    print("RRLB NMPC Run 0")
    # plt.figure(figsize=(5, 5))
    res_rrlb = subexp2(rrlb=True, build=True)
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
    # plt.figure(figsize=(5, 5))
    res_reg = subexp2(rrlb=False, build=True)
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

    # plot the results ==========================================================
    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"figure.dpi": "100", "font.size": 12})

    # plot runtimes
    plt.figure(figsize=(5, 3))
    plt.boxplot([1000 * rrlb_runtimes, 1000 * reg_runtimes])
    plt.yscale("log")
    # plt.boxplot([np.log(1000 * rrlb_runtimes), np.log(1000 * reg_runtimes)])
    plt.xticks([1, 2], ["RRLB", "Regular"])
    plt.ylabel("Runtime [ms]")
    plt.tight_layout()
    plt.savefig("exp2_runtimes.png", bbox_inches="tight", dpi=300)
    plt.savefig("exp2_runtimes.eps", bbox_inches="tight", dpi=300)

    # plot discrepancies
    plt.figure(figsize=(5, 3))
    plt.semilogy(res_rrlb["discrepancies"], label="RRLB NMPC")
    plt.semilogy(res_reg["discrepancies"], label="Regular NMPC")
    plt.legend()
    plt.ylabel(r"distance to $x^*$")
    plt.xlabel("simulation step")
    plt.tight_layout()
    plt.savefig("exp2_discrepancies.png", bbox_inches="tight", dpi=500)
    plt.savefig("exp2_discrepancies.eps", bbox_inches="tight", dpi=500)

    plt.show()


if __name__ == "__main__":
    exp2()
