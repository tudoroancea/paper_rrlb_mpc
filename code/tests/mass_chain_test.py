import matplotlib.pyplot as plt
import numpy as np

from work import run_closed_loop_simulation

if __name__ == "__main__":
    params = {
        "M": 7,
        "x_last": np.array([1.0, 0.0, 0.0]),
        "N": 40,
        "Nsim": 1000,
        "dt": 20 / 3600,
        "epsilon_0": 100.0,
        "epsilon_rate": 0.2,
    }
    res_reg = run_closed_loop_simulation(
        problem="mass_chain",
        problem_params=params,
        rrlb=False,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    # plt.suptitle("Mass Chain - Regular MPC")
    # res_rrlb = run_closed_loop_simulation(
    #     problem="mass_chain",
    #     params=params,
    #     rrlb=True,
    #     show_plot=False,
    #     # generate_code=False,
    #     # build_solver=False,
    # )
    # plt.suptitle("Mass Chain - RRLB MPC")
    print(f"n_conv_reg: {res_reg['n_convergence']}")
    # print(f"n_conv_rrlb: {res_rrlb['n_convergence']}")
    # print average runtimes
    print(
        f"avg_time_reg: {1000*np.mean(res_reg['time_tot'])} ms, std: {1000*np.std(res_reg['time_tot'])} ms"
    )
    # print(
    #     f"avg_time_rrlb: {1000*np.mean(res_rrlb['time_tot'])} ms, std: {1000*np.std(res_rrlb['time_tot'])} ms"
    # )
    # plot time_tot
    plt.figure()
    plt.plot(1000 * res_reg["time_tot"], label="regular")
    # plt.plot(1000 * res_rrlb["time_tot"], label="rrlb")
    plt.legend()
    plt.ylabel("time [ms]")
    plt.xlabel("time step")
    plt.title("Runtimes")

    # # plot epsilon
    # plt.figure()
    # plt.plot(res_rrlb["epsilon"], label="rrlb")
    # plt.legend()
    # plt.ylabel("$\epsilon$")
    # plt.xlabel("time step")
    # plt.title("evolution barrier parameter $\epsilon$")

    # plot discrepancies
    plt.figure()
    plt.plot(res_reg["discrepancies"], label="regular")
    # plt.plot(res_rrlb["discrepancy"], label="rrlb")
    plt.legend()
    plt.ylabel("discrepancy")
    plt.xlabel("time step")
    plt.title("evolution discrepancy")

    plt.show()
