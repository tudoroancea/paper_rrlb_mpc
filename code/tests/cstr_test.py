import matplotlib.pyplot as plt
import numpy as np

from work import run_closed_loop_simulation
from work.cstr import find_cstr_steady_state


def main():
    x_ref, u_ref = find_cstr_steady_state(1)
    params = {
        "N": 100,
        "Nsim": 20,
        "dt": 20 / 3600,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": np.array([1.0, 0.5, 100.0, 100.0]),
        "epsilon_0": 30.0,
        # "epsilon_rate": 0.9,
    }
    res_reg = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=False,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    plt.suptitle("CSTR - Regular MPC")
    res_rrlb = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    plt.suptitle("CSTR - RRLB MPC")
    print(f"n_conv_reg: {res_reg['n_convergence']}")
    print(f"n_conv_rrlb: {res_rrlb['n_convergence']}")
    # print average runtimes
    print(
        f"avg_time_reg: {1000*np.mean(res_reg['time_tot'])} ms, std: {1000*np.std(res_reg['time_tot'])} ms"
    )
    print(
        f"avg_time_rrlb: {1000*np.mean(res_rrlb['time_tot'])} ms, std: {1000*np.std(res_rrlb['time_tot'])} ms"
    )
    # plot time_tot
    plt.figure()
    plt.plot(1000 * res_reg["time_tot"], label="regular")
    plt.plot(1000 * res_rrlb["time_tot"], label="rrlb")
    plt.legend()
    plt.ylabel("time [ms]")
    plt.xlabel("time step")
    plt.title("Runtimes")

    # plot epsilon
    plt.figure()
    plt.plot(res_rrlb["epsilon"], label="rrlb")
    plt.legend()
    plt.ylabel("$\epsilon$")
    plt.xlabel("time step")
    plt.title("evolution barrier parameter $\epsilon$")

    plt.show()


def main2():
    x_ref, u_ref = find_cstr_steady_state(1)
    params = {
        "N": 100,
        "Nsim": 20,
        "dt": 20 / 3600,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": np.array([1.0, 0.5, 100.0, 100.0]),
        "epsilon_0": 100.0,
        "epsilon_rate": 0.2,
    }
    res_var_eps = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    plt.suptitle("CSTR - RRLB MPC, variable epsilon")
    params["epsilon_rate"] = 1.0
    res_fixed_eps = run_closed_loop_simulation(
        problem="cstr",
        problem_params=params,
        rrlb=True,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    plt.suptitle("CSTR - RRLB MPC, fixed epsilon")
    print(f"n_conv_fixed_eps: {res_fixed_eps['n_convergence']}")
    print(f"n_conv_var_eps: {res_var_eps['n_convergence']}")
    # print average runtimes
    print(
        f"avg_time_fixed_eps: {1000*np.mean(res_fixed_eps['time_tot'])} ms, std: {1000*np.std(res_fixed_eps['time_tot'])} ms"
    )
    print(
        f"avg_time_var_eps: {1000 * np.mean(res_var_eps['time_tot'])} ms, std: {1000 * np.std(res_var_eps['time_tot'])} ms"
    )
    # plot time_tot
    plt.figure()
    plt.plot(1000 * res_fixed_eps["time_tot"], label="fixed epsilon")
    plt.plot(1000 * res_var_eps["time_tot"], label="variable epsilon")
    plt.legend()
    plt.ylabel("time [ms]")
    plt.xlabel("time step")
    plt.title("Runtimes")

    # plot epsilon
    plt.figure()
    plt.plot(res_var_eps["epsilon"])
    plt.plot(res_fixed_eps["epsilon"])
    plt.ylabel("$\epsilon$")
    plt.xlabel("time step")
    plt.title("evolution barrier parameter $\epsilon$")

    plt.show()


if __name__ == "__main__":
    main()
    # main2()
