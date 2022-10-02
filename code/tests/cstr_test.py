import matplotlib.pyplot as plt
import numpy as np

from work import run_closed_loop_simulation
from work.cstr import find_cstr_steady_state

if __name__ == "__main__":
    x_ref, u_ref = find_cstr_steady_state(1)
    params = {
        "N": 100,
        "Nsim": 40,
        "dt": 20 / 3600,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "xinit": np.array([1.0, 0.5, 100.0, 100.0]),
    }
    res_reg = run_closed_loop_simulation(
        problem="cstr",
        params=params,
        rrlb=False,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    res_rrlb = run_closed_loop_simulation(
        problem="cstr",
        params=params,
        rrlb=True,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
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

    # plot epsilon
    plt.figure()
    plt.plot(res_rrlb["epsilon"], label="rrlb")
    plt.legend()
    plt.ylabel("epsilon")
    plt.xlabel("time step")

    plt.show()
