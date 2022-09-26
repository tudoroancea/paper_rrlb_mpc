import matplotlib.pyplot as plt
import numpy as np

from cstr_experiment import run_closed_loop_simulation

if __name__ == "__main__":
    res_reg = run_closed_loop_simulation(
        rrlb=False,
        show_plot=False,
    )
    res_rrlb = run_closed_loop_simulation(
        rrlb=True,
        show_plot=False,
        # x0=xr1,
    )
    print(f"n_conv_reg: {res_reg['n_convergence']}")
    print(f"n_conv_rrlb: {res_rrlb['n_convergence']}")
    # print average runtimes
    print(
        f"avg_time_reg: {1000*np.mean(res_reg['time_tot'])} ms, std: {1000*np.std(res_reg['time_tot'])} ms"
    )
    print(
        f"avg_time_rrlb: {1000*np.mean(res_rrlb['time_tot'])} m, std: {1000*np.std(res_rrlb['time_tot'])} ms"
    )

    plt.show()
