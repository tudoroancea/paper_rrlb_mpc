import matplotlib.pyplot as plt

from cstr_experiment import run_closed_loop_simulation
from work.cstr.cstr_model import xr1

if __name__ == "__main__":
    x_sim_reg, u_sim_reg, n_conv_reg = run_closed_loop_simulation(
        rrlb=False,
        show_plot=False,
        # x0=xr1,
    )
    x_sim_rrlb, u_sim_rrlb, n_conv_rrlb = run_closed_loop_simulation(
        rrlb=True,
        show_plot=False,
        # x0=xr1,
    )
    print(f"n_conv_reg: {n_conv_reg}")
    print(f"n_conv_rrlb: {n_conv_rrlb}")
    plt.show()
