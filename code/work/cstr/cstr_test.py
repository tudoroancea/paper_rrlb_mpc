import matplotlib.pyplot as plt
import numpy as np

from cstr_experiment import run_closed_loop_simulation
from work.cstr.cstr_model import xr1

if __name__ == "__main__":
    run_closed_loop_simulation(
        rrlb=False,
        show_plot=False,
        # x0=xr1,
    )
    x_sim, u_sim, n_conv = run_closed_loop_simulation(
        rrlb=True,
        show_plot=False,
        # x0=xr1,
    )
    if np.isnan(x_sim).any():
        print("NaN in x_sim: {}".format(x_sim))
    if np.isnan(u_sim).any():
        print("NaN in u_sim: {}".format(u_sim))

    plt.show()
