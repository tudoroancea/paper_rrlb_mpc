import matplotlib.pyplot as plt
import numpy as np

from mass_chain.mass_chain_model import find_steady_state
from mass_chain.mass_chain_plot import plot_state


def main():
    M = 9
    x_last = np.array([1.0, 0.0, 0.0])
    xrest = find_steady_state(M, x_last=x_last)
    print(xrest)

    plt.figure()
    plot_state(xrest)
    plt.show()

    # perturb the system for 5 sampling times
    # x = xrest
    # model = export_cstr_model(0.2, M)
    # f_disc = ca.Function("f_disc", [model.x, model.u], [model.disc_dyn_expr])
    #
    # plot(x)
    # plt.pause(1.0)
    # for i in range(5):
    #     x = f_disc(x, np.array([-1.0, 1.0, 1.0])).full().flatten()
    #     plot(x)
    #     plt.pause(1.0)
    #
    # plt.show()


if __name__ == "__main__":
    main()
