import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp

__all__ = ["plot_cstr"]


def plot_cstr(
    ocp: AcadosOcp,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    x_sim: np.ndarray,
    u_sim: np.ndarray,
    dt: float,
    file_name: str = "",
    show: bool = True,
):
    # solid line: real data
    # dashed line: reference values
    # dotted line: bounds
    t_x = np.linspace(0, dt * (x_sim.shape[0] - 1), x_sim.shape[0])
    t_u = np.linspace(0, dt * (u_sim.shape[0] - 1), u_sim.shape[0])

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t_x, x_sim[:, 0], "r-")
    plt.plot(t_x, x_sim[:, 1], "b-")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[0], "r--")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[1], "b--")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[0]["lbx"], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[1]["lbx"], "b:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[0]["ubx"], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[1]["ubx"], "b:")

    plt.subplot(4, 1, 2)
    plt.plot(t_x, x_sim[:, 2], "r-")
    plt.plot(t_x, x_sim[:, 3], "b-")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[2], "r--")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[3], "b--")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[2]["lbx"], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[3]["lbx"], "b:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[2]["ubx"], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ocp.dims.nx[3]["ubx"], "b:")

    plt.subplot(4, 1, 3)
    plt.plot(t_u, u_sim[:, 0], "r-")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[0], "r--")
    plt.plot(t_u, np.ones_like(t_u) * ocp.dims.nu[0]["lbu"], "r:")
    plt.plot(t_u, np.ones_like(t_u) * ocp.dims.nu[0]["ubu"], "r:")

    plt.subplot(4, 1, 4)
    plt.plot(t_u, u_sim[:, 1], "b-")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[1], "b--")
    plt.plot(t_u, np.ones_like(t_u) * ocp.dims.nu[1]["lbu"], "b:")
    plt.plot(t_u, np.ones_like(t_u) * ocp.dims.nu[1]["ubu"], "b:")

    if file_name != "":
        plt.savefig(file_name, dpi=300, format="png")

    if show:
        plt.show()
