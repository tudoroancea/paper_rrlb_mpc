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
    lbx: np.ndarray = np.array([10.0, 10.0, 150.0, 150.0]),
    ubx: np.ndarray = np.array([0.0, 0.0, 98.0, 92.0]),
    lbu: np.ndarray = np.array([35.0, 0.0]),
    ubu: np.ndarray = np.array([3.0, -9000.0]),
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
    plt.plot(t_x, np.ones_like(t_x) * lbx[0], "r:")
    plt.plot(t_x, np.ones_like(t_x) * lbx[1], "b:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[0], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[1], "b:")

    plt.subplot(4, 1, 2)
    plt.plot(t_x, x_sim[:, 2], "r-")
    plt.plot(t_x, x_sim[:, 3], "b-")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[2], "r--")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[3], "b--")
    plt.plot(t_x, np.ones_like(t_x) * lbx[2], "r:")
    plt.plot(t_x, np.ones_like(t_x) * lbx[3], "b:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[2], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[3], "b:")
    # plt.ylim(())

    plt.subplot(4, 1, 3)
    plt.plot(t_u, u_sim[:, 0], "r-")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[0], "r--")
    plt.plot(t_u, np.ones_like(t_u) * lbu[0], "r:")
    plt.plot(t_u, np.ones_like(t_u) * ubu[0], "r:")

    plt.subplot(4, 1, 4)
    plt.plot(t_u, u_sim[:, 1], "b-")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[1], "b--")
    plt.plot(t_u, np.ones_like(t_u) * lbu[1], "b:")
    plt.plot(t_u, np.ones_like(t_u) * ubu[1], "b:")

    plt.tight_layout()
    plt.grid()

    if file_name != "":
        plt.savefig(file_name, dpi=300, format="png")

    if show:
        plt.show()
