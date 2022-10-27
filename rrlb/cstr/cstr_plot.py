import matplotlib as mpl
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

    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"figure.dpi": "100"})

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
    plt.ylabel("concentrations [mol/l]")
    plt.legend(
        [
            r"$c_A$",
            r"$c_B$",
            r"$c_{A,ref}$",
            r"$c_{B,ref}$",
            r"$c_{A,lb},~c_{A,ub}$",
            r"$c_{B,lb},~c_{B,ub}$",
        ],
        loc="upper right",
    )

    plt.subplot(4, 1, 2)
    plt.plot(t_x, x_sim[:, 2], "r-")
    plt.plot(t_x, x_sim[:, 3], "b-")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[2], "r--")
    plt.plot(t_x, np.ones_like(t_x) * x_ref[3], "b--")
    plt.plot(t_x, np.ones_like(t_x) * lbx[2], "r:")
    plt.plot(t_x, np.ones_like(t_x) * lbx[3], "b:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[2], "r:")
    plt.plot(t_x, np.ones_like(t_x) * ubx[3], "b:")
    plt.ylabel("temperatures [°C]")
    plt.legend(
        [
            r"$\theta$",
            r"$\theta_K$",
            r"$\theta_{ref}$",
            r"$\theta_{K,ref}$",
            r"$\theta_{lb}$",
            r"$\theta_{K,lb}$",
            r"$\theta_{ub}$",
            r"$\theta_{K,ub}$",
        ],
        loc="upper right",
    )

    plt.subplot(4, 1, 3)
    plt.step(t_u, u_sim[:, 0], "r-", where="post")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[0], "r--")
    plt.plot(t_u, np.ones_like(t_u) * lbu[0], "r:")
    plt.plot(t_u, np.ones_like(t_u) * ubu[0], "r:")
    plt.ylabel("scaled feed inflow [1/h]")
    plt.legend(
        [
            r"$u_1$",
            r"$u_{1ref}$",
            r"$u_{1,lb}$",
            r"$u_{1,ub}$",
        ],
        loc="upper right",
    )

    plt.subplot(4, 1, 4)
    plt.step(t_u, u_sim[:, 1], "b-", where="post")
    plt.plot(t_u, np.ones_like(t_u) * u_ref[1], "b--")
    plt.plot(t_u, np.ones_like(t_u) * lbu[1], "b:")
    plt.plot(t_u, np.ones_like(t_u) * ubu[1], "b:")
    plt.ylabel("heat removal rate [kJ/h]")
    plt.xlabel("time [s]")
    plt.legend(
        [
            r"$u_2$",
            r"$u_{2ref}$",
            r"$u_{2,lb}$",
            r"$u_{2,ub}$",
        ],
        loc="upper right",
    )

    plt.tight_layout()

    if file_name != "":
        plt.savefig(file_name, dpi=300, format="png", bbox_inches="tight")

    if show:
        plt.show()
