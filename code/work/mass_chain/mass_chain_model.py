import time

import casadi
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, Function, rootfinder, MX, jacobian, DM
import casadi as ca

__all__ = []

from matplotlib import pyplot as plt

# model constants
# x0 = np.zeros(3)  # [m]
g = np.array([0.0, 0.0, -9.81])  # [m.s^-2]
L = 0.033  # [m]
D = 0.1  # [N]
m = 0.03  # [kg]


def export_cstr_model(dt: float, M: int, rk4_nodes: int = 10) -> AcadosModel:
    """
    Create an AcadosModel for the mass chain model.

    :param dt: sampling time
    :type dt: float
    :param M: number of chained masses
    :type M: int
    :param rk4_nodes: number of nodes for the Runge-Kutta 4 integrator
    :type rk4_nodes: int

    :return: the AcadosModel instance and the function giving the accelerations of the intermediate masses
    """
    X = SX.sym("X", 3 * (M + 1))
    V = SX.sym("V", 3 * M)
    x = vertcat(X, V)
    u = SX.sym("u", 3)

    xdot = SX.sym("xdot", (2 * M + 1) * 3)

    forces = [
        D
        * (1 - L / casadi.norm_2(X[3 * (i + 1) : 3 * (i + 2)] - X[3 * i : 3 * (i + 1)]))
        * (X[3 * (i + 1) : 3 * (i + 2)] - X[3 * i : 3 * (i + 1)])
        for i in range(M)
    ]
    forces.insert(0, D * (1 - L / casadi.norm_2(X[:3])) * (X[:3]))

    A = vertcat(*[(forces[i] - forces[i - 1]) / m + g for i in range(1, M + 1)])

    f_cont = Function(
        "f_cont",
        [x, u],
        [vertcat(V, u, A)],
    )

    new_x = x
    for j in range(rk4_nodes):
        k1 = f_cont(new_x, u)
        k2 = f_cont(new_x + dt / 2 * k1, u)
        k3 = f_cont(new_x + dt / 2 * k2, u)
        k4 = f_cont(new_x + dt * k3, u)
        new_x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    model = AcadosModel()
    model.f_impl_expr = xdot - f_cont(x, u)
    model.f_expl_expr = f_cont(x, u)
    model.disc_dyn_expr = new_x
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = "mass_chain_" + str(M)

    return model


def find_steady_state(M: int, x_last: np.ndarray) -> np.ndarray:
    model = export_cstr_model(0.1, M)
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    # initial guess for the state
    x0 = np.zeros(nx)
    x0[: 3 * (M + 1)] = np.array(
        [
            np.linspace(0.0, x_last[0], M + 2)[1:],
            np.linspace(0.0, x_last[1], M + 2)[1:],
            np.linspace(0.0, x_last[2], M + 2)[1:],
        ]
    ).ravel("F")

    # decision variable
    w = ca.vertcat(model.x, model.xdot, model.u)
    # initial guess
    w0 = ca.vertcat(x0, np.zeros(nx), np.zeros(nu))

    # misuse IPOPT as a nonlinear equation solver
    nlp = {
        "x": w,
        "f": 0.0,
        "g": ca.vertcat(
            model.f_expl_expr,  # steady state equations
            model.x[3 * M : 3 * (M + 1)] - x_last,  # last mass position
        ),
    }
    solver = ca.nlpsol(
        "solver", "ipopt", nlp, {"ipopt.print_level": 0, "ipopt.sb": "yes"}
    )
    res = solver(x0=w0, lbg=0, ubg=0)
    w_opt = res["x"]
    print("residual=", ca.norm_inf(res["g"]))

    return w_opt[:nx].full().flatten()


if __name__ == "__main__":
    M = 9
    x_last = np.array([1.0, 0.0, 0.0])
    xrest = find_steady_state(M, x_last=x_last)

    def plot(bruh):
        bruh = np.append(np.zeros(3), bruh)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bruh[0 : 3 * (M + 2) : 3], bruh[1 : 3 * (M + 2) : 3], "o-")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(bruh[0 : 3 * (M + 2) : 3], bruh[2 : 3 * (M + 2) : 3], "o-")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

    # plt.figure()
    # plot(xrest)
    # plt.show()

    # perturb the system for 5 sampling times
    x = xrest
    model = export_cstr_model(0.2, M)
    f_disc = Function("f_disc", [model.x, model.u], [model.disc_dyn_expr])

    plot(x)
    plt.pause(1.0)
    for i in range(5):
        x = f_disc(x, np.array([-1.0, 1.0, 1.0])).full().flatten()
        plot(x)
        plt.pause(1.0)

    plt.show()
