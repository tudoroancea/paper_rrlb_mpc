from time import perf_counter

import casadi as ca
import numpy as np
from acados_template import AcadosModel

from ..common import export_model

__all__ = ["export_mass_chain_model", "find_mass_chain_steady_state"]


def export_mass_chain_model(dt: float, M: int, num_rk4_nodes: int = 10) -> AcadosModel:
    """
    Create an AcadosModel for the mass chain model.

    :param dt: sampling time
    :type dt: float
    :param M: number of chained masses
    :type M: int
    :param num_rk4_nodes: number of nodes for the Runge-Kutta 4 integrator
    :type num_rk4_nodes: int

    :return: the AcadosModel instance and the function giving the accelerations of the
        intermediate masses
    """
    X = ca.SX.sym("X", 3 * (M + 1))
    V = ca.SX.sym("V", 3 * M)
    x = ca.vertcat(X, V)
    u = ca.SX.sym("u", 3)

    # model constants
    g = np.array([0.0, 0.0, -9.81])  # [m.s^-2]
    L = 0.033  # [m]
    D = 0.1  # [N]
    m = 0.03  # [kg]

    forces = [
        D
        * (1 - L / ca.norm_2(X[3 * (i + 1) : 3 * (i + 2)] - X[3 * i : 3 * (i + 1)]))
        * (X[3 * (i + 1) : 3 * (i + 2)] - X[3 * i : 3 * (i + 1)])
        for i in range(M)
    ]
    forces.insert(0, D * (1 - L / ca.norm_2(X[:3])) * (X[:3]))

    A = ca.vertcat(*[(forces[i] - forces[i - 1]) / m + g for i in range(1, M + 1)])

    f_cont = ca.Function(
        "f_cont",
        [x, u],
        [ca.vertcat(V, u, A)],
    )

    return export_model("mass_chain_" + str(M), x, u, f_cont, dt, num_rk4_nodes)


def find_mass_chain_steady_state(M: int, x_last: np.ndarray) -> np.ndarray:
    model = export_mass_chain_model(0.1, M)
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
        "solver",
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
        },
    )
    start = perf_counter()
    res = solver(x0=w0, lbg=0, ubg=0)
    end = perf_counter()
    print(f"steady state computation time: {1000*(end - start):.3f} ms")
    w_opt = res["x"]

    return w_opt[:nx].full().flatten()
