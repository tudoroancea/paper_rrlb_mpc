from typing import Union, Callable, Optional

import numpy as np
from acados_template import AcadosOcpSolver
from casadi import Function
from scipy.linalg import solve_discrete_are

from cstr_model import *
from cstr_ocp import *
from cstr_plot import *

__all__ = ["run_closed_loop_simulation"]


def run_closed_loop_simulation(
    dt: float = 20 / 3600,
    N: int = 100,
    Nsim: int = 350,
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    rrlb: bool = True,
    show_plot: bool = True,
    plot_filename: str = "",
) -> dict[str, Union[float, bool, np.ndarray]]:
    # create ocp object to formulate the OCP
    ocp, stuff = export_cstr_ocp(dt=dt, N=N, x0=x0, x_ref=xr, u_ref=ur, rrlb=rrlb)

    # extract informations from ocp (dimensions, dynamics, matrices, etc...)
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    f_disc = Function("f_disc", [ocp.model.x, ocp.model.u], [ocp.model.disc_dyn_expr])
    Q = stuff["Q"]
    R = stuff["R"]
    A = stuff["A"]
    B = stuff["B"]
    M_x = stuff["M_x"]
    M_u = stuff["M_u"]

    # declare the variables that will contain the simulation data
    xcurrent = x0
    last_prediction = np.zeros((N + 1, nx + nu))
    n_convergence = Nsim + 1
    sim_data = {
        "x_sim": [x0],
        "u_sim": [],
        "time_tot": [],
        "epsilon": [],
    }

    # compute the first runtime parameters for the RRLB MPC (barrier parameter epsilon and terminal cost P)
    def compute_runtime_parameters(iteration: Optional[int] = None):
        epsilon = 30.0 * 0.5**iteration
        print(f"epsilon: {epsilon}")
        P = solve_discrete_are(A, B, Q + epsilon * M_x, R + epsilon * M_u)
        return np.append(epsilon, P.ravel("F"))

    if rrlb:
        ocp.parameter_values = np.zeros(1 + nx * nx)

    # create an acados ocp solver
    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="acados_ocp_" + ocp.model.name + ".json",
    )

    # control loop
    for i in range(Nsim):
        # define initial guess for the solver
        if i == 0:
            # apply the reference control ur to the current state xcurrent
            xtpr = xcurrent
            for j in range(N):
                acados_ocp_solver.set(j, "x", xtpr)
                acados_ocp_solver.set(j, "u", ur)
                xtpr = f_disc(xtpr, ur).full().flatten()
            acados_ocp_solver.set(N, "x", xtpr)
        else:
            # shift the last prediction
            for j in range(N):
                acados_ocp_solver.set(j, "x", last_prediction[j + 1, :nx])
                acados_ocp_solver.set(j, "u", last_prediction[j + 1, nx:])
            acados_ocp_solver.set(
                N,
                "x",
                f_disc(last_prediction[-1, :nx], last_prediction[-1, nx:])
                .full()
                .flatten(),
            )

        # set runtime parameters in solver
        if rrlb:
            params = compute_runtime_parameters(iteration=i)
            sim_data["epsilon"].append(params[0])
            for j in range(N):
                acados_ocp_solver.set(j, "p", params)

        # solve ocp
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        status = acados_ocp_solver.solve()
        if status != 0:
            raise Exception(
                "acados ocp solver returned status {}. Exiting.".format(status)
            )
        # extract prediction
        for j in range(N):
            last_prediction[j, :] = np.append(
                acados_ocp_solver.get(j, "x"), acados_ocp_solver.get(j, "u")
            )
        last_prediction[N, :] = np.append(acados_ocp_solver.get(N, "x"), ur)

        sim_data["u_sim"].append(acados_ocp_solver.get(0, "u"))
        xcurrent = f_disc(xcurrent, sim_data["u_sim"][-1]).full().flatten()
        sim_data["x_sim"].append(xcurrent)
        sim_data["time_tot"].append(acados_ocp_solver.get_stats("time_tot"))

        # check if there is convergence in relative norm
        if np.linalg.norm(xcurrent - xr) / np.linalg.norm(xr) < 1.0e-6:
            n_convergence = i + 1
            break

    # create np.ndarrays for the simulation data
    for key, val in sim_data.items():
        if isinstance(val, list):
            sim_data[key] = np.array(val)

    # plot data
    plot_cstr(
        ocp,
        xr,
        ur,
        sim_data["x_sim"],
        sim_data["u_sim"],
        dt * 3600,
        file_name=plot_filename,
        show=show_plot,
    )

    sim_data["n_convergence"] = n_convergence
    return sim_data
