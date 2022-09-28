from typing import Union

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
    # verbosity: int = 0,
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

    # pre-allocate the variables that will contain the simulation data
    xcurrent = x0
    current_discrepancy = np.linalg.norm(xcurrent - xr) / np.linalg.norm(xr)
    x_sim = np.zeros((Nsim + 1, nx))
    x_sim[0, :] = x0
    u_sim = np.zeros((Nsim, nu))
    time_tot = np.ndarray((Nsim, 1))
    n_convergence = Nsim + 1

    # compute the first runtime parameters for the RRLB MPC (barrier parameter epsilon and terminal cost P)
    def compute_runtime_parameters(rel_discrepancy):
        epsilon = 100 * rel_discrepancy**2
        print(f"epsilon: {epsilon}")
        P = solve_discrete_are(A, B, Q + epsilon * M_x, R + epsilon * M_u)
        return np.append(epsilon, P.ravel("F"))

    if rrlb:
        ocp.parameter_values = compute_runtime_parameters(current_discrepancy)

    # create an acados ocp solver
    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_" + ocp.model.name + ".json"
    )

    # control loop
    for i in range(Nsim):
        # define initial guess for the solver by applying the reference control ur to the current state xcurrent
        xtpr = xcurrent
        for j in range(N):
            acados_ocp_solver.set(j, "x", xtpr)
            acados_ocp_solver.set(j, "u", ur)
            xtpr = f_disc(xtpr, ur).full().flatten()

        # set runtime parameters in solver
        if rrlb:
            params = compute_runtime_parameters(current_discrepancy)
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
        u_sim[i, :] = acados_ocp_solver.get(0, "u")

        # get statistics
        time_tot[i] = acados_ocp_solver.get_stats("time_tot")

        # update state
        xcurrent = f_disc(xcurrent, u_sim[i, :]).full().flatten()
        x_sim[i + 1, :] = xcurrent

        # check if there is convergence in relative norm
        current_discrepancy = np.linalg.norm(xcurrent - xr) / np.linalg.norm(xr)
        if current_discrepancy < 1e-3:
            n_convergence = i + 1
            break

    x_sim = x_sim[: n_convergence + 1, :]
    u_sim = u_sim[:n_convergence, :]
    time_tot = time_tot[:n_convergence, :]

    # plot data
    plot_cstr(
        ocp,
        xr,
        ur,
        x_sim,
        u_sim,
        dt * 3600,
        file_name=plot_filename,
        show=show_plot,
    )

    return {
        "x_sim": x_sim,
        "u_sim": u_sim,
        "n_convergence": n_convergence,
        "time_tot": time_tot,
    }
