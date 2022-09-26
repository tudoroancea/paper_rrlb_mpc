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
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    rrlb: bool = True,
    show_plot: bool = True,
    plot_filename: str = "",
    verbosity: int = 0,
) -> dict[str, Union[float, bool, np.ndarray]]:
    ocp, stuff = export_cstr_ocp(dt=dt, N=N, x0=x0, x_ref=xr, u_ref=ur, rrlb=rrlb)
    if rrlb:
        rel_discrepancy = np.linalg.norm(x0 - xr) / np.linalg.norm(xr)
        epsilon = np.exp(rel_discrepancy) / 100.0
        print(f"epsilon = {epsilon}")
        P = solve_discrete_are(
            stuff["A"],
            stuff["B"],
            stuff["Q"] + epsilon * stuff["M_x"],
            stuff["R"] + epsilon * stuff["M_u"],
        )
        params = np.append(epsilon, P.ravel("F"))
        ocp.parameter_values = params
    else:
        epsilon = None
        P = None

    f_disc = Function("f_disc", [ocp.model.x, ocp.model.u], [ocp.model.disc_dyn_expr])

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_" + ocp.model.name + ".json"
    )

    Nsim = 350
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    x_sim = np.zeros((Nsim + 1, nx))
    u_sim = np.zeros((Nsim, nu))
    x_sim[0, :] = x0
    xcurrent = x0

    time_tot = np.ndarray((Nsim, 1))

    n_convergence = Nsim + 1
    for i in range(Nsim):
        # define initial guess for the solver
        xtpr = xcurrent
        for j in range(N):
            acados_ocp_solver.set(j, "x", xtpr)
            acados_ocp_solver.set(j, "u", ur)
            xtpr = f_disc(xtpr, ur).full().flatten()

        # set parameters
        if rrlb:
            P = solve_discrete_are(
                stuff["A"],
                stuff["B"],
                stuff["Q"] + epsilon * stuff["M_x"],
                stuff["R"] + epsilon * stuff["M_u"],
            )
            params = np.append(epsilon, P.ravel("F"))
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
        rel_discrepancy = np.linalg.norm(xcurrent - xr) / np.linalg.norm(xr)
        epsilon = np.exp(rel_discrepancy) / 100.0
        print(f"epsilon = {epsilon}")
        if rel_discrepancy < 1e-3:
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
