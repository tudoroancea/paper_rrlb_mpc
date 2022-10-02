from typing import Union, Optional

import numpy as np
import tqdm
from acados_template import AcadosOcpSolver
from casadi import Function
from scipy.linalg import solve_discrete_are

from cstr import cstr_model
from cstr import cstr_ocp
from cstr import cstr_plot

__all__ = ["run_closed_loop_simulation"]

# options for CSTR: dt, N, rrlb


def run_closed_loop_simulation(
    problem: str,
    dt: float = None,
    N: int = None,
    M: int = None,  # only useful for Mass chain problem
    Nsim: int = None,
    x0: np.ndarray = None,  # only useful for CSTR problem
    xr: np.ndarray = None,
    ur: np.ndarray = None,
    rrlb: bool = True,
    generate_code: bool = True,
    build_solver: bool = True,
    show_plot: bool = True,
    plot_filename: str = "",
) -> dict[str, Union[float, bool, np.ndarray]]:
    # check input data and give default values
    assert problem in {"cstr", "mass_chain"}
    if problem == "cstr":
        if dt is None:
            dt = 20 / 3600
        if N is None:
            N = 100
        if Nsim is None:
            Nsim = 40
        if x0 is None:
            x0 = np.array([1.0, 0.5, 100.0, 100.0])
        if xr is None:
            xr = cstr_model.xr1
        if ur is None:
            ur = cstr_model.ur1
    elif problem == "mass_chain":
        if dt is None:
            dt = 0.2
        if N is None:
            N = 40
        if M is None:
            M = 5
        if Nsim is None:
            Nsim = 40
        if x0 is None:
            x0 = np.array([1.0, 1.0])
        if xr is None:
            xr = np.array([0.0, 0.0])
        if ur is None:
            ur = np.array([0.0])

    # create ocp object to formulate the OCP
    if problem == "cstr":
        ocp, stuff = cstr_ocp.export_cstr_ocp(
            dt=dt, N=N, x0=x0, x_ref=xr, u_ref=ur, rrlb=rrlb
        )
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # extract information from ocp (dimensions, dynamics, matrices, etc...)
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
    if problem == "cstr":
        xcurrent = x0
    else:
        xcurrent = f_disc(x0, np.array([-1.0, 1.0, 1.0]))

    last_prediction = np.zeros((N + 1, nx + nu))
    n_convergence = Nsim + 1
    sim_data = {
        "x_sim": [x0],
        "u_sim": [],
        "time_tot": [],
        "epsilon": [],
    }

    # compute the first runtime parameters for the RRLB MPC (barrier parameter epsilon
    # and terminal cost P)
    def compute_runtime_parameters(iteration: Optional[int] = None):
        epsilon = 30.0 * 0.5**iteration
        P = solve_discrete_are(A, B, Q + epsilon * M_x, R + epsilon * M_u)
        return np.append(epsilon, P.ravel("F"))

    if rrlb:
        ocp.parameter_values = np.zeros(1 + nx * nx)

    # create an acados ocp solver
    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file=ocp.model.name + "_ocp_" + ("rrlb" if rrlb else "reg") + ".json",
        generate=generate_code,
        build=build_solver,
    )

    # control loop
    for i in tqdm.trange(Nsim):
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

        # update simulation data
        sim_data["u_sim"].append(acados_ocp_solver.get(0, "u"))
        xcurrent = f_disc(xcurrent, sim_data["u_sim"][-1]).full().flatten()
        sim_data["x_sim"].append(xcurrent)
        sim_data["time_tot"].append(acados_ocp_solver.get_stats("time_tot"))

        # check if there is convergence in relative norm
        if (
            np.linalg.norm(xcurrent - xr) / np.linalg.norm(xr) < 1.0e-6
            and n_convergence == Nsim + 1
        ):
            n_convergence = i + 1
            # break

    # create np.ndarrays for the simulation data
    for key, val in sim_data.items():
        if isinstance(val, list):
            sim_data[key] = np.array(val)

    # plot data
    if problem == "cstr":
        cstr_plot.plot_cstr(
            ocp=ocp,
            x_ref=xr,
            u_ref=ur,
            x_sim=sim_data["x_sim"],
            u_sim=sim_data["u_sim"],
            dt=dt * 3600,
            file_name=plot_filename,
            show=show_plot,
        )

    sim_data["n_convergence"] = n_convergence
    return sim_data
