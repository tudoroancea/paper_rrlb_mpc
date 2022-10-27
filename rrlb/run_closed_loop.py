import io
from contextlib import redirect_stdout
from time import perf_counter
from typing import Union, Optional

import numpy as np
from acados_template import AcadosOcpSolver
from casadi import Function
from scipy.linalg import solve_discrete_are
from tqdm import trange

from .cstr import *
from .mass_chain import *

__all__ = ["run_closed_loop_simulation"]


def run_closed_loop_simulation(
    problem: str,
    problem_params: dict,
    rrlb: bool = True,
    rrlb_params: Optional[dict] = None,
    verbose: bool = False,
    generate_code: bool = True,
    build_solver: bool = True,
    plot: bool = True,
    show_plot: bool = True,
    plot_filename: str = "",
) -> dict[str, Union[float, bool, np.ndarray]]:
    """

    :param problem:
    :param problem_params: should contain everything describing the problem: sampling time,
    horizon size, number of iterations, reference points, initial state, etc.
    :param rrlb:
    :param rrlb_params:
    :param verbose: whether to print stuff or not
    :param generate_code: whether to generate code or not
    :param build_solver:
    :param show_plot:
    :param plot_filename:
    :return: sim_data contains the following keys:
        x_sim: the state trajectory
        u_sim: the control trajectory
        time_tot: the total solver run time
        epsilon: the value of epsilon used in the RRLB at each iteration
        n_convergence: the number of iterations needed to converge
        discrepancies: the discrepancies between the reference and current state at each
            iteration
        constraint_violations: the constraint violations at each iteration
        performance_measure: the performance measure at each iteration (quadratic costs
            on state and control)
    """
    # check inputs ================================================================
    assert problem in {"cstr", "mass_chain"}
    if rrlb:
        assert rrlb_params is not None
        assert "fun" in rrlb_params or (
            "epsilon_0" in rrlb_params and "epsilon_rate" in rrlb_params
        )

    # set up problem ==============================================================
    # create ocp object to formulate the OCP
    if problem == "cstr":
        dt = problem_params.get("dt", 20 / 3600)
        N = problem_params.get("N", 100)
        Nsim = problem_params.get("Nsim", 40)
        x_ref_default, u_ref_default = cstr_model.find_cstr_steady_state(1)
        x_ref = problem_params.get("x_ref", x_ref_default)
        u_ref = problem_params.get("u_ref", u_ref_default)
        xinit = problem_params.get("xinit", np.array([1.0, 0.5, 100.0, 100.0]))
        ocp, stuff = cstr_ocp.export_cstr_ocp(
            dt=dt, N=N, x_ref=x_ref, u_ref=u_ref, rrlb=rrlb
        )
    elif problem == "mass_chain":
        dt = problem_params.get("dt", 0.2)
        N = problem_params.get("N", 40)
        Nsim = problem_params.get("Nsim", 40)
        M = problem_params.get("M", 9)
        x_last = problem_params.get("x_last", np.array([1.0, 0.0, 0.0]))
        x_ref = mass_chain_model.find_mass_chain_steady_state(M=M, x_last=x_last)
        u_ref = np.zeros(3)
        xinit = x_ref
        ocp, stuff = mass_chain_ocp.export_mass_chain_ocp(
            dt=dt, N=N, M=M, x_last=x_last, rrlb=rrlb
        )
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # extract information from ocp (dimensions, dynamics, matrices, etc...)
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    f_disc_ca = Function(
        "f_disc", [ocp.model.x, ocp.model.u], [ocp.model.disc_dyn_expr]
    )
    f_disc = lambda x, u: f_disc_ca(x, u).full().flatten()
    Q = stuff["Q"]
    R = stuff["R"]
    A = stuff["A"]
    B = stuff["B"]
    M_x = stuff["M_x"]
    M_u = stuff["M_u"]
    x_ub = stuff["x_ub"]
    x_lb = stuff["x_lb"]
    u_ub = stuff["u_ub"]
    u_lb = stuff["u_lb"]

    # for the mass chain problem, perturb the initial state with a control [-1,1,1] for
    # 5 sampling times
    if problem == "mass_chain":
        for i in range(5):
            xinit = f_disc(xinit, np.array([-1.0, 1.0, 1.0]))

    # declare the variables that will contain the simulation data
    xcurrent = xinit
    last_prediction = np.zeros((N + 1, nx + nu))
    n_convergence = Nsim + 1
    performance_measure = 0.0
    sim_data = {
        "x_sim": [xinit],
        "u_sim": [],
        "time_tot": [],
        "epsilon": [],
        "discrepancies": [],
        "constraint_violations": [],
    }
    sim_data["discrepancies"].append(np.linalg.norm(xcurrent - x_ref))

    # compute the first runtime parameters for the RRLB MPC (barrier parameter epsilon
    # and terminal cost P)
    def compute_runtime_parameters(iteration: int) -> np.ndarray:
        if rrlb:
            if "fun" in rrlb_params:
                return rrlb_params["fun"](iteration)
            else:
                epsilon_0 = rrlb_params.get("epsilon_0", 50.0)
                epsilon = epsilon_0 * rrlb_params.get("epsilon_rate", 1.0) ** iteration

            P = solve_discrete_are(A, B, Q + epsilon * M_x, R + epsilon * M_u)
            return np.append(epsilon, P.ravel("F"))
        else:
            return np.array([])

    if rrlb:
        ocp.parameter_values = np.zeros(nx**2 + 1)

    # create an acados ocp solver
    start = perf_counter()
    with io.StringIO() as buffer, redirect_stdout(buffer):
        acados_ocp_solver = AcadosOcpSolver(
            ocp,
            json_file=ocp.model.name + "_ocp_" + ("rrlb" if rrlb else "reg") + ".json",
            generate=generate_code,
            build=build_solver,
        )
    end = perf_counter()
    if generate_code or build_solver:
        print(f"acados ocp solver built in {end - start:.2f} seconds")

    # run the control loop ========================================================
    if verbose:
        print("Running closed-loop simulation...")

    # for i in trange(Nsim):
    for i in range(Nsim):
        # define initial guess for the solver
        if i == 0:
            # apply the reference control ur to the current state xcurrent
            xtpr = xcurrent
            for j in range(N):
                acados_ocp_solver.set(j, "x", xtpr)
                acados_ocp_solver.set(j, "u", u_ref)
                xtpr = f_disc(xtpr, u_ref)
            acados_ocp_solver.set(N, "x", xtpr)
        else:
            # shift the last prediction
            for j in range(N):
                acados_ocp_solver.set(j, "x", last_prediction[j + 1, :nx])
                acados_ocp_solver.set(j, "u", last_prediction[j + 1, nx:])
            acados_ocp_solver.set(
                N,
                "x",
                f_disc(last_prediction[-1, :nx], last_prediction[-1, nx:]),
            )

        # set runtime parameters in solver
        if rrlb:
            p = compute_runtime_parameters(iteration=i)
            sim_data["epsilon"].append(p[0])
            for j in range(N + 1):
                acados_ocp_solver.set(j, "p", p)

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
        last_prediction[N, :] = np.append(acados_ocp_solver.get(N, "x"), u_ref)
        if np.any(np.isnan(last_prediction)):
            raise ValueError("NaNs in prediction. Exiting.")

        # update simulation data
        sim_data["u_sim"].append(acados_ocp_solver.get(0, "u"))
        xcurrent = f_disc(xcurrent, sim_data["u_sim"][-1])
        sim_data["x_sim"].append(xcurrent)
        sim_data["time_tot"].append(acados_ocp_solver.get_stats("time_tot")[0])

        # update performance measure
        performance_measure += np.dot(
            np.dot(sim_data["u_sim"][-1] - u_ref, R), sim_data["u_sim"][-1] - u_ref
        ) + np.dot(
            np.dot(sim_data["x_sim"][-2] - x_ref, Q), sim_data["x_sim"][-2] - x_ref
        )

        # update constraint violation
        sim_data["constraint_violations"].append(
            np.linalg.norm(np.maximum(sim_data["x_sim"][-2] - x_ub, 0.0))
            + np.linalg.norm(np.maximum(x_lb - sim_data["x_sim"][-2], 0.0))
            + np.linalg.norm(np.maximum(sim_data["u_sim"][-1] - u_ub, 0.0))
            + np.linalg.norm(np.maximum(u_lb - sim_data["u_sim"][-1], 0.0))
        )

        # check if there is convergence in relative norm
        sim_data["discrepancies"].append(np.linalg.norm(xcurrent - x_ref))
        if sim_data["discrepancies"][-1] < problem_params.get("convergence_tol", 1e-4):
            n_convergence = i + 1
            if verbose:
                print("Convergence reached at iteration {}".format(n_convergence))
            break

    if verbose:
        print("Simulation finished.")

    # post-process the simulation data =============================================
    # create np.ndarrays for the simulation data
    for key, val in sim_data.items():
        if isinstance(val, list):
            sim_data[key] = np.array(val)

    # plot closed-loop trajectories data
    if plot:
        if problem == "cstr":
            cstr_plot.plot_cstr(
                ocp=ocp,
                x_ref=x_ref,
                u_ref=u_ref,
                x_sim=sim_data["x_sim"],
                u_sim=sim_data["u_sim"],
                dt=dt * 3600,
                file_name=plot_filename,
                show=show_plot,
            )

    sim_data["n_convergence"] = n_convergence
    sim_data["performance_measure"] = performance_measure
    return sim_data
