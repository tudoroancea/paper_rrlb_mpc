import numpy as np
from cstr_model import *
from cstr_ocp import *
from cstr_plot import *
from acados_template import AcadosOcpSolver, AcadosSimSolver


def run_closed_loop_simulation(
    dt: float = 20 / 3600,
    N: int = 100,
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    rrlb: bool = True,
):
    ocp = export_cstr_ocp(dt=dt, N=N, x0=x0, x_ref=xr, u_ref=ur, rrlb=rrlb)
    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_" + ocp.model.name + ".json"
    )
    acados_sim_solver = AcadosSimSolver(
        ocp, json_file="acados_sim_" + ocp.model.name + ".json"
    )

    Nsim = 350
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    x_sim = np.zeros((Nsim + 1, nx))
    u_sim = np.zeros((Nsim, nu))
    x_sim[0, :] = x0

    xcurrent = x0

    n_convergence = Nsim + 1
    for i in range(Nsim):
        # solve ocp
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        status = acados_ocp_solver.solve()
        acados_ocp_solver.print_statistics()
        if status != 0:
            raise Exception(
                "acados ocp solver returned status {}. Exiting.".format(status)
            )

        u_sim[i, :] = acados_ocp_solver.get(0, "u")

        # simulate system
        acados_sim_solver.set("x", xcurrent)
        acados_sim_solver.set("u", u_sim[i, :])
        status = acados_sim_solver.solve()
        if status != 0:
            raise Exception(
                "acados sim solver returned status {}. Exiting.".format(status)
            )

        # update state
        xcurrent = acados_sim_solver.get("x")
        x_sim[i + 1, :] = xcurrent

        # check if there is convergence in relative norm
        if np.linalg.norm(x_sim[i + 1, :] - xr) / np.linalg.norm(xr) < 1e-3:
            n_convergence = i + 1
            break

    x_sim = x_sim[: n_convergence + 1, :]
    u_sim = u_sim[:n_convergence, :]

    # plot data
    plot_cstr(
        ocp,
        xr,
        ur,
        x_sim,
        u_sim,
        dt * 3600,
        # file_name="cstr_closed_loop_simulation",
        show=True,
    )

    return x_sim, u_sim, n_convergence
