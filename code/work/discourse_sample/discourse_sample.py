from acados_template import *
from casadi import *
from scipy.linalg import solve_discrete_are

xr1 = np.array(
    [
        2.1402105301746182e00,
        1.0903043613077321e00,
        1.1419108442079495e02,
        1.1290659291045561e02,
    ]
)
ur1 = np.array([14.19, -1113.50])
xr2 = np.array([2.7151681, 1.02349152, 105.50585058, 100.8920758])
ur2 = np.array([13.66640639, -3999.58908628])
xr3 = np.array([2.97496989, 0.95384459, 101.14965441, 95.19386292])
ur3 = np.array([12.980148, -5162.95653026])


def export_cstr_model(dt: float, rk4_nodes: int = 6):
    c_A = SX.sym("c_A")
    c_B = SX.sym("c_B")
    theta = SX.sym("theta")
    theta_K = SX.sym("theta_K")
    u_1 = SX.sym("u_1")
    u_2 = SX.sym("u_2")
    x = vertcat(c_A, c_B, theta, theta_K)
    u = vertcat(u_1, u_2)
    c_A_dot = SX.sym("c_A_dot")
    c_B_dot = SX.sym("c_B_dot")
    theta_dot = SX.sym("theta_dot")
    theta_K_dot = SX.sym("theta_K_dot")
    xdot = vertcat(c_A_dot, c_B_dot, theta_dot, theta_K_dot)

    k_10 = 1.287e12  # [h^-1]
    k_20 = 1.287e12  # [h^-1]
    k_30 = 9.043e9  # [h^-1]
    E_1 = -9758.3  # [1]
    E_2 = -9758.3  # [1]
    E_3 = -8560.0  # [1]
    H_1 = 4.2  # [kJ.mol^-1]
    H_2 = -11.0  # [kJ.mol^-1]
    H_3 = -41.85  # [kJ.mol^-1]
    rho = 0.9342  # [kg.L^-1]
    C_p = 3.01  # [kJ/(kg.K)]
    k_w = 4032.0  # [kJ/(h.m^2.K)]
    A_R = 0.215  # [m^2]
    V_R = 10.0  # [l]
    m_K = 5.0  # [kg]
    C_PK = 2.0  # [kJ/(kg.K)]
    c_A0 = 5.1  # [mol/l]
    theta_0 = 104.9  # [°C]

    k_1 = k_10 * exp(E_1 / (x[2] + 273.15))
    k_2 = k_20 * exp(E_2 / (x[2] + 273.15))
    k_3 = k_30 * exp(E_3 / (x[2] + 273.15))

    f_cont = Function(
        "f_cont",
        [x, u],
        [
            vertcat(
                u[0] * (c_A0 - x[0]) - k_1 * x[0] - k_3 * x[0] * x[0],
                -u[0] * x[1] + k_1 * x[0] - k_2 * x[1],
                u[0] * (theta_0 - x[2])
                + (x[3] - x[2]) * k_w * A_R / (rho * C_p * V_R)
                - (k_1 * x[0] * H_1 + k_2 * x[1] * H_2 + k_3 * x[0] * x[0] * H_3)
                / (rho * C_p),
                (u[1] + k_w * A_R * (x[2] - x[3])) / (m_K * C_PK),
            )
        ],
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
    model.name = "cstr"

    return model


def export_cstr_ocp(
    dt: float = 20 / 3600,
    N: int = 100,
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    x_ref: np.ndarray = xr1,
    u_ref: np.ndarray = ur1,
):
    ocp = AcadosOcp()
    ocp.model = export_cstr_model(dt=dt)
    ocp.dims.N = N
    ocp.solver_options.tf = dt * N

    nx = ocp.model.x.size()[0]
    ocp.dims.nx = nx
    nu = ocp.model.u.size()[0]
    ocp.dims.nu = nu
    x = ocp.model.x
    u = ocp.model.u

    Q = np.diag([0.2, 1.0, 0.5, 0.2])
    R = np.diag([0.5, 5.0e-7])
    C_x = np.vstack((np.eye(nx), -np.eye(nx)))
    d_x = np.array([10.0, 10.0, 150.0, 150.0, 0.0, 0.0, -98.0, -92.0])
    q_x = 2 * nx
    C_u = np.vstack((np.eye(nu), -np.eye(nu)))
    d_u = np.array([35.0, 0.0, -3.0, 9000.0])
    q_u = 2 * nu
    epsilon = 1.0e-3

    # cost function
    jac_f_x = Function("jac_f_x", [x, u], [jacobian(ocp.model.disc_dyn_expr, x)])
    jac_f_u = Function("jac_f_u", [x, u], [jacobian(ocp.model.disc_dyn_expr, u)])
    A = np.array(jac_f_x(x_ref, u_ref))
    B = np.array(jac_f_u(x_ref, u_ref))
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    P = np.array(solve_discrete_are(A, B, Q, R))
    ocp.cost.W_e = P
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref = np.append(x_ref, u_ref)
    ocp.cost.yref_e = x_ref

    # constraints
    ocp.constraints.x0 = x0
    ocp.constraints.lbx = d_x[:nx]
    ocp.constraints.ubx = d_x[nx:]
    ocp.constraints.idxbx = np.arange(nx)
    ocp.constraints.lbu = d_u[:nu]
    ocp.constraints.ubu = d_u[nu:]
    ocp.constraints.idxbu = np.arange(nu)

    # solver options
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    return ocp


def run_closed_loop_simulation(
    dt: float = 20 / 3600,
    N: int = 100,
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
):
    ocp = export_cstr_ocp(dt=dt, N=N, x0=x0, x_ref=xr, u_ref=ur)
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

    return x_sim, u_sim, n_convergence


if __name__ == "__main__":
    run_closed_loop_simulation()
