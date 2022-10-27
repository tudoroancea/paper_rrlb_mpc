import numpy as np
from acados_template import AcadosOcp, AcadosModel
import casadi as ca


def export_ocp(
    model: AcadosModel,
    Q: np.ndarray,
    R: np.ndarray,
    C_x: np.ndarray,
    C_u: np.ndarray,
    d_x: np.ndarray,
    d_u: np.ndarray,
    dt: float,
    N: int,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    rrlb: bool,
) -> tuple[AcadosOcp, dict[str, np.ndarray]]:
    """
    Export the CSTR OCP for the specified parameters.

    :param model:
    :param Q:
    :param R:
    :param C_x:
    :param C_u:
    :param d_x:
    :param d_u:
    :param dt: sampling time
    :type dt: float
    :param N: number of control intervals / horizon length
    :type N: int
    :param x_ref: reference state
    :type x_ref: np.ndarray
    :param u_ref: reference control
    :type u_ref: np.ndarray
    :param rrlb: whether to use RRLB MPC or not
    :type rrlb: bool

    :return: the AcadosOcp instance and some useful matrices computed inside (A, B, Q, R, M_x, M_u)
    """
    # create ocp object to formulate the OCP ======================================
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.solver_options.tf = dt * N

    # retrieve the state and control variables ====================================
    x = ocp.model.x
    u = ocp.model.u
    nx = x.size()[0]
    nu = u.size()[0]
    q_x = C_x.shape[0]
    q_u = C_u.shape[0]

    # check dimensions ===========================================================
    assert Q.shape == (nx, nx)
    assert R.shape == (nu, nu)
    assert C_x.shape == (q_x, nx)
    assert C_u.shape == (q_u, nu)
    assert d_x.shape == (q_x,)
    assert d_u.shape == (q_u,)
    assert x_ref.shape == (nx,)
    assert u_ref.shape == (nu,)

    # compute rrlb functions ====================================================
    if rrlb:
        # compute the relaxation parameter delta
        d_x_tilde = d_x - C_x @ x_ref
        delta_x = np.min(np.abs(d_x_tilde))
        d_u_tilde = d_u - C_u @ u_ref
        delta_u = np.min(np.abs(d_u_tilde))
        delta = 0.5 * np.min([delta_x, delta_u])

        # define the function beta
        z = ca.SX.sym("z")
        beta = ca.Function(
            "beta",
            [z],
            [0.5 * (((z - 2 * delta) / delta) ** 2 - 1) - ca.log(delta)],
        )

        B_x, M_x = create_rrlb_function(x, x_ref, C_x, d_x, d_x_tilde, q_x, beta, delta)
        B_u, M_u = create_rrlb_function(u, u_ref, C_u, d_u, d_u_tilde, q_u, beta, delta)

    else:
        B_x = None
        B_u = None
        M_x = None
        M_u = None

    # cost function ==============================================================
    jac_f_x = ca.Function("jac_f_x", [x, u], [ca.jacobian(ocp.model.disc_dyn_expr, x)])
    jac_f_u = ca.Function("jac_f_u", [x, u], [ca.jacobian(ocp.model.disc_dyn_expr, u)])
    A = np.array(jac_f_x(x_ref, u_ref))
    B = np.array(jac_f_u(x_ref, u_ref))

    if rrlb:
        epsilon = ca.SX.sym("epsilon")

        # lagrange cost (quadratic costs + RRLB functions)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            ca.bilin(Q, x - x_ref, x - x_ref)
            + ca.bilin(R, u - u_ref, u - u_ref)
            + epsilon * B_x(x)
            + epsilon * B_u(u)
        )

        # mayer cost (quadratic costs only)
        ocp.cost.cost_type_e = "EXTERNAL"
        P = ca.SX.sym("P", nx, nx)
        ocp.model.cost_expr_ext_cost_e = ca.bilin(P, x - x_ref, x - x_ref)

        # declare runtime params in ocp model
        ocp.model.p = ca.vertcat(epsilon, ca.reshape(P, nx * nx, 1))
    else:
        # lagrange cost (quadratic costs + RRLB functions)
        # ocp.cost.cost_type = "LINEAR_LS"
        # ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        # ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
        # ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
        # ocp.cost.yref = np.append(x_ref, u_ref)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = ca.bilin(Q, x - x_ref, x - x_ref) + ca.bilin(
            R, u - u_ref, u - u_ref
        )

        # mayer cost (quadratic costs only)
        # ocp.cost.cost_type_e = "EXTERNAL"
        # P = ca.SX.sym("P", nx, nx)
        # ocp.model.cost_expr_ext_cost_e = ca.bilin(P, x - x_ref, x - x_ref)

    # constraints
    ocp.constraints.x0 = np.zeros(nx)

    if not rrlb:
        ocp.constraints.ubx = d_x[:nx]
        ocp.constraints.lbx = -d_x[nx:]
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.ubx_e = d_x[:nx]
        ocp.constraints.lbx_e = -d_x[nx:]
        ocp.constraints.idxbx_e = np.arange(nx)
        ocp.constraints.ubu = d_u[:nu]
        ocp.constraints.lbu = -d_u[nu:]
        ocp.constraints.idxbu = np.arange(nu)

    # solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE" if rrlb else "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.print_level = 0
    ocp.code_export_directory = model.name + "_gen_code_" + ("rrlb" if rrlb else "reg")

    return ocp, {
        "A": A,
        "B": B,
        "Q": Q,
        "R": R,
        "M_x": M_x,
        "M_u": M_u,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "x_ub": d_x[:nx],
        "x_lb": -d_x[nx:],
        "u_ub": d_u[:nu],
        "u_lb": -d_u[nu:],
    }


def create_rrlb_function(
    var: ca.SX,
    ref: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    d_tilde: np.ndarray,
    q: int,
    beta: ca.Function,
    delta: float,
) -> tuple[ca.Function, np.ndarray]:
    """Create the RRLB function for a given variable."""
    n = var.shape[0]
    assert ref.shape == (n,)
    assert C.shape == (q, n)
    assert d.shape == (q,)
    assert d_tilde.shape == (q,)
    assert delta > 0

    # compute the individual functions B_i
    tpr = []
    for i in range(q):
        z_i = d[i] - ca.dot(C[i, :], var)
        tpr.append(
            ca.if_else(
                z_i > delta,
                ca.log(d_tilde[i]) - ca.log(z_i),
                ca.log(d_tilde[i]) + beta(z_i),
            )
        )

    # compute the weight vector w
    w = np.append(d_tilde[:n] / d_tilde[n:], d_tilde[n:] / d_tilde[:n])

    # assemble the RRLB function B
    B = ca.Function("B", [var], [ca.dot(w + np.ones(q), ca.vertcat(*tpr))])
    grad_B = ca.Function("grad_B", [var], [ca.jacobian(B(var), var)])
    assert np.allclose(grad_B(ref), np.zeros(n))
    hess_B = ca.Function("hess_B", [var], [ca.hessian(B(var), var)[0]])
    M = hess_B(ref).full()

    return B, M
