import numpy as np
from acados_template import AcadosOcp
from casadi import *
from scipy.linalg import solve_discrete_are

from cstr_model import *

__all__ = ["export_cstr_ocp"]


def export_cstr_ocp(
    dt: float = 20 / 3600,
    N: int = 100,
    x0: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    rrlb: bool = True,
):
    ocp = AcadosOcp()
    ocp.model = export_cstr_model(dt=dt)
    ocp.dims.N = N
    ocp.solver_options.tf = dt * N

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    # ny = nx + nu
    # ny_e = nx
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

    # compute rrlb functions
    if rrlb:
        # compute the relaxation parameter delta
        d_x_tilde = d_x - C_x @ xr
        delta_x = np.min(np.abs(d_x_tilde))
        d_u_tilde = d_u - C_u @ ur
        delta_u = np.min(np.abs(d_u_tilde))
        delta = 0.5 * np.min([delta_x, delta_u])

        # define the function beta
        z = SX.sym("z")
        beta = Function(
            "beta",
            [z],
            [0.5 * (((z - 2 * delta) / delta) ** 2 - 1) - log(delta)],
        )

        # compute the individual functions B_x_i
        tpr = []
        for i in range(q_x):
            z_i = d_x[i] - dot(C_x[i, :], x)
            tpr.append(if_else(z_i > delta, log(d_x_tilde[i]) - log(z_i), beta(z_i)))

        # compute the weight vector w_x
        w_x = np.ones(q_x)
        w_x[nx:] = d_x_tilde[nx:] / d_x_tilde[:nx]

        # assemble the RRLB function B_x
        B_x = Function("B_x", [x], [dot(w_x, vertcat(*tpr))])
        hess_B_x = Function("hess_B_x", [x], [hessian(B_x(x), x)[0]])
        M_x = hess_B_x(xr).full()

        # compute the individual functions B_u_i
        tpr = []
        for i in range(q_u):
            z_i = d_u[i] - dot(C_u[i, :], u)
            tpr.append(if_else(z_i > delta, log(d_u_tilde[i]) - log(z_i), beta(z_i)))

        # compute the weight vector w_u
        w_u = np.ones(q_u)
        w_u[nu:] = d_u_tilde[nu:] / d_u_tilde[:nu]

        # assemble the RRLB function B_u
        B_u = Function("B_u", [u], [dot(w_u, vertcat(*tpr))])
        hess_B_u = Function("hess_B_u", [u], [hessian(B_u(u), u)[0]])
        M_u = hess_B_u(ur).full()

    else:
        B_x = None
        B_u = None
        M_x = None
        M_u = None

    # cost function
    l = Function("l", [x, u], [bilin(Q, x - xr, x - xr) + bilin(R, u - ur, u - ur)])
    jac_f_x = Function("jac_f_x", [x, u], [jacobian(ocp.model.disc_dyn_expr, x)])
    jac_f_u = Function("jac_f_u", [x, u], [jacobian(ocp.model.disc_dyn_expr, u)])
    A = np.array(jac_f_x(xr, ur))
    B = np.array(jac_f_u(xr, ur))
    if rrlb:
        P = np.array(solve_discrete_are(A, B, Q + epsilon * M_x, R + epsilon * M_u))
        l_tilde = Function(
            "l_tilde",
            [x, u],
            [l(x, u) + epsilon * B_x(x) + epsilon * B_u(u)],
        )
    else:
        P = np.array(solve_discrete_are(A, B, Q, R))
        l_tilde = None

    F = Function("F", [x], [bilin(P, (x - xr), (x - xr))])
    ocp.cost.cost_expr = l_tilde if rrlb else l
    ocp.cost.cost_expr_e = F

    # constraints
    ocp.constraints.x0 = x0
    if not rrlb:
        ocp.constraints.lbx = d_x[:nx]
        ocp.constraints.ubx = d_x[nx:]
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbu = d_u[:nu]
        ocp.constraints.ubu = d_u[nu:]
        ocp.constraints.idxbu = np.arange(nu)

    # solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    return ocp
