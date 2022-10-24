from acados_template import AcadosModel
import casadi as ca


def export_model(
    name: str,
    x: ca.SX,
    u: ca.SX,
    f_cont: ca.Function,
    dt: float,
    num_rk4_nodes: int = 10,
):
    new_x = x
    for j in range(num_rk4_nodes):
        k1 = f_cont(new_x, u)
        k2 = f_cont(new_x + dt / 2 * k1, u)
        k3 = f_cont(new_x + dt / 2 * k2, u)
        k4 = f_cont(new_x + dt * k3, u)
        new_x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    xdot = ca.SX.sym("xdot", x.shape)
    model = AcadosModel()
    model.f_impl_expr = xdot - f_cont(x, u)
    model.f_expl_expr = f_cont(x, u)
    model.disc_dyn_expr = new_x
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = name

    return model
