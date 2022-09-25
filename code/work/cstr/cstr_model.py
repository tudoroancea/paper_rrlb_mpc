import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, exp, Function

__all__ = ["export_cstr_model", "ur1", "ur2", "ur3", "xr1", "xr2", "xr3"]

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
    model.f_expl_expr = f_cont(x, u)
    model.disc_dyn_expr = new_x
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = "cstr"

    return model
