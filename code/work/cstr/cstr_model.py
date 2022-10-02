import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, exp, Function

from ..common import export_model

__all__ = [
    "export_cstr_model",
    "find_cstr_steady_state",
]


def export_cstr_model(dt: float, num_rk4_nodes: int = 10) -> AcadosModel:
    """
    Create an AcadosModel for the CSTR model.

    :param dt: sampling time
    :type dt: float
    :param num_rk4_nodes: number of nodes for the Runge-Kutta 4 integrator
    :type num_rk4_nodes: int

    :return: the AcadosModel instance
    """
    c_A = SX.sym("c_A")
    c_B = SX.sym("c_B")
    theta = SX.sym("theta")
    theta_K = SX.sym("theta_K")
    u_1 = SX.sym("u_1")
    u_2 = SX.sym("u_2")
    x = vertcat(c_A, c_B, theta, theta_K)
    u = vertcat(u_1, u_2)

    # model constants
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

    return export_model("cstr", x, u, f_cont, dt, num_rk4_nodes)


def find_cstr_steady_state(num: int) -> tuple[np.ndarray, np.ndarray]:
    if num == 1:
        xr = np.array(
            [
                2.1402105301746182,
                1.0903043613077321,
                114.19108442079495,
                112.90659291045561,
            ]
        )
        ur = np.array([14.19, -1113.50])
    elif num == 2:
        xr = np.array([2.7151681, 1.02349152, 105.50585058, 100.8920758])
        ur = np.array([13.66640639, -3999.58908628])
    elif num == 3:
        xr = np.array([2.97496989, 0.95384459, 101.14965441, 95.19386292])
        ur = np.array([12.980148, -5162.95653026])
    else:
        raise ValueError(f"Invalid steady state number {num}")

    return xr, ur
