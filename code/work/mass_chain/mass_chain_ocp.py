from acados_template import AcadosOcp
from casadi import *

from ..common import export_ocp
from .mass_chain_model import find_mass_chain_steady_state, export_mass_chain_model

__all__ = ["export_mass_chain_ocp"]


def export_mass_chain_ocp(
    dt: float = 0.2,
    N: int = 40,
    M: int = 9,
    x_last: np.ndarray = np.array([1.0, 0.0, 0.0]),
    rrlb: bool = True,
) -> tuple[AcadosOcp, dict[str, np.ndarray]]:
    """
    Export the Mass Chain OCP for the specified parameters.

    :param dt: sampling time
    :type dt: float
    :param N: number of control intervals / horizon length
    :type N: int
    :param M: number of (intermediate) chained masses
    :type M: int
    :param x_last: position of the last mass
    :type x_last: np.ndarray
    :param rrlb: whether to use RRLB MPC or not
    :type rrlb: bool

    :return: the AcadosOcp instance and some useful matrices computed inside (A, B, Q, R, M_x, M_u)
    """
    # problem dimensions
    nx = 3 * (2 * M + 1)
    nu = 3

    # constants for the mass chain model
    beta = 25.0  # [s^-2]
    gamma = 1.0
    delta_R = 0.01

    # costs
    Q = np.zeros((nx, nx))
    Q[-3 * (M + 1) :, -3 * (M + 1) :] = np.array([[beta] + [gamma] * 3 * M])
    R = delta_R * np.eye(nu)

    # reference state and control
    x_ref = find_mass_chain_steady_state(M=M, x_last=x_last)
    u_ref = np.zeros(nu)

    # constraints
    z_min = np.min(x_ref[2 : 3 * (M + 1) : 3])
    C_x = np.vstack((np.eye(nx), -np.eye(nx)))
    C_u = np.vstack((np.eye(nu), -np.eye(nu)))
    d_x = np.zeros(2 * nx)
    d_x[: 3 * (M + 1)] = np.tile([1.0, 1.0, 1.5], (M + 1))
    d_x[nx : nx + 3 * (M + 1)] = np.tile([0.1, 0.1, 5.0 - z_min], (M + 1))
    d_x[3 * (M + 1) : nx] = np.tile([2.0, 2.0, 2.0], M)
    d_x[-3 * M :] = np.tile([2.0, 2.0, 2.0], M)

    d_u = np.ones(2 * nu)

    return export_ocp(
        model=export_mass_chain_model(dt=dt, M=M),
        Q=Q,
        R=R,
        C_x=C_x,
        d_x=d_x,
        C_u=C_u,
        d_u=d_u,
        dt=dt,
        N=N,
        x_ref=x_ref,
        u_ref=u_ref,
        rrlb=rrlb,
    )
