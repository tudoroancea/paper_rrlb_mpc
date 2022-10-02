from acados_template import AcadosOcp
from casadi import *

from ..common import export_ocp
from .cstr_model import export_cstr_model


__all__ = ["export_cstr_ocp"]


def export_cstr_ocp(
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    dt: float = 20 / 3600,
    N: int = 100,
    rrlb: bool = True,
) -> tuple[AcadosOcp, dict[str, np.ndarray]]:
    """
    Export the CSTR OCP for the specified parameters.

    :param dt: sampling time
    :type dt: float
    :param N: number of control intervals / horizon length
    :type N: int
    :param x0: initial state
    :type x0: np.ndarray
    :param x_ref: reference state
    :type x_ref: np.ndarray
    :param u_ref: reference control
    :type u_ref: np.ndarray
    :param rrlb: whether to use RRLB MPC or not
    :type rrlb: bool

    :return: the AcadosOcp instance and some useful matrices computed inside (A, B, Q, R, M_x, M_u)
    """

    return export_ocp(
        model=export_cstr_model(dt),
        Q=np.diag([0.2, 1.0, 0.5, 0.2]),
        R=np.diag([0.5, 5.0e-7]),
        C_x=np.vstack((np.eye(4), -np.eye(4))),
        d_x=np.array([10.0, 10.0, 150.0, 150.0, 0.0, 0.0, -98.0, -92.0]),
        C_u=np.vstack((np.eye(2), -np.eye(2))),
        d_u=np.array([35.0, 0.0, -3.0, 9000.0]),
        dt=dt,
        N=N,
        x_ref=x_ref,
        u_ref=u_ref,
        rrlb=rrlb,
    )
