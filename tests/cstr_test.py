import matplotlib.pyplot as plt
import numpy as np

from rrlb import run_closed_loop_simulation
from rrlb.cstr import find_cstr_steady_state


def main():
    x_ref, u_ref = find_cstr_steady_state(1)
    res_rrlb = run_closed_loop_simulation(
        problem="cstr",
        problem_params={
            "N": 10,
            "Nsim": 100,
            "dt": 20 / 3600,
            "x_ref": x_ref,
            "u_ref": u_ref,
            "xinit": np.array([7.93103448, 6.55172414, 130.27586207, 94.0]),
        },
        rrlb=True,
        rrlb_params={"epsilon_0": 0.0, "epsilon_rate": 1.0},
        plot=True,
        show_plot=False,
        # generate_code=False,
        # build_solver=False,
    )
    print(f"n_conv_rrlb: {res_rrlb['n_convergence']}")
    print(f"constraint violations: {np.sum(res_rrlb['constraint_violations'])}")
    plt.show()


if __name__ == "__main__":
    main()
