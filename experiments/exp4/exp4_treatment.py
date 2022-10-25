import numpy as np
import matplotlib.pyplot as plt


def plot_slice():
    plt.scatter()


if __name__ == "__main__":
    constraint_violations = np.load("exp4_constraint_violations.npy")
    initial_states = np.load("exp4_initial_states.npy")
    c_A_init = initial_states[:, 0, 0, 0, 0]
    c_B_init = initial_states[0, :, 0, 0, 1]
    theta_init = initial_states[0, 0, :, 0, 2]
    theta_K_init = initial_states[0, 0, 0, :, 3]
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].contourf(c_A_init, c_B_init, constraint_violations[:, :, 0, 0])
    ax[0, 1].contourf(c_A_init, theta_init, constraint_violations[:, 0, :, 0])
    ax[1, 0].contourf(c_A_init, theta_K_init, constraint_violations[:, 0, 0, :])
    ax[1, 1].contourf(c_B_init, theta_init, constraint_violations[0, :, :, 0])
    plt.show()
