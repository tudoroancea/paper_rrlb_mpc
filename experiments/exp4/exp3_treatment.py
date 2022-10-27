import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def treatment(n: int, epsilon: float):
    initial_states = np.load("exp3_initial_states_{}_{}.npy".format(n, int(epsilon)))
    constraint_violations = np.load(
        "exp3_constraint_violations_{}_{}.npy".format(n, int(epsilon))
    )
    print(
        "proportion of NaNs in constraint violations: {}".format(
            np.count_nonzero(np.isnan(constraint_violations))
            / constraint_violations.size
        )
    )
    # find the initial states with NaN in constraint violations
    nan_indices = np.random.permutation(np.argwhere(np.isnan(constraint_violations)))[
        :10
    ]
    print("nan indices: {}".format(nan_indices))

    c_A_init = initial_states[:, 0, 0, 0, 0]
    c_B_init = initial_states[0, :, 0, 0, 1]
    theta_init = initial_states[0, 0, :, 0, 2]
    theta_K_init = initial_states[0, 0, 0, :, 3]

    xinit = np.array([4.0, 5.0, 120.0, 120.0])
    # find the index of the initial state that is the closest to np.array([1.0, 0.5, 100.0, 100.0])
    idx = np.argmin(
        np.linalg.norm(
            initial_states
            - np.expand_dims(
                xinit,
                axis=(0, 1, 2, 3),
            ),
            axis=4,
        )
    )
    c_A_id, c_B_id, theta_id, theta_K_id = np.unravel_index(
        idx, constraint_violations.shape
    )
    c_B_id += 1
    print(
        "c_A_id ={}\nc_B_id={}\ntheta_id={}\ntheta_K_id={}".format(
            c_A_id,
            c_B_id,
            theta_id,
            theta_K_id,
        )
    )
    print(initial_states[c_A_id, c_B_id, theta_id, theta_K_id, :])

    # extract 2D slice from 4D array and make a scatter plot of constraint violations vs. initial states
    # theta_id = 8
    # theta_K_id = 6
    plt.subplots(2, 3)
    plt.suptitle(r"$\epsilon={}$".format(epsilon))
    # theta, theta_K fixed
    plt.subplot(2, 3, 1)
    I = initial_states[:, :, theta_id, theta_K_id, [0, 1]]
    C = constraint_violations[:, :, theta_id, theta_K_id]
    plt.scatter(
        I[:, :, 0].ravel(),
        I[:, :, 1].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_A$")
    plt.ylabel(r"$c_B$")
    plt.title(
        r"$\theta={}, \theta_K={}$".format(
            np.round(theta_init[theta_id], 2), np.round(theta_K_init[theta_K_id], 2)
        )
    )

    # theta, c_B fixed
    plt.subplot(2, 3, 2)
    I = initial_states[:, c_B_id, theta_id, :, [0, 3]]
    C = constraint_violations[:, c_B_id, theta_id, :]
    plt.scatter(
        I[0, :, :].ravel(),
        I[1, :, :].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_A$")
    plt.ylabel(r"$\theta$_K")
    plt.title(
        r"$c_B={}, \theta={}$".format(
            np.round(c_B_init[c_B_id], 2), np.round(theta_init[theta_id], 2)
        )
    )

    # theta, c_A fixed
    plt.subplot(2, 3, 3)
    I = initial_states[c_A_id, :, theta_id, :, [1, 3]]
    C = constraint_violations[c_A_id, :, theta_id, :]
    plt.scatter(
        I[0, :, :].ravel(),
        I[1, :, :].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_B$")
    plt.ylabel(r"$\theta_K$")
    plt.title(
        r"$c_A={}, \theta={}$".format(
            np.round(c_A_init[c_A_id], 2), np.round(theta_init[theta_id], 2)
        )
    )

    # theta_K, c_A fixed
    plt.subplot(2, 3, 4)
    I = initial_states[c_A_id, :, :, theta_K_id, [1, 2]]
    C = constraint_violations[c_A_id, :, :, theta_K_id]
    plt.scatter(
        I[0, :, :].ravel(),
        I[1, :, :].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_B$")
    plt.ylabel(r"$\theta$")
    plt.title(
        r"$c_A={}, \theta_K={}$".format(
            np.round(c_A_init[c_A_id], 2), np.round(theta_K_init[theta_K_id], 2)
        )
    )

    # theta_K, c_B fixed
    plt.subplot(2, 3, 5)
    I = initial_states[:, c_B_id, :, theta_K_id, [0, 2]]
    C = constraint_violations[:, c_B_id, :, theta_K_id]
    plt.scatter(
        I[0, :, :].ravel(),
        I[1, :, :].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_A$")
    plt.ylabel(r"$\theta$")
    plt.title(
        r"$c_B={}, \theta_K={}$".format(
            np.round(c_B_init[c_B_id], 2), np.round(theta_K_init[theta_K_id], 2)
        )
    )

    # c_A, c_B fixed
    plt.subplot(2, 3, 6)
    I = initial_states[c_A_id, c_B_id, :, :, [2, 3]]
    C = constraint_violations[c_A_id, c_B_id, :, :]
    plt.scatter(
        I[0, :, :].ravel(),
        I[1, :, :].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\theta_K$")
    plt.title(
        r"$c_A={}, c_B={}$".format(
            np.round(c_A_init[c_A_id], 2), np.round(c_B_init[c_B_id], 2)
        )
    )

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2)


if __name__ == "__main__":
    treatment(10, 30.0)
    treatment(20, 30.0)
    # treatment(10, 50.0)
    plt.show()
