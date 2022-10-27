import matplotlib.pyplot as plt
import numpy as np


def treatment(n: int, epsilon: float):
    initial_states = np.load("exp3_initial_states_{}_{}.npy".format(n, int(epsilon)))
    constraint_violations = np.load(
        "exp3_constraint_violations_{}_{}.npy".format(n, int(epsilon))
    )
    # c_A_init = initial_states[:, 0, 0]
    # c_B_init = initial_states[0, :, 0]
    print(
        "theta={}, theta_K={}".format(initial_states[0, 0, 2], initial_states[0, 0, 3])
    )

    # extract 2D slice from 4D array and make a scatter plot of constraint violations vs. initial states
    # theta, theta_K fixed
    I = initial_states[:, :, [0, 1]]
    C = constraint_violations
    plt.scatter(
        I[:, :, 0].ravel(),
        I[:, :, 1].ravel(),
        c=C.ravel(),
    )
    plt.colorbar()
    plt.xlabel(r"$c_A$")
    plt.ylabel(r"$c_B$")
    plt.title(r"$\epsilon={}$".format(epsilon))


if __name__ == "__main__":
    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"figure.dpi": "100"})
    fig, _ = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    plt.subplot(1, 2, 1)
    treatment(30, 1.0)
    plt.subplot(1, 2, 2)
    treatment(30, 10.0)
    plt.tight_layout()
    plt.savefig("exp3.png", dpi=300, format="png", bbox_inches="tight")
    plt.show()
