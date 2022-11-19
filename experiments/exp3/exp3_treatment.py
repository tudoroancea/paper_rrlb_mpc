import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style import *


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
    plt.imshow(
        np.transpose(
            constraint_violations + 1e-16 * np.ones_like(constraint_violations)
        ),
        origin="lower",
        cmap="turbo",
        norm="log",
    )
    # plt.imshow(constraint_violations.T, origin="lower", cmap="turbo", norm="linear")

    plt.colorbar()
    plt.xlabel(r"$c_A$")
    plt.ylabel(r"$c_B$")
    plt.axis("equal")
    plt.title(r"$\epsilon={}$".format(epsilon))


if __name__ == "__main__":
    print(available)
    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"figure.dpi": "100", "font.size": 12, "lines.markersize": 3})
    fig = plt.figure(figsize=(5, 2))
    plt.subplot(1, 2, 1)
    treatment(30, 1.0)
    plt.subplot(1, 2, 2)
    treatment(30, 10.0)
    plt.tight_layout(pad=0.0)
    plt.savefig("exp3.eps", dpi=300, format="eps", bbox_inches="tight")
    plt.savefig("exp3.png", dpi=300, format="png", bbox_inches="tight")
    plt.show()
