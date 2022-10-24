import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_state"]


def plot_state(X: np.ndarray):
    """
    Plots 2 views of the current state of the system: a view from the top (x and y) and
     a view from the side (x and z).

    :param X: the state of the system, should at least contain the positions of the
        masses x1, x2, ..., xM, xM+1
    :type X: np.ndarray
    """
    M = len(X) // 6
    X = np.append(np.zeros(3), X)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(X[0 : 3 * (M + 2) : 3], X[1 : 3 * (M + 2) : 3], "o-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(X[0 : 3 * (M + 2) : 3], X[2 : 3 * (M + 2) : 3], "o-")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid(True)
    plt.tight_layout()
