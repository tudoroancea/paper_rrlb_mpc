import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import AxesZero


def beta(x: float, delta: float):
    return 0.5 * ((x - 2 * delta) ** 2 / (delta**2) - 1) - np.log(delta)


plt.style.use(["science", "ieee"])
plt.rcParams.update({"figure.dpi": "100", "font.size": 10})


fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(121, axes_class=AxesZero)
for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

x = np.linspace(1e-8, 1, 1000)
ax.plot(x, -np.log(x), "k-", linewidth=2)
x = np.linspace(-0.2, 0.1, 100)
ax.plot(x, [beta(a, 0.1) for a in x], "k-.")
ax.plot([0.1, 0.1], [0.0, 5.0], "k:")
ax.text(0.1, -0.5, r"$\delta$")
ax.set_ylim(-0.5, 5.0)
ax.set_xlim(-0.1, 0.7)


def B1(x: float, delta: float):
    if 2 - x > delta:
        return np.log(2) - np.log(2 - x)
    else:
        return np.log(2) + beta(2 - x, delta)


def B2(x: float, delta: float):
    if 1 + x > delta:
        return -np.log(1 + x)
    else:
        return beta(1 + x, delta)


def B(x, delta):
    return 3 * B1(x, delta) + 1.5 * B2(x, delta)


f1 = lambda x: B(x, 0.01)
f2 = lambda x: B(x, 0.1)
f3 = lambda x: B(x, 0.5)
f4 = lambda x: B(x, 1.0)


# plt.subplot(1, 2, 2)
ax = fig.add_subplot(122, axes_class=AxesZero)
for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)
delta = 0.1
x_vals = np.linspace(-2, 3, 100)
plt.plot(x_vals, [f1(x) for x in x_vals], label=r"$\delta=0.01$")
plt.plot(x_vals, [f2(x) for x in x_vals], label=r"$\delta=0.1$")
plt.plot(x_vals, [f3(x) for x in x_vals], label=r"$\delta=0.5$")
plt.plot(x_vals, [f4(x) for x in x_vals], label=r"$\delta=1.0$")
plt.plot([-1.0, -1.0], [-2, 20], "k:")
plt.plot([2.0, 2.0], [-2, 20], "k:")
plt.ylim([-2, 20])
# plt.xlabel(r"$x$")
# plt.ylabel(r"$B(x)$")
# legend in upper left corner
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("rrlb.png", dpi=300, bbox_inches="tight")
plt.show()
