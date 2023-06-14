import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.axisartist.axislines import AxesZero
import sys

plt.style.use(["science"])
plt.rcParams.update({"figure.dpi": "150", "font.size": 20})

lb = -1.0
ub = 1.5
delta = 0.2
xlim = (lb - 0.3, ub + 0.3)
ylim = (-0.5, 5.0)
wub = 1.0
wlb = -lb / ub * (1 + wub) - 1
#  wub, wlb = 2.0, 0.5


def beta(x: float, delta: float):
    return 0.5 * ((x - 2 * delta) ** 2 / (delta**2) - 1) - np.log(delta)


def Bub(x: float, delta: float = 0.0):
    if ub - x > delta:
        return np.log(ub) - np.log(ub - x)
    else:
        return np.log(ub) + beta(ub - x, delta)


def Blb(x: float, delta: float = 0.0):
    if x - lb > delta:
        return np.log(-lb) - np.log(x - lb)
    else:
        return np.log(-lb) + beta(x - lb, delta)


def B(x, delta):
    return (1 + wub) * Bub(x, delta) + (1 + wlb) * Blb(x, delta)


def create():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, axes_class=AxesZero)
    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([-1.0, -0.5, -0.05, 0.5, 1.0, 1.5])
    ax.set_xticklabels([r"$-1$", r"$-0.5$", r"$0$", r"$0.5$", r"$1$", r"$1.5$"])
    ax.set_yticks([])
    return fig, ax


# plot 1 : log-barrier function for -1<=z
print("Plot 1")
fig, ax = create()
x = np.linspace(lb + 1e-8, ub - 1e-8, 1000)
ax.plot(x, -np.log(x - lb), "b-", linewidth=2, label=rf"$B_1(z)$")
ax.plot([lb, lb], ylim, "b:")
ax.fill_between([lb, xlim[1]], ylim[1], color="b", alpha=0.1)
ax.legend(loc="center", bbox_to_anchor=(0.6, 0.9))
plt.tight_layout()
plt.savefig("rrlb_slides_1.png", dpi=500, bbox_inches="tight")

# plot 2 : log-barrier function for -1<=z and z<=2
print("Plot 2")
ax.fill_between([xlim[0], ub], ylim[1], color="r", alpha=0.1)
ax.plot(x, -np.log(ub - x), "r-", linewidth=2, label=rf"$B_2(z)$")
ax.plot([ub, ub], ylim, "r:")
ax.legend(loc="center", bbox_to_anchor=(0.6, 0.9))
plt.tight_layout()
plt.savefig("rrlb_slides_2.png", dpi=500, bbox_inches="tight")


# plot 3: recentered log-barrier function for -1<=z<=2
print("Plot 3")
fig, ax = create()
ax.plot(
    x,
    [B(a, 0.0) for a in x],
    color="tab:purple",
    linewidth=2,
    label=rf"$B(z)$",
)
ax.fill_between([lb, ub], ylim[1], color="r", alpha=0.1)
ax.fill_between([lb, ub], ylim[1], color="b", alpha=0.1)
ax.legend(loc="center", bbox_to_anchor=(0.6, 0.9))
plt.tight_layout()
plt.savefig("rrlb_slides_3.png", dpi=500, bbox_inches="tight")

# plot 4: relaxed log-barrier function for -1<=z
print("Plot 4")
fig, ax = create()
x = np.linspace(xlim[0], lb + delta, 100)
ax.plot(x, [beta(a - lb, delta) for a in x], "b-.", label=r"$\beta_\delta(1+z)$")
ax.plot([lb, lb], ylim, "b:")
ax.plot([lb + delta, lb + delta], ylim, "b:")
ax.fill_between([lb, xlim[1]], ylim[1], color="b", alpha=0.1)
x = np.linspace(lb + 1e-8, xlim[1], 900)
ax.plot(x, [Blb(a, 0.0) for a in x], "b-", linewidth=2, label=rf"$-\log(z+{-lb:.1f})$")
ax.add_patch(
    patches.FancyArrowPatch(
        (lb, 0.2),
        (lb + delta, 0.2),
        arrowstyle="<->",
        mutation_scale=10,
        color="b",
    )
)
ax.text(lb + delta / 2.0, 0.3, r"$\delta$", color="b")
ax.legend(loc="center", bbox_to_anchor=(0.6, 0.9))
plt.tight_layout()
plt.savefig("rrlb_slides_4.png", dpi=500, bbox_inches="tight")

# plot 5: relaxed recentered log-barrier function for -1<=z<=2
print("Plot 5")
fig, ax = create()
ax.fill_between([lb, ub], ylim[1], color="r", alpha=0.1)
ax.fill_between([lb, ub], ylim[1], color="b", alpha=0.1)
x = np.linspace(lb + 1e-8, ub - 1e-8, 900)
# x = np.linspace(lb + delta, ub - delta, 900)
ax.plot(x, [B(a, 0.0) for a in x], "tab:purple", linewidth=2, label=rf"original $B(z)$")
x = np.linspace(xlim[0], lb + delta, 100)
ax.plot(x, [B(a, delta) for a in x], "tab:purple", linestyle="-.", label=rf"relaxed $B(z)$")
x = np.linspace(ub - delta, xlim[1], 100)
ax.plot(x, [B(a, delta) for a in x], "tab:purple", linestyle="-.")
ax.plot([lb, lb], ylim, "b:")
ax.plot([lb + delta, lb + delta], ylim, "b:")
ax.plot([ub, ub], ylim, "r:")
ax.plot([ub - delta, ub - delta], ylim, "r:")
ax.add_patch(
    patches.FancyArrowPatch(
        (lb, 0.2),
        (lb + delta, 0.2),
        arrowstyle="<->",
        mutation_scale=10,
        color="b",
    )
)
ax.text(lb + delta / 2.0, 0.3, r"$\delta$", color="b")
ax.add_patch(
    patches.FancyArrowPatch(
        (ub, 0.2),
        (ub - delta, 0.2),
        arrowstyle="<->",
        mutation_scale=10,
        color="r",
    )
)
ax.legend(loc="center", bbox_to_anchor=(0.6, 0.9))
ax.text(ub - delta / 2.0, 0.3, r"$\delta$", color="r")
plt.tight_layout()
plt.savefig("rrlb_slides_5.png", dpi=500, bbox_inches="tight")

if sys.argv[-1] == "show":
    plt.show()
