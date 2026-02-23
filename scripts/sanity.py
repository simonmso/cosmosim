# import RecombinationHistory
# import Perturbations
# import PowerSpectrum
import argparse
from os import path
import matplotlib.pyplot as plt
import numpy as np

from BackgroundCosmology import BackgroundCosmology
from Global import const
from Global import APS_COL_W as apsw

# At some point we could add some help text
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()
url = args.output

# Set plot style
plt.style.use("./style.mplstyle")

npts = 2000
x = np.linspace(-20, 5, num=npts)

cosmo = BackgroundCosmology(
    name="LCDM",
    x_pts=x,
)  # Label
cosmo.info()
cosmo.solve()

# -----------------------------------
# Testing omegas
fig, axs = plt.subplots(figsize=(apsw, 2.3 * apsw), nrows=4, sharex=True)
ax = axs[0]

ax.plot(x, cosmo.OmegaRtot(x), label=r"$\Omega_R$")
ax.plot(x, cosmo.OmegaM(x), label=r"$\Omega_M$")
ax.plot(x, cosmo.OmegaLambda(x), label=r"$\Omega_\Lambda$")

ax.set_title(r"$\Omega_i(x)$, $\Lambda$CDM")
# ax.set_xlabel("$x$")
ax.set_ylabel(r"$\Omega_i(x)$")
ax.legend(loc="center left")


def mark_equalities(ax):
    ax.axvline(cosmo.x_rm, c="k", alpha=0.2, ls="--")
    ax.axvline(cosmo.x_mlam, c="k", alpha=0.2, ls="--")
    ax.annotate(
        r"RM", (cosmo.x_rm + 0.2, 0.94), size=7, xycoords=("data", "axes fraction")
    )
    ax.annotate(
        r"M$\Lambda$",
        (cosmo.x_mlam + 0.2, 0.94),
        size=7,
        xycoords=("data", "axes fraction"),
    )


mark_equalities(ax)

# -----------------------------------
# (1 / Hp) (dHp / dx)
ax = axs[1]

# expectations
for y, lab in zip([-1.0, -0.5, 1.0], ["R", "M", r"$\Lambda$"]):
    ax.axhline(y, c="k", ls="-.", lw=0.3)
    ax.annotate(lab, (0.95, y + 0.05), size=7, xycoords=("axes fraction", "data"))

ax.plot(
    x,
    (1 / cosmo.Hp(x)) * cosmo.dHpdx(x),
    label=r"$\frac{1}{\mathcal{H}}\frac{d\mathcal{H}}{dx}$",
)

ax.set_title(r"(A)")
ax.set_ylabel(r"$(1/\mathcal{H})(d\mathcal{H}/dx)$")
ax.set_ylim(-1.2, 1.2)

mark_equalities(ax)

ax.legend()

# -----------------------------------
# (1 / Hp) (d2Hp / dx2)
ax = axs[2]

# expectations
for y, lab in zip([1.0, 0.25], [r"R,$\Lambda$", "M"]):
    ax.axhline(y, c="k", ls="-.", lw=0.3)
    ax.annotate(lab, (0.93, y + 0.03), size=7, xycoords=("axes fraction", "data"))


ax.plot(
    x,
    (1 / cosmo.Hp(x)) * cosmo.d2Hpdx2(x),
    label=r"$\frac{1}{\mathcal{H}}\frac{d^2\mathcal{H}}{dx^2}$",
)

ax.set_title(r"(B)")
ax.set_ylabel(r"$(1/\mathcal{H})(d^2\mathcal{H}/dx^2)$")
ax.set_ylim(0, 1.4)

mark_equalities(ax)

ax.legend()

# ------------------------------------
# eta H / c
ax = axs[3]
Hp = cosmo.Hp(x)
eta = cosmo.eta(x)
ax.plot(x, (eta * Hp / const.c), label=r"$\frac{\eta\mathcal{H}}{c}$")
ax.set_title("(C)")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta\mathcal{H}/c$")
ax.set_xlim(-21, 6)

ax.set_yscale("log")

mark_equalities(ax)

ax.axhline(1, c="k", ls="-.", lw=0.3)
ax.annotate("R", (0.93, 1 + 0.1), size=7, xycoords=("axes fraction", "data"))

ax.legend()


fig.savefig(path.join(url, "domination"))
plt.close(fig)
