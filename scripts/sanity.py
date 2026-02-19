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

rad_dom = BackgroundCosmology(
    name="Radiation Dominated",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.0,  # Baryon density
    OmegaCDM0=0.0,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=32.3,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)

rad_dom.info()

matter_dom = BackgroundCosmology(
    name="Matter Dominated",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.3,  # Baryon density
    OmegaCDM0=0.69,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
matter_dom.info()

lamb_dom = BackgroundCosmology(
    name="Dark Energy Dominated",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.0,  # Baryon density
    OmegaCDM0=0.0,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
lamb_dom.info()

test = BackgroundCosmology(
    name="Test",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.5,  # Baryon density
    OmegaCDM0=0.0,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    Neff=0.0,  # Effective number of neutrinos
)
test.info()


# Solve them all
rad_dom.solve()
matter_dom.solve()
lamb_dom.solve()
cosmo.solve()
test.solve()

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

# Find equalities
rg = (x > -10.0) & (x < 0.0)
rm_eq_idx = np.argmin(np.abs(cosmo.OmegaRtot(x) - cosmo.OmegaM(x))[rg])
mlam_eq_idx = np.argmin(np.abs(cosmo.OmegaLambda(x) - cosmo.OmegaM(x))[rg])
rm_eq = x[rg][rm_eq_idx]
mlam_eq = x[rg][mlam_eq_idx]


def mark_equalities(ax):
    ax.axvline(rm_eq, c="k", alpha=0.2, ls="--")
    ax.axvline(mlam_eq, c="k", alpha=0.2, ls="--")
    ax.annotate(r"RM", (rm_eq + 0.2, 0.94), size=7, xycoords=("data", "axes fraction"))
    ax.annotate(
        r"M$\Lambda$", (mlam_eq + 0.2, 0.94), size=7, xycoords=("data", "axes fraction")
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


# -----------------------------------
# # Testing H and H'
# # H
# x = np.linspace(-20, 5, num=npts)
# fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))

# for cosmology in [cosmo, matter_dom, rad_dom, lamb_dom]:
#     H = cosmology.H(x) / (const.km / const.s / const.Mpc)

#     ax.plot(x, H, label=cosmology.name)

# fig.legend(loc="outside lower center", frameon=False)
# ax.set_title(r"$H(x)$")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$H(x)$ (km/s/Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "H"))
# plt.close(fig)

# # H'
# fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))

# for cosmology in [cosmo, matter_dom, rad_dom, lamb_dom]:
#     dH = cosmology.dHdx(x) / (const.km / const.s / const.Mpc)

#     ax.plot(x, -dH, label=cosmology.name)

# fig.legend(loc="outside lower center", frameon=False)
# ax.set_title(r"$dH(x)/dx$")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$-dH(x)/dx$ (km/s/Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "dHdx"))
# plt.close(fig)


# # -----------------------------------
# # Testing eta (LCDM)
# fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))
# x = np.linspace(cosmo.x_start, cosmo.x_end, npts)

# eta = cosmo.eta(x) / const.Mpc

# ax.plot(x, eta)
# ax.set_title(r"$\eta(x)$, $\Lambda$CDM")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\eta(x)$ (Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "eta"))
# plt.close(fig)

# # Testing eta (Test)
# fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))
# x = np.linspace(-12, 0, npts)

# eta = test.eta(x) / const.Mpc

# ax.plot(x, eta)
# ax.set_title(r"$\eta(x)$, Test")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\eta(x)$ (Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "eta_test"))
# plt.close(fig)


# # ------------------------------------
# # Testing Hp
# fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
# x = np.linspace(-12, 0, npts)
# Hp = cosmo.Hp(x) / (const.km / const.s / const.Mpc)
# ax.plot(x, Hp)
# ax.set_title(r"$\mathcal{H}(x)$")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\mathcal{H}(x)$ (km/s/Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "Hp"))
# plt.close(fig)

# # Testing Hp, again for the test universe
# fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
# x = np.linspace(-12, 0, npts)
# Hp = test.Hp(x) / (const.km / const.s / const.Mpc)
# ax.plot(x, Hp)
# ax.set_title(r"$\mathcal{H}(x)$")
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\mathcal{H}(x)$ (km/s/Mpc.)")
# ax.set_yscale("log")

# fig.savefig(path.join(url, "Hp_test"))
# plt.close(fig)
