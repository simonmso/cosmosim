import argparse
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from BackgroundCosmology import BackgroundCosmology

from Global import APS_COL_W as apsw
from Global import const

plt.style.use("./style.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-d", "--data", required=True, help="Directory of the data files")
args = parser.parse_args()
dest = args.output
data = args.data


products = np.load(path.join(data, "m4_data_products.npz"))
lls = products["lls"]
ks = products["ks"]
ks_eta = products["ks_eta"]
ks_Mpc = products["ks_Mpc"]
transfers = products["transfers"]
integrands = products["integrands"]
full_ell = products["full_ell"]
Cl = products["Cl"]
Dl = products["Dl"]
matter = products["matter"]
Cl_planck = products["Cl_planck"]
Dl_planck = products["Dl_planck"]
Cl_sw = products["Cl_sw"]
Dl_sw = products["Dl_sw"]
Cl_isw = products["Cl_isw"]
Dl_isw = products["Dl_isw"]
Cl_doppler = products["Cl_doppler"]
Dl_doppler = products["Dl_doppler"]
Cl_thompson = products["Cl_thompson"]
Dl_thompson = products["Dl_thompson"]

cmap = mpl.colormaps["viridis"]

# Transfer function
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))

for i, l in enumerate(lls):
    c = cmap(i / len(lls))
    ax.plot(
        ks_eta,
        transfers[i],
        label=f"${l}$",
        c=c,
        lw=0.5,
    )

fig.legend(
    loc="outside right center",
    frameon=False,
    ncols=1,
    reverse=True,
    handlelength=1.2,
    title=r"$\ell$",
    borderpad=0,
)
ax.set_xlabel(r"$k\eta_0$")
ax.set_title(r"Transfer function $\sqrt{\ell (\ell + 1)} \Theta_\ell(k)$")

fig.savefig(path.join(args.output, "transfer"))
plt.close(fig)


# Integrand
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))

for i, l in enumerate(lls):
    c = cmap(i / len(lls))
    ax.plot(
        ks_eta,
        integrands[i],
        label=f"${l}$",
        c=c,
        lw=0.5,
    )


fig.legend(
    loc="outside right center",
    frameon=False,
    ncols=1,
    reverse=True,
    handlelength=1.2,
    title=r"$\ell$",
    borderpad=0,
)

ax.set_title(r"$C_\ell$ Integrand $\ell(\ell + 1)|\Theta_\ell(k)|^2 / k$")
ax.set_xlabel(r"$k\eta_0$")
ax.set_ylabel(r"(Mpc.)")

fig.savefig(path.join(args.output, "integrand"))
plt.close(fig)


# Power spectrum
low = pd.read_csv(
    path.join(args.data, "planck_cell_low.txt"),
    sep=r"\s+",
    skiprows=1,
    names=[
        "l",
        "Dl",
        "uncDl+",
        "uncDl-",
    ],  # Assuming these are Dl even though the header says Cl
)
high = pd.read_csv(
    path.join(args.data, "COM_PowerSpect_CMB-TT-binned_R3.01.txt"),
    sep=r"\s+",
    skiprows=1,
    names=["l", "Dl", "uncDl-", "uncDl+", "fit"],
)

# Low
rg = full_ell <= 50

fig, ax = plt.subplots(figsize=(apsw, 0.9 * apsw))

p = {"ms": 0.7, "c": "k"}

ax.errorbar(
    low["l"],
    low["Dl"],
    yerr=(low["uncDl-"], low["uncDl+"]),
    fmt=".",
    **p,
    label=r"Planck low-$\ell$",
)

ax.plot(full_ell[rg], Dl_planck[rg], label="Sim. (Planck 2018)", c="steelblue")
ax.plot(full_ell[rg], Dl[rg], label="Sim. (Toy)", c="tan")

ax.set_title(r"$C_\ell$, low-$\ell$ TT spectrum ")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K]$^2$")

fig.legend(loc="outside lower center", frameon=False)

fig.savefig(path.join(args.output, "planck_low"))
plt.close(fig)


# High
l_cheat = full_ell
# l_cheat = full_ell**1.018
Dl_cheat = Dl_planck * np.exp(-0.05 * (full_ell / 200) ** 1.5)
# Dl_cheat = Dl_planck * np.exp(-0.05 * (l_cheat / 200) ** 1.5)

# this is a bit of a hack
# TODO: when new results come in
# factor = Dl / (full_ell * (full_ell + 1) * Cl)
# Dl_cheat = (
#     l_cheat * (l_cheat + 1) * factor * Cl * np.exp(-0.05 * (l_cheat / 200) ** 1.5)
# )


fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))

ax.errorbar(
    high["l"],
    high["Dl"],
    yerr=(high["uncDl-"], high["uncDl+"]),
    fmt=".",
    **p,
    label="Planck",
)

ax.plot(full_ell, Dl_planck, label="Sim. (Planck 2018)", c="steelblue")
ax.plot(
    l_cheat, Dl_cheat, label="Sim. (Plank 2018 + He fudging)", c="steelblue", ls="--"
)
ax.plot(full_ell, Dl, label="Sim. (Toy)", c="tan")

ax.set_title(r"$C_\ell$, full TT spectrum")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K]$^2$")

fig.legend(loc="outside lower center", frameon=False)

ax.set_xlim(0, 2750)

fig.savefig(path.join(args.output, "planck_high"))
plt.close(fig)


# CMB breakdown
fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))

ax.plot(full_ell, Dl, label=r"Full $C_\ell$", c="k")
ax.plot(full_ell, Dl_sw, label="SW")
ax.plot(full_ell, Dl_isw, label="ISW")
ax.plot(full_ell, Dl_doppler, label="Doppler")
ax.plot(full_ell, Dl_thompson, label="Polarization")

fig.legend(loc="outside lower center", frameon=False, ncols=3)

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_title(r"$C_\ell$ Components")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K]$^2$")

ax.set_ylim(1e-2, 2e5)
ax.set_xlim(1, 2750)

fig.savefig(path.join(args.output, "components"))
plt.close(fig)


# Matter power spectrum
cosmo = BackgroundCosmology(
    name="LCDM",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.05,  # Baryon density
    OmegaCDM0=0.45,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
cosmo.solve()

a_rm = np.exp(cosmo.x_rm)
H_rm = cosmo.H(cosmo.x_rm)

k_eq = a_rm * H_rm / const.c / (cosmo.h0 / const.Mpc)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.plot(ks_Mpc, matter)
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$k$ (h/Mpc.)")
ax.set_ylabel(r"$P(k)$ (Mpc./h)$^3$")

ax.set_title("Matter power spectrum")

ax.axvline(k_eq, c="k", lw=0.5)
ax.annotate(r"$k_{\rm eq}$", (k_eq * 1.1, 9e2))
# print("k_eq", k_eq)

fig.savefig(path.join(args.output, "matter"))
plt.close(fig)
