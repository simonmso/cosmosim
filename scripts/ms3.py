import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy import optimize
from scipy import integrate

from RecombinationHistory import RecombinationHistory
from BackgroundCosmology import BackgroundCosmology
from Perturbations import Perturbations
from Global import APS_COL_W as apsw
from Global import const

plt.style.use("./style.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()
dest = args.output


h0 = 0.6737
cosmo_planck = BackgroundCosmology(
    name="LCDM",  # Label
    h0=0.6766,  # Hubble parameter
    OmegaB0=0.02233 / h0**2,  # Baryon density
    # OmegaB0=0.046,  # Baryon density
    OmegaCDM0=0.1198 / h0**2,  # CDM density
    # OmegaCDM0=0.224,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
cosmo = cosmo_planck
# cosmo = BackgroundCosmology(
#     name="LCDM",  # Label
#     h0=0.7,  # Hubble parameter
#     OmegaB0=0.05,  # Baryon density
#     # OmegaB0=0.046,  # Baryon density
#     OmegaCDM0=0.45,  # CDM density
#     # OmegaCDM0=0.224,  # CDM density
#     OmegaK0=0.0,  # Curvature density parameter
#     TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
#     Neff=0.0,  # Effective number of neutrinos
# )

print("Solving Background")
cosmo.solve()

x_start = -18

rec = RecombinationHistory(
    BackgroundCosmology=cosmo,
    Yp=0.24,  # Primordial helium fraction
    reionization=False,  # Include reionization
    z_reion=11.0,  # Reionization redshift
    delta_z_reion=0.5,  # Reionization width
    helium_reionization=False,  # Helium double reionization
    z_helium_reion=3.5,  # Helium double reionization redshift
    delta_z_helium_reion=0.5,
    x_start=x_start,
    x_end=0,
)  # Helium double reionization width

print("Solving Recombination")
rec.solve()

transition = -8.3
pert = Perturbations(
    BackgroundCosmology=cosmo,
    RecombinationHistory=rec,
    n_ell_theta=10,  # Number of ells (0,1,...,n-1) to include in the Boltzmann hierarchy
    # keta_max=300.0,  # Set kmax based on keta0. 3000 typically enough for Cell, lower for testing
    keta_max=3000.0,  # Set kmax based on keta0. 3000 typically enough for Cell, lower for testing
    # npts_k=20,  # 100-200 typically enough for Cell, lower for testing
    npts_k=300,  # 100-200 typically enough for Cell, lower for testing
    x_start=x_start,
    x_end=0,
    transition=transition,
)

print("Solving Perturbations")
pert.solve()


ks_bare = [0.002, 0.02, 0.2]
# ks_bare = [0.001, 0.002, 0.003]
ks = np.array(ks_bare) / const.Mpc

npts = 4000
x = np.linspace(-17, 0, num=npts)

k = ks[0]

c = {
    "gamma": "orange",
    "cdm": "forestgreen",
    "b": "navy",
}

ls = ["-", "--", ":"]

# Density
fig, ax = plt.subplots(figsize=(apsw, 0.9 * apsw))

for ki, k in enumerate(ks):
    ax.plot(x, 4 * pert.Theta((k, x), 0), c=c["gamma"], ls=ls[ki])
    ax.plot(x, np.abs(pert.deltaB((k, x))), c=c["b"], ls=ls[ki])
    ax.plot(x, pert.deltaCDM((k, x)), c=c["cdm"], ls=ls[ki])

ax.set_title("Density Perturbations")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\delta_i$")

ax.set_yscale("log")
ax.set_xlim(-17, 0)
ax.set_ylim(1e-2)

# legend fiddliness
style_b = lines.Line2D([], [], c=c["b"])
style_gamma = lines.Line2D([], [], c=c["gamma"])
style_cdm = lines.Line2D([], [], c=c["cdm"])

k_lines = [lines.Line2D([], [], c="k", ls=ls[i]) for i in range(len(ks))]
k_labels = [f"$k$ = {k} / Mpc" for k in ks_bare]

fig.legend(
    [style_gamma, style_cdm, style_b, *k_lines],
    [r"$\delta_\gamma$", r"$\delta_{\rm CDM}$", r"$|\delta_{\rm b}|$", *k_labels],
    loc="outside lower center",
    ncols=2,
    frameon=False,
)

fig.savefig(path.join(dest, "density"))
plt.close(fig)


# velocity
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(apsw, 1.2 * apsw))

ax = axs[0]
ax.set_title("Velocity Perturbations")
for ki, k in enumerate(ks):
    ax.plot(x, np.abs(pert.vB((k, x))), c=c["b"], ls=ls[ki])
    ax.plot(x, pert.vCDM((k, x)), c=c["cdm"], ls=ls[ki])
# ax.set_yscale("log")
ax.set_ylabel(r"$v_i$")

ax = axs[1]
fade_points = [0, -3.7, -7.3]
for ki, k in enumerate(ks):
    v = np.abs(-3 * pert.Theta((k, x), 1))

    rg_full = x <= fade_points[ki]
    rg_faded = x > fade_points[ki]

    plt_args = {"c": c["gamma"], "ls": ls[ki]}
    ax.plot(x[rg_full], v[rg_full], **plt_args, alpha=1.0)
    ax.plot(x[rg_faded], v[rg_faded], **plt_args, alpha=0.1)

fig.legend(
    [style_gamma, style_cdm, style_b, *k_lines],
    [r"$|v_\gamma|$", r"$v_{\rm CDM}$", r"$|v_{\rm b}|$", *k_labels],
    loc="outside lower center",
    ncols=2,
    frameon=False,
)

ax.set_yscale("log")
ax.set_xlim(-17, 0)
ax.set_ylabel(r"$v_i$")
ax.set_xlabel(r"$x$")

fig.savefig(path.join(dest, "velocity"))
plt.close(fig)


# Theta 2
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

fade_points = [0, -2, -5.5]
for ki, k in enumerate(ks):
    plt_args = {"c": "k", "ls": ls[ki]}
    theta2 = pert.Theta((k, x), 2)

    rg_full = x <= fade_points[ki]
    rg_faded = x > fade_points[ki]
    ax.plot(x[rg_full], theta2[rg_full], **plt_args, alpha=1.0, label=k_labels[ki])
    ax.plot(x[rg_faded], theta2[rg_faded], **plt_args, alpha=0.1)


ax.set_ylabel(r"$\Theta_2$")
ax.set_xlabel(r"$x$")
ax.set_xlim(-17, 0)
ax.legend()
ax.set_title(r"Photon Quadrupole $\Theta_2$")

fig.savefig(path.join(dest, "theta2"))
plt.close(fig)


# Phi
fig, ax = plt.subplots(figsize=(apsw, 0.9 * apsw))

ax.set_title("Potentials")

c_phi = "slategrey"
c_sum = "firebrick"

for ki, k in enumerate(ks):
    Phi = pert.Phi((k, x))
    Psi = pert.Psi((k, x))
    ax.plot(x, Phi, ls=ls[ki], c=c_phi)
    ax.plot(x, np.abs(Phi + Psi), ls=ls[ki], c=c_sum)


style_phi = lines.Line2D([], [], c=c_phi)
style_sum = lines.Line2D([], [], c=c_sum)
style_blank = lines.Line2D([], [], c="k", alpha=0.0)

fig.legend(
    [style_phi, style_sum, style_blank, *k_lines],
    [r"$\Phi$", r"$|\Phi + \Psi|$", "", *k_labels],
    loc="outside lower center",
    ncols=2,
    frameon=False,
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"potential")
ax.set_xlim(-17, 0)

fig.savefig(path.join(dest, "phi"))
plt.close(fig)
