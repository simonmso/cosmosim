import numpy as np
from os import path

# import RecombinationHistory
import Perturbations

# import PowerSpectrum
import argparse
import matplotlib.pyplot as plt

from scipy import integrate

from BackgroundCosmology import BackgroundCosmology
from RecombinationHistory import RecombinationHistory
from Global import APS_COL_W as apsw
from Global import const

"""

The whole project runs from this file. The task is split into four milestones
for which you have to implement four classes. Each of them has a solve, plot and info
method and methods to retrieve things you will compute. On top of that add what you want/need.
Do them in order and remove the exit after each milestone when you are ready to move on

Any physical constants and units you need can be found in [const]

"""

# At some point we could add some help text
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output")
args = parser.parse_args()

# Only make plots if an output was specified
make_plots = args.output is not None

# Set plot style
plt.style.use("./style.mplstyle")


# Milestone 1: Solve the background
# ============================================
cosmo = BackgroundCosmology(
    name="LCDM",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.05,  # Baryon density
    # OmegaB0=0.046,  # Baryon density
    OmegaCDM0=0.45,  # CDM density
    # OmegaCDM0=0.224,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)

# Solve and plot
# cosmo.info()
cosmo.solve()
if make_plots:
    cosmo.plot(args.output)

x_start = -18


# Milestone 2: Solve the recombination history
# ============================================
rec = RecombinationHistory(
    BackgroundCosmology=cosmo,
    Yp=0.24,  # Primordial helium fraction
    reionization=True,  # Include reionization
    z_reion=11.0,  # Reionization redshift
    delta_z_reion=0.5,  # Reionization width
    helium_reionization=True,  # Helium double reionization
    z_helium_reion=3.5,  # Helium double reionization redshift
    delta_z_helium_reion=0.5,
    x_start=x_start,
    x_end=0,
)  # Helium double reionization width


# # Solve and plot
# rec.info()
rec.solve()
if make_plots:
    rec.plot(args.output)

# pert_start = -20
pert_start = x_start
transition = -8.3
# Milestone 3: Integrate the perturbations
# ============================================
pert = Perturbations.Perturbations(
    BackgroundCosmology=cosmo,
    RecombinationHistory=rec,
    n_ell_theta=10,  # Number of ells (0,1,...,n-1) to include in the Boltzmann hierarchy
    keta_max=3000.0,  # Set kmax based on keta0. 3000 typically enough for Cell, lower for testing
    npts_k=200,  # 100-200 typically enough for Cell, lower for testing
    x_start=pert_start,
    x_end=0,
    transition=transition,
)


# Solve and plot
# pert.info()
pert.solve()

# --------- Perturbation debugging plots -------------
# kval = 0.0014
kval = 1000.0 / cosmo.eta(0)
print("kval", kval)
x_array = np.linspace(pert_start, 0, 5000)
kmpc = "{:.3g}".format(kval * const.Mpc)

k = np.ones_like(x_array) * kval

kx = np.array((k, x_array)).T

# Fetch data from splines
data_deltaCDM = pert.deltaCDM(kx)
data_deltaB = pert.deltaB(kx)
data_vCDM = pert.vCDM(kx)
data_vB = pert.vB(kx)
data_Phi = pert.Phi(kx)
data_Psi = pert.Psi(kx)
data_Theta0 = pert.Theta(kx, 0)
data_Theta1 = pert.Theta(kx, 1)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.set_yscale("log")
ax.set_title("Density perturbations k = " + str(kmpc) + "/ Mpc")
ax.plot(x_array, np.abs(data_deltaB), label="deltaB")
ax.plot(x_array, data_deltaCDM, label="deltaCDM")
ax.axvline(transition)
# ax.plot(x_array, np.abs(3.0 * data_Theta0), label="deltaR")
ax.legend()

fig.savefig(path.join(args.output, "density"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.set_yscale("symlog", linthresh=1e-5)
ax.set_title("Velocity perturbations k = " + str(kmpc) + "/ Mpc")
ax.plot(x_array, data_vB, label="vB")
ax.plot(x_array, data_vCDM, label="vCDM")
ax.axvline(transition)
# ax.scatter(pert.sol_tight.t, pert.sol_tight.y[1])
# ax.plot(x_array, -3.0 * data_Theta1, label="vR")
ax.legend()

fig.savefig(path.join(args.output, "velocity"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
# ax.set_yscale("log")
ax.set_title("Potentials k = " + str(kmpc) + "/ Mpc")
ax.plot(x_array, data_Phi, label=r"$\Phi$")
ax.plot(x_array, np.abs(data_Psi), label=r"$|\Psi|$")

ax.axvline(transition)
# print("data_Psi", data_Psi)
ax.legend()

fig.savefig(path.join(args.output, "potentials"))
plt.close(fig)

fig, axs = plt.subplots(nrows=4, figsize=(apsw, 2.3 * apsw), sharex=True)
# axs[0].plot(x_array, pert.Theta(kx, 1))
axs[0].plot(
    x_array,
    -(20 / 45)
    * (const.c * k / (cosmo.Hp(x_array) * rec.dtau(x_array)))
    * pert.Theta(kx, 1),
)
axs[0].scatter(pert.results_x, pert.results[7, 0, :], s=1, c="magenta")  # theta2 sol
axs[0].set_yscale("symlog", linthresh=1e-10)
# axs[0].set_yscale("log")
axs[1].plot(x_array, pert.Theta(kx, 2))
axs[1].scatter(pert.results_x, pert.results[7, 0, :], s=1, c="magenta")  # theta2 sol
axs[1].set_yscale("symlog", linthresh=1e-10)
axs[2].plot(x_array, cosmo.OmegaR0 * pert.Theta(kx, 2))
axs[2].set_yscale("symlog", linthresh=1e-10)
axs[3].plot(
    x_array,
    -12
    * (cosmo.H0 / (const.c * k * np.exp(x_array))) ** 2
    * (cosmo.OmegaR0 * pert.Theta(kx, 2)),
)
# axs[4].plot(
#     x_array,
#     -data_Phi
#     - 12
#     * (cosmo.H0 / (const.c * k * np.exp(x_array))) ** 2
#     * (cosmo.OmegaR0 * pert.Theta(kx, 2)),
# )
axs[0].axvline(transition)
axs[1].axvline(transition)
axs[2].axvline(transition)
axs[3].axvline(transition)

axs[3].set_xlim(-10, -7)
fig.savefig(path.join(args.output, "TEST"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
ax.set_title("Thetas k = " + str(kmpc) + "/ Mpc")
ax.plot(x_array, data_Theta0, label=r"$\Theta_0$")
ax.plot(x_array, data_Theta1, label=r"$\Theta_1$")
ax.plot(x_array, pert.Theta(kx, 2), label=r"$\Theta_2$")
ax.plot(x_array, pert.Theta(kx, 3), label=r"$\Theta_3$")
ax.axvline(transition)

# print("data_Psi", data_Psi)
ax.legend()

fig.savefig(path.join(args.output, "theta"))
plt.close(fig)
# --------- / Perturbation debugging plots -------------

# # Remove when done with milestone
# exit()


# # Milestone 4: PowerSpectrum evaluation
# # ============================================
# power = PowerSpectrum.PowerSpectrum(
#     BackgroundCosmology=cosmo,
#     RecombinationHistory=rec,
#     Perturbations=pert,
#     kpivot_mpc=0.05,  # Pivot scale in 1/Mpc
#     n_s=0.96,  # Spectral index
#     A_s=2e-9,  # Primordial amplitude
#     ell_max=1500,
# )  # Maximum ell to compute Cell up to

# # Solve and plot
# power.info()
# power.solve()
# if show_plots:
#     power.plot()
