# import RecombinationHistory
# import Perturbations
# import PowerSpectrum
import argparse
import matplotlib.pyplot as plt

from BackgroundCosmology import BackgroundCosmology
from RecombinationHistory import RecombinationHistory

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
    OmegaB0=0.046,  # Baryon density
    OmegaCDM0=0.224,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)

# Solve and plot
cosmo.info()
cosmo.solve()
# if make_plots:
#     cosmo.plot(args.output)

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
)  # Helium double reionization width

# # Solve and plot
rec.info()
rec.solve()
if make_plots:
    rec.plot(args.output)

# # Remove when done with milestone
exit()


# # Milestone 3: Integrate the perturbations
# # ============================================
# pert = Perturbations.Perturbations(
#     BackgroundCosmology=cosmo,
#     RecombinationHistory=rec,
#     n_ell_theta=10,  # Number of ells (0,1,...,n-1) to include in the Boltzmann hierarchy
#     keta_max=1500.0,  # Set kmax based on keta0. 3000 typically enough for Cell, lower for testing
#     npts_k=100,
# )  # 100-200 typically enough for Cell, lower for testing

# # Solve and plot
# pert.info()
# pert.solve()
# if show_plots:
#     pert.plot(pert.k_min)
# if show_plots:
#     pert.plot(pert.k_max)

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
