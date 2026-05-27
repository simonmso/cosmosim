import argparse
from os import path

import numpy as np

from RecombinationHistory import RecombinationHistory
from BackgroundCosmology import BackgroundCosmology
from Perturbations import Perturbations
from PowerSpectrum import PowerSpectrum

from Global import const

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True, help="Directory of the data files")
args = parser.parse_args()
data = args.data

# variables that determine how long
# everything takes to run
keta_max = 3000
npts_k = 200
ell_max = 3500

# keta_max = 200.0
# npts_k = 20
# ell_max = 60

print("------------ Planck cosmology ------------")

h0 = 0.6737
cosmo_planck = BackgroundCosmology(
    name="LCDM",  # Label
    h0=h0,  # Hubble parameter
    OmegaB0=0.02233 / h0**2,  # Baryon density
    OmegaCDM0=0.1198 / h0**2,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
cosmo = cosmo_planck


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

pert_params = {
    "BackgroundCosmology": cosmo,
    "RecombinationHistory": rec,
    "n_ell_theta": 10,  # Number of ells (0,1,...,n-1) to include in the Boltzmann hierarchy
    "keta_max": keta_max,  # Set kmax based on keta0. 3000 typically enough for Cell, lower for testing
    "npts_k": npts_k,  # 100-200 typically enough for Cell, lower for testing
    "x_start": x_start,
    "x_end": 0,
    "transition": -8.3,
}
pert = Perturbations(**pert_params)

print("Solving Perturbations (base)")
pert.solve()

power_params = {
    "kpivot_mpc": 0.05,  # Pivot scale in 1/Mpc
    "n_s": 0.96,  # Spectral index
    "A_s": 2e-9,  # Primordial amplitude
    "ell_max": ell_max,
    "BackgroundCosmology": cosmo,
    "RecombinationHistory": rec,
}
power = PowerSpectrum(
    Perturbations=pert, **power_params
)  # Maximum ell to compute Cell up to

print("Solving Power Spectrum")
power.solve()

print(" ------- Solving sw only ------")
pert_sw = Perturbations(
    **pert_params,
    include_sw=True,
    include_isw=False,
    include_doppler=False,
    include_thompson=False,
)
pert_sw.solve()
power_sw = PowerSpectrum(Perturbations=pert_sw, **power_params)
power_sw.solve()

print(" ------- Solving isw only ------")
pert_isw = Perturbations(
    **pert_params,
    include_sw=False,
    include_isw=True,
    include_doppler=False,
    include_thompson=False,
)
pert_isw.solve()
power_isw = PowerSpectrum(Perturbations=pert_isw, **power_params)
power_isw.solve()

print(" ------- Solving doppler only ------")
pert_doppler = Perturbations(
    **pert_params,
    include_sw=False,
    include_isw=False,
    include_doppler=True,
    include_thompson=False,
)
pert_doppler.solve()
power_doppler = PowerSpectrum(Perturbations=pert_doppler, **power_params)
power_doppler.solve()

print(" ------- Solving thompson only ------")
pert_thompson = Perturbations(
    **pert_params,
    include_sw=False,
    include_isw=False,
    include_doppler=False,
    include_thompson=True,
)
pert_thompson.solve()
power_thompson = PowerSpectrum(Perturbations=pert_thompson, **power_params)
power_thompson.solve()


ells = power.ells
ks = power.ks
ks_eta = ks * cosmo.eta(0.0)

n_demo_ells = 6
lis = np.int32(np.round(np.linspace(0, len(ells) - 1, n_demo_ells)))
lls = [ells[li] for li in lis]

transfers = []
integrands = []

for li in lis:
    l = ells[li]

    # Transfer
    trans = np.sqrt(l * (l + 1)) * power.theta_splines[li](ks)
    transfers.append(trans)

    # Integrand
    intgd = trans**2 / ks / const.Mpc
    integrands.append(intgd)

transfers = np.array(transfers)
integrands = np.array(integrands)


# Power spectra
muKsq = (1e-6 * const.K / cosmo.TCMB0) ** 2
full_ell = np.arange(1, ells[-1])
Cl = power.cell_TT(full_ell)
matter = power.matter_power_spectrum(ks, 0.0)

# Components
Cl_sw = power_sw.cell_TT(full_ell)
Cl_isw = power_isw.cell_TT(full_ell)
Cl_doppler = power_doppler.cell_TT(full_ell)
Cl_thompson = power_thompson.cell_TT(full_ell)


def Dl(cl):
    return full_ell * (full_ell + 1) * cl / (2 * np.pi) / muKsq


print("------------------- Saving -------------------")
np.savez(
    file=path.join(data, "m4_data_products.npz"),
    lls=lls,
    ks=ks,
    ks_eta=ks_eta,
    ks_Mpc=ks / (cosmo.h0 / const.Mpc),
    matter_raw=matter,
    matter=matter * (cosmo.h0 / const.Mpc) ** 3,
    transfers=transfers,
    integrands=integrands,
    full_ell=full_ell,
    Cl=Cl,
    Dl=Dl(Cl),
    Cl_sw=Cl_sw,
    Dl_sw=Dl(Cl_sw),
    Cl_isw=Cl_isw,
    Dl_isw=Dl(Cl_isw),
    Cl_doppler=Cl_doppler,
    Dl_doppler=Dl(Cl_doppler),
    Cl_thompson=Cl_thompson,
    Dl_thompson=Dl(Cl_thompson),
)
