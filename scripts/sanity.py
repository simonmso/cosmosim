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

cosmo = BackgroundCosmology(
    name="LCDM",  # Label
    h0=0.7,  # Hubble parameter
    OmegaB0=0.046,  # Baryon density
    OmegaCDM0=0.224,  # CDM density
    OmegaK0=0.0,  # Curvature density parameter
    TCMB_in_K=2.7255,  # Temperature of CMB today in Kelvin
    Neff=0.0,  # Effective number of neutrinos
)
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


npts = 2000
# -----------------------------------
# Testing omegas
x = np.linspace(-20, 5, num=npts)
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.plot(x, cosmo.OmegaRtot(x), label=r"$\Omega_R$")
ax.plot(x, cosmo.OmegaM(x), label=r"$\Omega_M$")
ax.plot(x, cosmo.OmegaLambda(x), label=r"$\Omega_\Lambda$")

ax.set_title(r"$\Omega_i(x)$, $\Lambda$CDM")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\Omega_i(x)$")
ax.legend()

fig.savefig(path.join(url, "Omega"))
plt.close(fig)

# Again for the Test
x = np.linspace(-20, 5, num=npts)
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.plot(x, test.OmegaRtot(x), label=r"$\Omega_R$")
ax.plot(x, test.OmegaM(x), label=r"$\Omega_M$")
ax.plot(x, test.OmegaLambda(x), label=r"$\Omega_\Lambda$")

ax.set_title(r"$\Omega_i(x)$, Test")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\Omega_i(x)$")
ax.legend()

fig.savefig(path.join(url, "Omega_test"))
plt.close(fig)


# -----------------------------------
# Testing H and H'
# H
fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))

for cosmology in [cosmo, matter_dom, rad_dom, lamb_dom]:
    H = cosmology.H(x) / (const.km / const.s / const.Mpc)

    ax.plot(x, H, label=cosmology.name)

fig.legend(loc="outside lower center", frameon=False)
ax.set_title(r"$H(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$H(x)$ (km/s/Mpc.)")
ax.set_yscale("log")

fig.savefig(path.join(url, "H"))
plt.close(fig)


# -----------------------------------
# Testing eta (LCDM)
fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))
x = np.linspace(cosmo.x_start, cosmo.x_end, npts)

eta = cosmo.eta(x) / const.Mpc

ax.plot(x, eta)
ax.set_title(r"$\eta(x)$, $\Lambda$CDM")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta(x)$ (Mpc.)")
ax.set_yscale("log")

fig.savefig(path.join(url, "eta"))
plt.close(fig)

# Testing eta (Test)
fig, ax = plt.subplots(figsize=(apsw, 1.0 * apsw))
x = np.linspace(-12, 0, npts)

eta = test.eta(x) / const.Mpc

ax.plot(x, eta)
ax.set_title(r"$\eta(x)$, Test")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta(x)$ (Mpc.)")
ax.set_yscale("log")

fig.savefig(path.join(url, "eta_test"))
plt.close(fig)


# ------------------------------------
# Testing Hp
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
x = np.linspace(-12, 0, npts)
Hp = cosmo.Hp(x) / (const.km / const.s / const.Mpc)
ax.plot(x, Hp)
ax.set_title(r"$\mathcal{H}(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\mathcal{H}(x)$ (km/s/Mpc.)")
ax.set_yscale("log")

fig.savefig(path.join(url, "Hp"))
plt.close(fig)

# Testing Hp, again for the test universe
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
x = np.linspace(-12, 0, npts)
Hp = test.Hp(x) / (const.km / const.s / const.Mpc)
ax.plot(x, Hp)
ax.set_title(r"$\mathcal{H}(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\mathcal{H}(x)$ (km/s/Mpc.)")
ax.set_yscale("log")

fig.savefig(path.join(url, "Hp_test"))
plt.close(fig)


# ------------------------------------
# eta H / c
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
x = np.linspace(cosmo.x_start, cosmo.x_end, npts)
Hp = cosmo.Hp(x)
eta = cosmo.eta(x)
ax.plot(x, (eta * Hp / const.c))
ax.set_title(r"$\frac{\eta\mathcal{H}}{c}$, $\Lambda$CDM")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta\mathcal{H}/c$")

fig.savefig(path.join(url, "etaHc"))
plt.close(fig)

# Again for the test universe
fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
x = np.linspace(test.x_start, test.x_end, npts)
Hp = test.Hp(x)
eta = test.eta(x)
ax.plot(x, (eta * Hp / const.c))
ax.set_title(r"$\frac{\eta\mathcal{H}}{c}$, Test")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta\mathcal{H}/c$")

fig.savefig(path.join(url, "etaHc_test"))
plt.close(fig)
