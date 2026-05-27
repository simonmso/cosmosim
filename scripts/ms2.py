import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate

from RecombinationHistory import RecombinationHistory
from BackgroundCosmology import BackgroundCosmology
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
rec.X_e_saha(-9)

rec.solve()

print("Plotting")
npts = 3000
x = np.linspace(-12, 0, num=npts)

# X_e
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))
ax.plot(x, rec.Xe(x), label="Saha + Peebles", c="slategrey")
ax.plot(x, rec.X_e_saha(x), label="Saha", c="slategrey", ls="--")

ax.set_title(r"Fractional electron density $X_e(x)$")
ax.set_ylabel(r"$X_e$")
ax.set_xlabel(r"$x$")

ax.set_yscale("log")
ax.set_ylim(1e-5, 2)
ax.set_xlim(-12, 0)
ax.legend()
fig.savefig(path.join(dest, "X_e"))
plt.close(fig)

# X_e (z)
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))
z = 1 / np.exp(x) - 1
ax.plot(z, rec.Xe(x), label="Saha + Peebles", c="slategrey")
ax.plot(z, rec.X_e_saha(x), label="Saha", c="slategrey", ls="--")

ax.set_title(r"Fractional electron density $X_e(z)$")
ax.set_ylabel(r"$X_e$")
ax.set_xlabel(r"$z$")

ax.set_yscale("log")
ax.set_ylim(1e-5, 2)
ax.set_xlim(1800, z[-1])
ax.legend()
fig.savefig(path.join(dest, "X_e_z"))
plt.close(fig)

# tau
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))

ax.plot(x, rec.tau(x), label=r"$\tau(x)$", c="darkblue", lw=0.5)
ax.plot(x, -rec.dtau(x), label=r"$-\tau'(x)$", c="darkblue", ls="--", lw=0.5)
ax.plot(x, rec.d2tau(x), label=r"$\tau''(x)$", c="darkblue", ls=":", lw=0.5)

ax.set_yscale("log")
ax.set_ylim(1e-8)

ax.legend()

ax.set_title(r"Optical depth $\tau(x)$ and derivatives")
ax.set_ylabel(r"$\tau$")
ax.set_xlabel(r"$x$")
ax.set_xlim(-12, 0)
fig.savefig(path.join(dest, "tau"))
plt.close(fig)


# g
fig, axs = plt.subplots(
    nrows=3, height_ratios=(2, 1, 1), sharex=True, figsize=(apsw, 1.0 * apsw)
)

ax = axs[0]
ax.set_title(r"Visibility func. $\tilde g(x)$ and derivatives")
ax.plot(x, rec.g_tilde(x), label=r"$\tilde g(x)$", c="saddlebrown", lw=0.5)
ax.set_ylabel(r"$\tilde g$")
ax.legend()

ax = axs[1]
ax.plot(x, rec.dg_tilde(x), label=r"$\tilde g'(x)$", ls="--", c="saddlebrown", lw=0.5)
ax.set_ylabel(r"$\tilde g'$")
ax.legend()

ax = axs[2]
ax.plot(x, rec.d2g_tilde(x), label=r"$\tilde g''(x)$", ls=":", c="saddlebrown", lw=0.5)
ax.set_ylabel(r"$\tilde g''$")
ax.legend(loc="upper right")
# fig.legend(bbox_to_anchor=axs[0].bbox)
ax.set_xlim(-12, 0)
ax.set_xlabel(r"$x$")
fig.savefig(path.join(dest, "g"))
plt.close(fig)

print()
print("g test: integral of g = ", integrate.trapezoid(rec.g_tilde(x), x))


print()

# Report last scattering
print("Last scattering")
opt = optimize.minimize_scalar(lambda x: -1 * rec.g_tilde(x), (-8, -7, -6))
x_ls = opt.x
z_ls = 1 / np.exp(x_ls) - 1
yr = const.s * 60 * 60 * 24 * 365
t_ls = cosmo.t(x_ls) / yr
print(r"$x_{\rm{ls}}$ & " + f"{x_ls:.3f}")
print(r"$z_{\rm{ls}}$ & " + f"{z_ls:.1f}")
print(r"$t_{\rm{ls}}$ (yr) & " + f"{t_ls:.0f}")
print()


# Report recombination
# TODO: check this vs X_e = 0.1
X_e_rec = 0.1
print(f"Recombination (X_e = {X_e_rec:.1f}):")
res = optimize.root_scalar(lambda x: rec.Xe(x) - X_e_rec, bracket=(-8, -6))
x_rc = res.root
z_rc = 1 / np.exp(x_rc) - 1
t_rc = cosmo.t(x_rc) / yr
print(r"$x_{\rm{rc}}$ & " + f"{x_rc:.3f} \\\\")
print(r"$z_{\rm{rc}}$ & " + f"{z_rc:.1f} \\\\")
print(r"$t_{\rm{rc}}$ (yr) & " + f"{t_rc:.0f} \\\\")
print()

print(r"Sound horizon at recombination")
print(r"$s(x_{\rm{rc, full}})$ (Mpc) & " + f"{rec.s(x_rc)[0] / const.Mpc:.2f} \\\\")
print()

print(f"Recombination (Saha exp.) (X_e = {X_e_rec:.1f}):")
res = optimize.root_scalar(lambda x: rec.X_e_saha(x) - X_e_rec, bracket=(-8, -6))
x_rc = res.root
z_rc = 1 / np.exp(x_rc) - 1
t_rc = cosmo.t(x_rc) / yr
print(r"$x_{\rm{rc}}$ & " + f"{x_rc:.3f} \\\\")
print(r"$z_{\rm{rc}}$ & " + f"{z_rc:.1f} \\\\")
print(r"$t_{\rm{rc}}$ (yr) & " + f"{t_rc:.0f} \\\\")
print()

print(r"Freezeout abundance")
print("X_e(x=0) =", f"{rec.Xe(0.0):.3e}")
print()
