import argparse
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MCMCResult import MCMCResult
from BackgroundCosmology import BackgroundCosmology
from Global import APS_COL_W as apsw
from Global import const

# Set plot style
plt.style.use("./style.mplstyle")

# Load the results
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True, help="Directory of the data files")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

dest = args.output


# -------------------------------------------------------
def fig_1():
    """Figure 1"""
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

    fig.savefig(path.join(dest, "domination"))
    plt.close(fig)


fig_1()
# -------------------------------------------------------


f = np.load(path.join(args.data, "supernova.npz"))
chain = f["chain"]
log_prob = f["log_prob"]

# Use the fiducial for comparisons
lcdm = BackgroundCosmology()
lcdm.solve(rtol=1e-13)
lcdm.info()

h0 = chain[:, 0]
omega_M = chain[:, 1]
omega_K = chain[:, 2]
omega_R = lcdm.OmegaR0tot * (lcdm.h0**2 / h0**2)

omega_lam = 1 - omega_M - omega_K - omega_R

# --------------- Combined --------------
combined_res = MCMCResult(
    chain=[omega_M * h0**2, omega_lam * h0**2],
    log_prob=log_prob,
    labels=[r"$\Omega_{m,0}h_0^2$", r"$\Omega_{\Lambda,0}h_0^2$"],
    fiducial=[lcdm.OmegaM0 * lcdm.h0**2, lcdm.OmegaLambda0 * lcdm.h0**2],
)

# hist
fig, ax = combined_res.hist2d()
ax.set_title("MCMC Params. for Supernova $d_L$")
fig.savefig(path.join(dest, "omega_hist"))
plt.close(fig)

# report
combined_res.report()

# --------------- M, Lambda ----------------
m_lam_res = MCMCResult(
    chain=[omega_M, omega_lam],
    log_prob=log_prob,
    labels=[r"$\Omega_{m,0}$", r"$\Omega_{\Lambda,0}$"],
    fiducial=[lcdm.OmegaM0, lcdm.OmegaLambda0],
)

fig, ax = m_lam_res.hist2d()
ax.set_title("MCMC Params. for Supernova $d_L$")
ax.set_xlim(0)
ax.set_ylim(0)
# ax.legend(scatterpoints=1, loc="center right")
fig.savefig(path.join(dest, "omega_hist_MLam"))
plt.close(fig)

m_lam_res.report()

# ------------------ h0 only ----------------
indiv_res = MCMCResult([chain.T[0]], log_prob, labels=["$h_0$"], fiducial=[lcdm.h0])

# hists
fig, axs = indiv_res.hist()
axs.set_title("$h_0$ sample distribution")
fig.savefig(path.join(dest, "h0_hist"))
plt.close()

# report
indiv_res.report()

# -------------- Individual ----------------
indiv_res = MCMCResult(
    chain.T,
    log_prob,
    labels=["$h_0$", r"$\Omega_{m,0}$", r"$\Omega_{k,0}$"],
    fiducial=[lcdm.h0, lcdm.OmegaM0, lcdm.OmegaK0],
)

# hists
fig, axs = indiv_res.hist()
fig.savefig(path.join(dest, "omega_hists_individual"))
plt.close()

# report
indiv_res.report()

# ------------ As compared to supernovea ----------
supernova_path = path.join(args.data, "supernovadata.txt")
supnva = pd.read_csv(
    supernova_path,
    sep=r"\s+",
    skiprows=1,
    names=["z", "d_L", "unc_d_L"],
)


def x(z):
    a = 1 / (1 + z)
    return np.log(a)


cos_best = BackgroundCosmology(
    h0=indiv_res.best[0],
    OmegaM0=indiv_res.best[1],
    OmegaK0=indiv_res.best[2],
    name="Best",
)
cos_best.solve(rtol=1e-13)

z = np.linspace(supnva["z"].min(), supnva["z"].max(), 1000)

fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

ax.plot(z, cos_best.dL(x(z)) / (1e3 * const.Mpc), label="Best", c="red")
ax.plot(z, lcdm.dL(x(z)) / (1e3 * const.Mpc), label="Fiducial", c="k")

ax.errorbar(supnva["z"], supnva["d_L"], yerr=supnva["unc_d_L"], fmt=".", ms=3, c="k")

ax.set_xlabel("redshift (z)")
ax.set_ylabel(r"$d_L$ (Gpc.)")
ax.set_title(r"Supernova $d_L$ vs. redshift")

ax.legend()

fig.savefig(path.join(dest, "supernovea"))
plt.close(fig)


# ----------------- Report results ---------
def z_of_x(x):
    a = np.exp(x)
    return (1 / a) - 1


gyr = const.s * 60 * 60 * 24 * 365 * 1e9

for cos in [lcdm, cos_best]:
    print()
    print(f"---------- Reporting: {cos.name} --------------")
    print("Radiation-Matter eq.:")
    print(
        f"x: {cos.x_rm:.5f}  z: {z_of_x(cos.x_rm):.5f}  t: {cos.t(cos.x_rm) / gyr:.5e} Gyr"
    )
    print(r"$x_{\rm rm}$ & " + f"{cos.x_rm:.3f} \\\\")
    print(r"$z_{\rm rm}$ & " + f"{z_of_x(cos.x_rm):.0f} \\\\")
    print(r"$t_{\rm rm}$ & " + f"{cos.t(cos.x_rm) / gyr:.3e} (Gyr.) \\\\")
    print()
    print("Matter-Lambda eq.:")
    print(
        f"x: {cos.x_mlam:.5f}  z: {z_of_x(cos.x_mlam):.5f}  t: {cos.t(cos.x_mlam) / gyr:.5e} Gyr"
    )
    print(r"$x_{\rm m\Lambda}$ & " + f"{cos.x_mlam:.4f} \\\\")
    print(r"$z_{\rm m\Lambda}$ & " + f"{z_of_x(cos.x_mlam):.4f} \\\\")
    print(r"$t_{\rm m\Lambda}$ & " + f"{cos.t(cos.x_mlam) / gyr:.3f} (Gyr.) \\\\")
    print()
    print("Acceleration:")
    print(r"$x_{\rm accel.}$ & " + f"{cos.x_accel:.4f} \\\\")
    print(r"$z_{\rm accel.}$ & " + f"{z_of_x(cos.x_accel):.4f} \\\\")
    print(r"$t_{\rm accel.}$ & " + f"{cos.t(cos.x_accel) / gyr:.3f} (Gyr.) \\\\")
    print()
    print(f"Age of current universe t(0) (Gyr): {cos.t(0) / gyr}")
    print(f"Age $t(x=0)$  & {cos.t(0) / gyr:.2f} (Gyr.)\\\\")
    print(
        f"Conformal time  of current universe eta(0)/c (Gyr): {cos.eta(0) / const.c / gyr}"
    )
    print(
        r"Conformal time $\eta(x=0)/c$"
        + f" & {cos.eta(0) / const.c / gyr:.2f} (Gyr.)\\\\"
    )
    print()


# ------------------ Hp(x) -----------------
toy = BackgroundCosmology(
    OmegaM0=0.5,
    Neff=0,
)
toy.solve()

x_arr = np.linspace(-12, 0, 1000)

fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))
units = 1 / (const.km / const.s / const.Mpc)

ax.plot(x_arr, cos_best.Hp(x_arr) * units, label="Best", c="r")
ax.plot(x_arr, lcdm.Hp(x_arr) * units, label="Fiducial", c="k")
ax.plot(x_arr, toy.Hp(x_arr) * units, label="Toy")

ax.set_title(r"$\mathcal{H}(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\mathcal{H}(x)$ (km/s/Mpc.)")
ax.set_yscale("log")
ax.set_xlim(x_arr[0], x_arr[-1])

ax.legend()

fig.savefig(path.join(dest, "Hp"))
plt.close(fig)

# ------------------ eta / c ----------------
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))

ax.plot(x_arr, cos_best.eta(x_arr) / const.c / gyr, label="Best", c="r")
ax.plot(x_arr, lcdm.eta(x_arr) / const.c / gyr, label="Fiducial", c="k")
ax.plot(x_arr, toy.eta(x_arr) / const.c / gyr, label="Toy")

ax.set_title(r"$\frac{\eta(x)}{c}$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\eta(x)/c$ (Gyr.)")
ax.set_yscale("log")
ax.set_xlim(x_arr[0], x_arr[-1])

ax.legend()

fig.savefig(path.join(dest, "eta"))
plt.close(fig)

# ------------------ t(x) ----------------
fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))

ax.plot(x_arr, cos_best.t(x_arr) / gyr, label="Best", c="r")
ax.plot(x_arr, lcdm.t(x_arr) / gyr, label="Fiducial", c="k")
ax.plot(x_arr, toy.t(x_arr) / gyr, label="Toy")

ax.set_xlabel("$x$")
ax.set_ylabel("$t$ (Gyr.)")
ax.set_title("$t(x)$")
ax.set_yscale("log")
ax.set_xlim(x_arr[0], x_arr[-1])

ax.legend()

fig.savefig(path.join(dest, "t"))
plt.close(fig)
