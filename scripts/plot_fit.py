import argparse
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MCMCResult import MCMCResult
from BackgroundCosmology import BackgroundCosmology
from Global import APS_COL_W as apsw
from Global import const


def main():
    # Set plot style
    # TODO: This should be defined dynamically
    plt.style.use("./style.mplstyle")

    # Load the results
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", required=True, help="Directory of the data files"
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    dest = args.output

    f = np.load(path.join(args.data, "supernova.npz"))
    chain = f["chain"]
    log_prob = f["log_prob"]

    # Use the fiducial for comparisons
    lcdm = BackgroundCosmology()
    lcdm.solve()

    h0 = chain[:, 0]
    omega_M = chain[:, 1]
    omega_K = chain[:, 2]
    omega_R = lcdm.OmegaR0tot * (lcdm.h0**2 / h0**2)

    omega_lam = 1 - omega_M - omega_K - omega_R

    # --------------- Combined --------------
    combined_res = MCMCResult(
        chain=[omega_M * h0**2, omega_lam * h0**2],
        log_prob=log_prob,
        labels=[r"$\Omega_{M,0}h_0^2$", r"$\Omega_{\Lambda,0}h_0^2$"],
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
        labels=[r"$\Omega_{M,0}$", r"$\Omega_{\Lambda,0}$"],
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
        labels=["$h_0$", r"$\Omega_{M,0}$", r"$\Omega_{K,0}$"],
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
        h0=indiv_res.best[0], OmegaM0=indiv_res.best[1], OmegaK0=indiv_res.best[2]
    )
    cos_best.solve(rtol=1e-9)

    z = np.linspace(supnva["z"].min(), supnva["z"].max(), 1000)

    fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

    ax.plot(z, cos_best.dL(x(z)) / (1e3 * const.Mpc), label="Best", c="red")
    ax.plot(z, lcdm.dL(x(z)) / (1e3 * const.Mpc), label="Fiducial", c="k")

    ax.errorbar(
        supnva["z"], supnva["d_L"], yerr=supnva["unc_d_L"], fmt=".", ms=3, c="k"
    )

    ax.set_xlabel("redshift (z)")
    ax.set_ylabel(r"$d_L$ (Gpc.)")
    ax.set_title(r"Supernova $d_L$ vs. redshift")

    ax.legend()

    fig.savefig(path.join(dest, "supernovea"))
    plt.close(fig)

    # ------------------ Hp(x) -----------------
    toy = BackgroundCosmology(
        OmegaM0=0.5,
        Neff=0,
    )
    toy.solve()

    fig, ax = plt.subplots(figsize=(apsw, 0.6 * apsw))
    x_arr = np.linspace(-12, 0, 1000)
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

    units = 1.0 / (const.s * 60 * 60 * 24 * 365 * 1e9)

    ax.plot(x_arr, cos_best.eta(x_arr) / const.c * units, label="Best", c="r")
    ax.plot(x_arr, lcdm.eta(x_arr) / const.c * units, label="Fiducial", c="k")
    ax.plot(x_arr, toy.eta(x_arr) / const.c * units, label="Toy")

    ax.set_title(r"$\frac{\eta(x)}{c}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\eta(x)/c$ (Gyr.)")
    ax.set_yscale("log")

    ax.legend()

    fig.savefig(path.join(dest, "eta"))
    plt.close(fig)


if __name__ == "__main__":
    main()
