import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt

from MCMCResult import MCMCResult
from BackgroundCosmology import BackgroundCosmology


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

    mcres = MCMCResult(
        chain=[omega_M * h0**2, omega_lam * h0**2],
        log_prob=log_prob,
        labels=[r"$\Omega_{M,0}h_0^2$", r"$\Omega_{\Lambda,0}h_0^2$"],
    )

    # --------------- Plot Chain --------------
    # fig, ax = mcres.hist2d()
    # ax.scatter(  # Include the fiducial values
    #     lcdm.OmegaM0 * lcdm.h0**2,
    #     lcdm.OmegaLambda0 * lcdm.h0**2,
    #     s=10,
    #     marker="x",
    #     c="red",
    #     label="LCDM",
    # )
    # fig.savefig(path.join(dest, "omega_hist"))
    # plt.close(fig)

    # ----------- Plot Contours --------------
    # fig, ax = mcres.contour()
    # fig.savefig(path.join(dest, "omega_contours"))
    # plt.close()

    # fig, axs = mcres.hist()
    # fig.savefig(path.join(dest, "omega_hists"))
    # plt.close()

    # ---------- Plot other chains ----------
    mcres = MCMCResult(
        chain.T,
        log_prob,
        labels=["$h_0$", r"$\Omega_{M,0}$", r"$\Omega_{K,0}$"],
    )
    fig, axs = mcres.hist()
    fig.savefig(path.join(dest, "omega_hists_individual"))
    plt.close()

    # mcres = MCMCResult(
    #     chain=[omega_M, omega_lam],
    #     log_prob=log_prob,
    #     labels=[r"$\Omega_{M,0}$", r"$\Omega_{\Lambda,0}$"],
    # )
    # fig, ax = mcres.hist2d()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1.4)
    # fig.savefig(path.join(dest, "omega_hist_e"))
    # plt.close(fig)

    # fig, ax = mcres.contour()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1.4)
    # fig.savefig(path.join(dest, "omega_contour_e"))
    # plt.close(fig)

    # mcres = MCMCResult(
    #     chain=[omega_M, omega_K],
    #     log_prob=log_prob,
    #     labels=[r"$\Omega_{M,0}$", r"$\Omega_{K,0}$"],
    # )
    # fig, ax = mcres.hist2d()
    # fig.savefig(path.join(dest, "omega_hist_b"))
    # plt.close(fig)

    # mcres = MCMCResult(
    #     chain=[omega_M, h0],
    #     log_prob=log_prob,
    #     labels=[r"$\Omega_{M,0}$", r"$h0$"],
    # )
    # fig, ax = mcres.hist2d()
    # fig.savefig(path.join(dest, "omega_hist_c"))
    # plt.close(fig)

    # mcres = MCMCResult(
    #     chain=[omega_K, h0],
    #     log_prob=log_prob,
    #     labels=[r"$\Omega_{K,0}$", r"$h0$"],
    # )
    # fig, ax = mcres.hist2d()
    # fig.savefig(path.join(dest, "omega_hist_d"))
    # plt.close(fig)

    # ------------ Test optimal values ----------
    # z = np.linspace(supnva["z"].min(), supnva["z"].max(), 1000)
    # fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

    # cos_a = BackgroundCosmology(h0=0.7, OmegaM0=0.3, OmegaK0=0.0)
    # cos_b = BackgroundCosmology(
    #     h0=startPos[0], OmegaM0=startPos[1], OmegaK0=startPos[2]
    # )
    # cos_a.solve()
    # cos_b.solve()

    # ax.errorbar(
    #     supnva["z"], supnva["d_L"], yerr=supnva["unc_d_L"], fmt=".", ms=3, c="k"
    # )

    # ax.plot(z, lcdm.dL(x(z)) / (1e3 * const.Mpc), label="Fiducial")
    # ax.plot(z, cos_a.dL(x(z)) / (1e3 * const.Mpc), label="(0.7, 0.3, 0)")
    # ax.plot(z, cos_b.dL(x(z)) / (1e3 * const.Mpc), label="Start pos")

    # ax.set_xlabel("redshift (z)")
    # ax.set_ylabel(r"$d_L$ (Gpc.)")
    # ax.set_title(r"Supernova $d_L$ vs. redshift")

    # ax.legend()

    # fig.savefig(path.join(dest, "supernovea"))
    # plt.close(fig)


if __name__ == "__main__":
    main()
