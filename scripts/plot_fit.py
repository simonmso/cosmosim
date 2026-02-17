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
    fig.savefig(path.join(dest, "omega_hist"))
    plt.close(fig)

    # report
    combined_res.report()

    # -------------- Individual ----------------
    indiv_res = MCMCResult(
        chain.T,
        log_prob,
        labels=["$h_0$", r"$\Omega_{M,0}$", r"$\Omega_{K,0}$"],
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


if __name__ == "__main__":
    main()
