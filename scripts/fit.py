import argparse
from os import path

import numpy as np
import pandas as pd
import emcee

from BackgroundCosmology import BackgroundCosmology
from Global import const


def x(z):
    a = 1 / (1 + z)
    return np.log(a)


# set up probability functions
def log_likelihood(theta, x, dL, unc_dL):
    h0, OmegaM0, OmegaK0 = theta
    cosmo = BackgroundCosmology(
        h0=h0,
        OmegaK0=OmegaK0,
        OmegaM0=OmegaM0,
    )
    # TODO: should other parameters also be allowed to vary?
    cosmo.solve()
    dL_theory = cosmo.dL(x)
    var = unc_dL**2
    return -0.5 * np.sum((dL - dL_theory) ** 2 / var + np.log(2 * np.pi * var))


# uniform prior
def log_prior(theta):
    h0, OmegaM0, OmegaK0 = theta

    h_ok = 0.1 < h0 < 1.5
    M_ok = 0.0 <= OmegaM0 <= 1.0
    K_ok = -1.0 <= OmegaK0 <= 1.0

    if h_ok and M_ok and K_ok:
        return 0.0
    return -np.inf


def log_prob(theta, x, dL, unc_dL):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return lp
    ll = log_likelihood(theta, x, dL, unc_dL)
    return lp + ll


def main():
    # Get data and figure paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", required=True, help="Directory of the data files"
    )
    args = parser.parse_args()

    # Read data
    supernova_path = path.join(args.data, "supernovadata.txt")
    supnva = pd.read_csv(
        supernova_path,
        sep=r"\s+",
        skiprows=1,
        names=["z", "d_L", "unc_d_L"],
    )

    # Run the MCMC
    startPos = np.array(
        [
            0.7,  # h0
            0.3,  # OmegaM0
            0.00001,  # OmegaK0
        ]
    )
    nwalkers = 64
    niters = 5000
    ndim = len(startPos)
    start = startPos + 0.001 * np.random.randn(nwalkers, ndim)

    x_val = x(supnva["z"])
    dL_val = supnva["d_L"] * (1e3 * const.Mpc)
    unc_dL_val = supnva["unc_d_L"] * (1e3 * const.Mpc)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        args=(x_val, dL_val, unc_dL_val),
    )
    sampler.run_mcmc(start, niters, progress=True)

    try:
        print("auto correlation time")
        print(sampler.get_autocorr_time())
    except emcee.autocorr.AutocorrError as e:
        print("!------ Chain too short ------!")
        print(e.args[0])

    np.savez(
        file=path.join(args.data, "supernova.npz"),
        chain=sampler.get_chain(flat=True),
        log_prob=sampler.get_log_prob(flat=True),
    )


if __name__ == "__main__":
    main()
