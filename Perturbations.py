from os import path
import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from matplotlib import pyplot as plt

from Global import const
from Global import APS_COL_W as apsw
import BackgroundCosmology
import RecombinationHistory

# A system to have control over where each
# quantity is in the ODE arrays (optional to use)
i = {
    "deltaCDM": 0,
    "vCDM": 1,
    "deltaB": 2,
    "vB": 3,
    "Phi": 4,
    "theta": 5,
}


class Perturbations:
    """
    This is a class for solving the perturbation history of the Universe
    After solving it holds functions for all perturbations and source functions

    Input Parameters:
      cosmo (BackgroundCosmology) : The cosmology we use to integrate perturbations
      rec   (RecombinationHistory): The recombination history we use to integrate perturbations
      keta_max             (float): The maximum k*eta(x=0) to integrate up to
      npts_k               (int)  : Number of k-values from kmin to kmax
      n_ell_theta          (int)  : Number of theta multipoles to include when solving the ODE

    Functions:
      deltaCDM             (k float,x float->float) : CDM density contrast as function of wavenumber k and x = log(a)
      deltaB               (k float,x float->float) : Baryon density contrast as function of wavenumber k and x = log(a)
      vCDM                 (k float,x float->float) : CDM velocity as function of wavenumber k and x = log(a)
      vB                   (k float,x float->float) : Baryon velocity as function of wavenumber k and x = log(a)
      Phi                  (k float,x float->float) : gii=(1+2Phi)a^2 metric potential as function of wavenumber k and x = log(a)
      Psi                  (k float,x float->float) : g00=-(1+2Psi) metric potential as function of wavenumber k and x = log(a)
      sourceT              (k float,x float->float) : Temperature source function as function of wavenumber k and x = log(a)
      Theta                (k float,x float, ell int->float) : Photon multipoles Theta_ell (ell=0,1,2) as function of wavenumber k and x = log(a)
    """

    # Settings x-integration
    x_start = np.log(1e-8)
    x_end = np.log(1.0)
    npts_x = 1000
    npts_tight = int(0.1 * npts_x)

    # Tight coupling hard max (start of recomb.)
    _tight_hard_stop = -8.3

    def __init__(
        self,
        BackgroundCosmology,
        RecombinationHistory,
        keta_max=3000.0,
        npts_k=100,
        n_ell_theta=10,
    ):
        """
        Intitialize the object
        """
        self.cosmo = BackgroundCosmology
        self.rec = RecombinationHistory
        self.k_min = 1.0 / self.cosmo.eta(0.0)
        self.k_max = keta_max / self.cosmo.eta(0.0)
        self.npts_k = npts_k

        # Number of ells to include and the total number of quantities in the ODE system
        self.n_ell_theta = n_ell_theta
        self.n_tot_tight = 5 + 2
        self.n_tot_full = 5 + n_ell_theta

        self.R0 = (4 * self.cosmo.OmegaR0) / (3 * self.cosmo.OmegaB0)

        # Compute x where tau' > 10
        self._set_tight_limit_tau_only()

        # PhD: you need to add polarization and neutrinos here and in the ODEs below
        # (and don't forget the differences between the tight coupling and full ODE)
        # ...

    def check_for_splines(self):
        if not hasattr(self, "splines") or len(self.splines) == 0:
            raise NameError("The perturbation splines have not been created")

    # =========================================================================
    # Functions availiable after solving
    # =========================================================================

    # def sourceT(self, k, x):
    #     self.check_for_splines()
    #     return self.sourceT_spline(k, x)

    def deltaCDM(self, kx):
        self.check_for_splines()
        return self.splines[i["deltaCDM"]](kx)

    def deltaB(self, kx):
        self.check_for_splines()
        return self.splines[i["deltaB"]](kx)

    def vCDM(self, kx):
        self.check_for_splines()
        return self.splines[i["vCDM"]](kx)

    def vB(self, kx):
        self.check_for_splines()
        return self.splines[i["vB"]](kx)

    def Phi(self, kx):
        self.check_for_splines()
        return self.splines[i["Phi"]](kx)

    def Psi(self, kx):
        self.check_for_splines()
        return self.psi_spline(kx)

    def Theta(self, kx, ell):
        self.check_for_splines()
        return self.splines[i["theta"] + ell](kx)

    # =========================================================================
    # =========================================================================
    # =========================================================================

    def info(self):
        """
        Print some useful info
        """
        print("")
        print("Perturbations:")
        print("kmin (1/Mpc): ", self.k_min * const.Mpc)
        print("kmax (1/Mpc): ", self.k_max * const.Mpc)
        print("Number of ell's: ", self.n_ell_theta)
        print("Number of k-points: ", self.npts_k)

    def solve(self):
        """
        The main driver for doing all the solving
        """
        self.integrate_perturbations()

    def plot(self, kval, url):
        """
        Plot the perturbations and source function as function of x = log(a) for a single value of k
        """
        x_array = np.linspace(self.x_start, self.x_end, self.npts_x)
        kmpc = "{:.3g}".format(kval * const.Mpc)

        k = np.ones_like(x_array) * kval

        kx = np.array((k, x_array)).T

        # Fetch data from splines
        data_deltaCDM = self.deltaCDM(kx)
        data_deltaB = self.deltaB(kx)
        data_vCDM = self.vCDM(kx)
        data_vB = self.vB(kx)
        data_Phi = self.Phi(kx)
        data_Psi = self.Psi(kx)
        data_Theta0 = self.Theta(kx, 0)
        data_Theta1 = self.Theta(kx, 1)

        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        ax.set_yscale("log")
        ax.set_title("Density perturbations k = " + str(kmpc) + "/ Mpc")
        ax.plot(x_array, np.abs(data_deltaB), label="deltaB")
        ax.plot(x_array, data_deltaCDM, label="deltaCDM")
        ax.plot(x_array, np.abs(3.0 * data_Theta0), label="deltaR")
        ax.legend()

        fig.savefig(path.join(url, "density"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        ax.set_yscale("log")
        ax.set_title("Velocity perturbations k = " + str(kmpc) + "/ Mpc")
        ax.plot(x_array, np.abs(data_vB), label="|vB|")
        ax.plot(x_array, np.abs(data_vCDM), label="|vCDM|")
        ax.plot(x_array, np.abs(-3.0 * data_Theta1), label="|vR|")

        fig.savefig(path.join(url, "velocity"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
        ax.set_yscale("log")
        ax.set_title("Potentials k = " + str(kmpc) + "/ Mpc")
        ax.plot(x_array, data_Phi, label="Phi")
        ax.plot(x_array, data_Psi, label="Psi")
        fig.savefig(path.join(url, "potentials"))
        plt.close(fig)

        # Plot the temperature source for the min and max k
        # plt.title("Source function k = " + str(kmpc) + "/ Mpc")
        # sourceT_data = self.sourceT_spline(kval, x_array)[0, :]
        # plt.plot(x_array, sourceT_data, label="Source")
        # plt.show()

    # =========================================================================
    # =========================================================================
    # =========================================================================

    def integrate_perturbations(self):
        """
        Integrate the tight coupling and the full system
        Combine the data from both and use this to make splines
        for all the quantities. Use this to make splines of the
        source function(s)
        """
        # Set up k-array
        ks = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num=self.npts_k)
        x_tc_end = -8.3
        x_tight = np.linspace(self.x_start, x_tc_end, self.npts_tight)
        x_full = np.linspace(x_tc_end, self.x_end, self.npts_x - self.npts_tight)
        print("x_full", x_full.shape)
        print("x_tight", x_tight.shape)
        x = np.concat((x_tight, x_full[1:]))

        print("")
        print("Start integrating perturbations")

        # 3D array to store the data
        results = np.zeros((self.n_tot_full, self.npts_k, len(x)))

        # Psi isn't solved differentially,
        # so is stored seperately
        Psi = np.zeros((self.npts_k, len(x)))

        # Loop over all k-values
        for idx, k in enumerate(ks):
            # Compute tight coupling time and set up x-arrays
            # x_tc_end = self.get_x_end_tight_coupling(k)

            print("Current k:", k * const.Mpc)

            # Compute IC for tight coupling
            y_tc_ic = self.get_ic(self.x_start, k)

            # Solve the tight coupling ODE
            sol_tight = integrate.solve_ivp(
                self.rhs_tight_coupling,
                [self.x_start, x_tc_end],
                y_tc_ic,
                t_eval=x_tight,
                rtol=1e-8,
                atol=1e-6,
                args=(k,),
            )
            assert (
                sol_tight.success
            ), f"Failed to find tight solution, {sol_tight.message}"

            results[: self.n_tot_tight, idx, : self.npts_tight] = sol_tight.y

            # calculate the non-differential tight values
            self.set_analytic_tight_values(results, x_tight, k, idx)

            y_full_ic = results[:, idx, self.npts_tight - 1]

            # Solve the full ODE
            sol_full = integrate.solve_ivp(
                self.rhs_full,
                [x_tc_end, self.x_end],
                y_full_ic,
                t_eval=x_full,
                rtol=1e-8,
                atol=1e-6,
                args=(k,),
            )
            assert sol_full.success, f"Failed to find full solution, {sol_full.message}"
            results[:, idx, self.npts_tight :] = sol_full.y[:, 1:]

            Psi[idx] = self.get_Psi(x, results[:, idx, :], k)

            # Compute quantities not solved for in the tight coupling regime
            # XXX TODO XXX

            # Combine arrays from the two regimes and store the data
            # XXX TODO XXX
            # e.g. cur_deltaCDM = np.concatenate((sol_tight.y[self.index_deltaCDM], sol_full.y[self.index_deltaCDM]))

            # Store the data
            # XXX TODO XXX
            # e.g. deltaCDM_data[ik,:] = cur_deltaCDM

        self.results = results

        self.splines = []

        for result in self.results:
            self.splines.append(
                interpolate.RegularGridInterpolator((ks, x), result, method="cubic")
            )

        self.psi_spline = interpolate.RegularGridInterpolator(
            (ks, x), Psi, method="cubic"
        )

        # self.Pi_spline       = RectBivariateSpline(k_array, x_array, Pi_data      )

        # Compute and spline the source-function (milestone 4)
        # XXX TODO XXX
        # self.sourceT_spline = RectBivariateSpline(k_array, x_array, sourceT_data)

        return

    def get_ic(self, x, k):
        """
        Set IC for the tight coupling system at given x = log(a) and wavenumber k
        """
        y = np.zeros(self.n_tot_tight)

        # Cosmological variables
        Hp = self.cosmo.Hp(x)
        ckHp = const.c * k / Hp
        OmegaNu = self.cosmo.OmegaNu(x)
        OmegaRtot = self.cosmo.OmegaRtot(x)
        f_nu = OmegaNu / OmegaRtot

        # Compute IC
        Psi = -1.0 / (1.5 + 2.0 * f_nu / 5.0)
        y[i["deltaCDM"]] = -(3 / 2) * Psi
        y[i["vCDM"]] = -0.5 * ckHp * Psi
        y[i["deltaB"]] = y[i["deltaCDM"]]
        y[i["vB"]] = y[i["vCDM"]]
        y[i["Phi"]] = -(1 + 2 * f_nu / 5) * Psi
        y[i["theta"] + 0] = -0.5 * Psi
        y[i["theta"] + 1] = 0.5 * ckHp * Psi

        return y

    def rhs_full(self, x, y, k):
        """
        Set the right hand side of the full ODE system dy/dx = RHS
        for a given value of x. The wavenumber k is set in the global variable k_current
        """
        # The array we are to fill and return
        dydx = np.zeros(self.n_tot_full)

        O2 = y[i["theta"] + 2]  # w/o polarization

        # --------- Same as tight regime -------
        H0 = self.cosmo.H0
        Hp = self.cosmo.Hp(x)
        ckHp = (const.c * k) / Hp
        a = np.exp(x)
        R = self.R0 / a
        dt = self.rec.dtau(x)
        O0 = y[i["theta"]]
        O1 = y[i["theta"] + 1]
        vB = y[i["vB"]]
        Phi = y[i["Phi"]]

        Psi = -Phi - 12 * (H0 / (const.c * k * a)) ** 2 * (
            self.cosmo.OmegaR0 * O2
        )  # N_2 = 0; i.e. no neutrinos
        dPhi = (
            Psi
            - (1 / 3) * ckHp**2 * Phi
            + 0.5
            * (H0 / Hp) ** 2
            * (
                self.cosmo.OmegaCDM0 * y[i["deltaCDM"]] / a
                + self.cosmo.OmegaB0 * y[i["deltaB"]] / a
                + 4 * self.cosmo.OmegaR0 / a**2 * O0
            )
        )
        dO0 = -ckHp * O1 - dPhi

        dydx[i["deltaCDM"]] = ckHp * y[i["vCDM"]] - 3 * dPhi
        dydx[i["vCDM"]] = -y[i["vCDM"]] - ckHp * Psi
        dydx[i["deltaB"]] = ckHp * vB - 3 * dPhi
        dydx[i["theta"] + 0] = dO0
        dydx[i["Phi"]] = dPhi
        # ------------------------------------

        dydx[i["theta"] + 1] = ckHp / 3 * (O0 - 2 * O2 + Psi) + dt * (O1 + vB / 3)
        dydx[i["vB"]] = -vB - ckHp * Psi + dt * R * (3 * O1 + vB)

        # Thetas 2 < ell < lmax
        for l in range(2, self.n_ell_theta - 1):
            idx = i["theta"] + l
            Pi = O2 if l == 2 else 0
            dydx[i["theta"] + l] = ckHp / (2 * l + 1) * (
                l * y[idx - 1] - (l + 1) * y[idx + 1]
            ) + dt * (y[idx] - Pi)

        # Theta lmax
        Olmax = y[i["theta"] + self.n_ell_theta - 1]
        dydx[i["theta"] + self.n_ell_theta - 1] = (
            ckHp
            * (
                y[i["theta"] + self.n_ell_theta - 2]
                - (l + 1) / (k * self.cosmo.eta(x)) * Olmax
            )
            + dt * Olmax
        )
        return dydx

    def rhs_tight_coupling(self, x, y, k):
        """
        Set the right hand side of the tight coupling ODE system dy/dx = RHS
        for a given value of x. The wavenumber k is set in the global variable k_current
        """
        # The array we are to fill and return
        dydx = np.zeros(self.n_tot_tight)

        H0 = self.cosmo.H0
        Hp = self.cosmo.Hp(x)
        dHp = self.cosmo.dHpdx(x)
        ckHp = (const.c * k) / Hp
        a = np.exp(x)
        R = self.R0 / a
        dt = self.rec.dtau(x)
        d2t = self.rec.d2tau(x)
        O0 = y[i["theta"]]
        O1 = y[i["theta"] + 1]
        O2 = -(20 / 45) * ckHp / dt * O1  # w/o polarization
        vB = y[i["vB"]]
        Phi = y[i["Phi"]]

        Psi = -Phi - 12 * (H0 / (const.c * k * a)) ** 2 * (
            self.cosmo.OmegaR0 * O2
        )  # N_2 = 0; i.e. no neutrinos
        dPhi = (
            Psi
            - (1 / 3) * ckHp**2 * Phi
            + 0.5
            * (H0 / Hp) ** 2
            * (
                self.cosmo.OmegaCDM0 * y[i["deltaCDM"]] / a
                + self.cosmo.OmegaB0 * y[i["deltaB"]] / a
                + 4 * self.cosmo.OmegaR0 / a**2 * O0
            )
        )
        dO0 = -ckHp * O1 - dPhi

        q = (
            -((1 - R) * dt + (1 + R) * d2t) * (3 * O1 + vB)
            - ckHp * Psi
            + (1 - dHp / Hp) * ckHp * (-O0 + 2 * O2)
            - ckHp * dO0
        ) / ((1 + R) * dt + dHp / Hp - 1)

        dvB = (1 / (1 + R)) * (
            -vB - ckHp * Psi + R * (q + ckHp * (-O0 + 2 * O2) - ckHp * Psi)
        )

        # Set the right hand side
        dydx[i["deltaCDM"]] = ckHp * y[i["vCDM"]] - 3 * dPhi
        dydx[i["vCDM"]] = -y[i["vCDM"]] - ckHp * Psi
        dydx[i["deltaB"]] = ckHp * vB - 3 * dPhi
        dydx[i["vB"]] = dvB
        dydx[i["Phi"]] = dPhi
        dydx[i["theta"] + 0] = dO0
        dydx[i["theta"] + 1] = 1 / 3 * (q - dvB)

        return dydx

    def get_Psi(self, x, y, k):
        return -y[i["Phi"]] - 12 * (self.cosmo.H0 / (const.c * k * np.exp(x))) ** 2 * (
            self.cosmo.OmegaR0 * y[i["theta"] + 2]
        )  # N_2 = 0; i.e. no neutrinos

    def _set_tight_limit_tau_only(self):
        """
        Independently find tau' > 10, so we don't need to do
        it each time
        """
        res = optimize.root_scalar(
            lambda x: np.abs(self.rec.dtau(x)) - 10, bracket=[-12, -6], x0=-8, rtol=1e-8
        )
        assert res.converged, f"Failed to find x for tau' > 10, {res.flag}"
        self._tight_stop_indep_k = min(res.root, self._tight_hard_stop)

    def set_analytic_tight_values(self, y, x, k, k_idx):
        for l in range(2, self.n_ell_theta):
            prev = y[i["theta"] + l - 1, k_idx, : self.npts_tight]
            y[i["theta"] + l, k_idx, : self.npts_tight] = (
                -l
                / (2 * l + 1)
                * const.c
                * k
                / (self.cosmo.Hp(x) * self.rec.dtau(x))
                * prev
            )

    def get_x_end_tight_coupling(self, k):
        """
        Compute when (x = log(a)) for when tight coupling ends for a given wavenumber k
        """
        res = optimize.root_scalar(
            lambda x: np.abs(self.rec.dtau(x)) - 10 * (const.c * k) / self.cosmo.Hp(x),
            bracket=[-12, -1],
            x0=-6,
            rtol=1e-6,
        )
        assert res.converged, f"Failed to find x for tau' > 10ck/Hp, {res.flag}"
        return min(self._tight_stop_indep_k, res.root)
