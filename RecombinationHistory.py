import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import interpolate
from os import path
import warnings

from Global import const
from Global import APS_COL_W as apsw
import BackgroundCosmology


class RecombinationHistory:
    """
    This is a class for solving the recombination (and reionization) history of the Universe.
    It holds recombination parameters and functions relevant for the recombination history.

    Input Parameters:
      cosmo (BackgroundCosmology) : The cosmology we use to solve for the recombination history
      Yp                   (float): Primordial helium fraction
      reionization         (bool) : Include reionization or not
      z_reion              (float): Reionization redshift
      delta_z_reion        (float): Reionization width
      helium_reionization  (bool) : Include helium+ reionization
      z_helium_reion       (float): Reionization redshift for helium+
      delta_z_helium_reion (float): Reionization width for helium+

    Attributes:
      tau_reion            (float): The optical depth at reionization
      z_star               (float): The redshift for the LSS (defined as peak of visibility function or tau=1)

    Functions:
      tau_of_x             (float->float) : Optical depth as function of x=log(a)
      dtaudx_of_x          (float->float) : First x-derivative of optical depth as function of x=log(a)
      ddtauddx_of_x        (float->float) : Second x-derivative of optical depth as function of x=log(a)
      g_tilde_of_x         (float->float) : Visibility function dexp(-tau)dx as function of x=log(a)
      dgdx_tilde_of_x      (float->float) : First x-derivative of visibility function as function of x=log(a)
      ddgddx_tilde_of_x    (float->float) : Second x-derivative of visibility function as function of x=log(a)
      Xe_of_x              (float->float) : Free electron fraction dXedx as function of x=log(a)
      ne_of_x              (float->float) : Electron number density as function of x=log(a)
    """

    # Settings for solver
    x_start = -12
    x_end = 0
    npts = 500
    npts_tau_before_reion = 1000
    npts_tau_during_reion = 1000
    npts_tau_after_reion = 1000
    Xe_saha_limit = 0.99

    def __init__(
        self,
        BackgroundCosmology,
        Yp=0.24,
        reionization=False,
        z_reion=11.0,
        delta_z_reion=0.5,
        helium_reionization=False,
        z_helium_reion=3.5,
        delta_z_helium_reion=0.5,
    ):
        self.cosmo = BackgroundCosmology

        self.Yp = Yp

        self.reionization = reionization
        self.z_reion = z_reion
        self.delta_z_reion = delta_z_reion

        self.helium_reionization = helium_reionization
        self.z_helium_reion = z_helium_reion
        self.delta_z_helium_reion = delta_z_helium_reion

    # =========================================================================
    # Methods availiable after solving
    # =========================================================================

    def tau_of_x(self, x):
        if not hasattr(self, "tau_of_x_spline"):
            raise NameError("The spline tau_of_x_spline has not been created")
        return self.tau_of_x_spline(x)

    def dtaudx_of_x(self, x):
        if not hasattr(self, "dtaudx_of_x_spline"):
            raise NameError("The spline dtaudx_of_x_spline has not been created")
        return self.dtaudx_of_x_spline(x)

    def ddtauddx_of_x(self, x):
        if not hasattr(self, "ddtauddx_of_x_spline"):
            raise NameError("The spline ddtauddx_of_x_spline has not been created")
        return self.ddtauddx_of_x_spline(x)

    def g_tilde_of_x(self, x):
        if not hasattr(self, "g_tilde_of_x_spline"):
            raise NameError("The spline g_tilde_of_x_spline has not been created")
        return self.g_tilde_of_x_spline(x)

    def dgdx_tilde_of_x(self, x):
        if not hasattr(self, "dgdx_tilde_of_x_spline"):
            raise NameError("The spline dgdx_tilde_of_x_spline has not been created")
        return self.dgdx_tilde_of_x_spline(x)

    def ddgddx_tilde_of_x(self, x):
        if not hasattr(self, "ddgddx_tilde_of_x_spline"):
            raise NameError("The spline ddgddx_tilde_of_x_spline has not been created")
        return self.ddgddx_tilde_of_x_spline(x)

    def Xe_of_x(self, x):
        if not hasattr(self, "log_Xe_of_x_spline"):
            raise NameError("The spline log_Xe_of_x_spline has not been created")
        return np.exp(self.log_Xe_of_x_spline(x))

    def ne_of_x(self, x):
        if not hasattr(self, "log_ne_of_x_spline"):
            raise NameError("The spline log_ne_of_x_spline has not been created")
        return np.exp(self.log_ne_of_x_spline(x))

    # =========================================================================
    # =========================================================================
    # =========================================================================

    def info(self):
        print("")
        print("Recombination History:")
        print("Yp:                   %8.7f" % self.Yp)
        print("reionization:         %8.7f" % self.reionization)
        print("z_reion:              %8.7f" % self.z_reion)
        print("delta_z_reion:        %8.7f" % self.delta_z_reion)
        print("helium_reionization:  %8.7f" % self.helium_reionization)
        print("z_helium_reion:       %8.7f" % self.z_helium_reion)
        print("delta_z_helium_reion: %8.7f" % self.delta_z_helium_reion)

    def solve(self):
        """
        Main driver for doing all the solving
        We first compute Xe(x) and ne(x)
        Then we compute the optical depth tau(x) and the visibility function g(x)
        """
        self.solve_number_density_electrons()

        # self.solve_for_optical_depth_tau()

        # Compute z_star (peak of visibility function or tau = 1)
        # XXX TODO XXX

        # PhD: compute optical depth at reionization
        # XXX TODO XXX

    def plot(self, url):
        """
        Make some useful plots
        """

        npts = 10000
        xarr = np.linspace(self.x_start, self.x_end, num=npts)
        Xe = self.Xe_of_x(xarr)
        ne = self.ne_of_x(xarr)
        # tau = [self.tau_of_x(xarr[i]) for i in range(npts)]
        # dtaudx = [-self.dtaudx_of_x(xarr[i]) for i in range(npts)]
        # ddtauddx = [self.ddtauddx_of_x(xarr[i]) for i in range(npts)]
        # g_tilde = self.g_tilde_of_x(xarr)
        # dgdx_tilde = self.dgdx_tilde_of_x(xarr)
        # ddgddx_tilde = self.ddgddx_tilde_of_x(xarr)

        # Recombination g_tilde
        # plt.xlim(-7.5, -6.5)
        # plt.ylim(-4, 6)
        # plt.title("Visibility function and derivatives close to recombination")
        # plt.plot(xarr, g_tilde, xarr, dgdx_tilde / 15.0, xarr, ddgddx_tilde / 300.0)
        # plt.show()

        # Reionization g_tilde
        # plt.xlim(-2.7, -2.0)
        # plt.ylim(-0.15, 0.15)
        # plt.title("Visibility function and derivatives close to reionization")
        # plt.plot(xarr, g_tilde, xarr, dgdx_tilde / 15.0, xarr, ddgddx_tilde / 300.0)
        # plt.show()

        fig, axs = plt.subplots(nrows=2, figsize=(apsw, 1.5 * apsw))

        # Xe(x) of x
        ax = axs[0]
        ax.set_title("Free electron fraction")
        ax.plot(xarr, Xe)
        ax.set_yscale("log")

        # ne of x
        ax = axs[1]
        ax.set_yscale("log")
        ax.set_title("Electron numberdensity")
        ax.plot(xarr, ne)

        fig.savefig(path.join(url, "Xe_and_ne"))
        plt.close(fig)

        # # tau
        # plt.yscale("log")
        # plt.title("Tau and derivatives")
        # plt.ylim(1e-8, 1e8)
        # plt.plot(xarr, tau, xarr, dtaudx, xarr, ddtauddx)
        # plt.show()

    # =========================================================================
    # =========================================================================
    # =========================================================================

    def solve_number_density_electrons(self):
        """
        Solve for the evolution of the electron number density by solving
        the Saha and Peebles equations
        """
        # Equations are missing some factors of c and h (probably),
        # so these methods only works in planck units
        assert const.name == "Planck"

        x = np.linspace(self.x_start, self.x_end, num=self.npts)
        Xe = np.zeros_like(x)
        ne = np.zeros_like(x)

        Xe_saha = self.X_e_saha(x)

        saha_rg = Xe_saha > self.Xe_saha_limit  # range where saha applies
        peebles_rg = Xe_saha <= self.Xe_saha_limit

        peebles_init_val = max(Xe_saha[peebles_rg])

        Xe_peebles = self.X_e_peebles(x[peebles_rg], X_e_0=peebles_init_val)

        Xe[saha_rg] = Xe_saha[saha_rg]
        Xe[peebles_rg] = Xe_peebles

        ne = self._n_H(x) * Xe

        self.log_Xe_of_x_spline = interpolate.make_interp_spline(x, np.log(Xe))
        self.log_ne_of_x_spline = interpolate.make_interp_spline(x, np.log(ne))

        # # Calculate recombination history
        # for i in range(npts):
        #     # Current scale factor
        #     x = x_array[i]
        #     a = np.exp(x)

        #     # ==============================================================
        #     # Get f_e from solving the Saha equation
        #     # ==============================================================
        #     Xe_current, ne_current = self.electron_fraction_from_saha_equation(x)

        #     # Two regimes: Saha and Peebles regime
        #     if Xe_current > self.Xe_saha_limit:

        #         # Store the results from the Saha equation
        #         Xe_arr[i] = Xe_current
        #         ne_arr[i] = ne_current

        #     else:

        #         # ==============================================================
        #         # We need to solve the Peebles equation for the rest of the time
        #         # ==============================================================

        #         # Make x-array for Peebles system from current time till the end
        #         # XXX TODO XXX

        #         # Set initial conditions
        #         # XXX TODO XXX

        #         # Solve the Peebles ODE
        #         # XXX TODO XXX

        #         # Fill up array with the result
        #         # XXX TODO XXX

        #         # We are done so exit for loop
        #         break

        # Make splines of log(Xe) and log(ne) as function of x = log(a)
        # XXX TODO XXX
        # self.log_Xe_of_x_spline = CubicSpline(x_array, np.log(Xe_arr))
        # self.log_ne_of_x_spline = CubicSpline(x_array, np.log(ne_arr))

        return

    def solve_for_optical_depth_tau(self):
        """
        Solve for the optical depth tau(x) by integrating up
        dtaudx = -c sigmaT ne/H
        (PhD: Include the effects of reionization if z_reion > 0)
        """

        # Set up x_array
        # XXX TODO XXX

        # Set initial conditions for tau
        # XXX TODO XXX

        # Solve the tau ODE and normalize it such that tau(0) = 0.0
        # XXX TODO XXX

        # Spline it up
        # XXX TODO XXX
        # self.tau_of_x_spline = CubicSpline(x_array, tau)

        # Compute and spline the derivatives of tau
        # XXX TODO XXX
        # self.dtaudx_of_x_spline = CubicSpline(x_array, dtaudx)
        # self.ddtauddx_of_x_spline = CubicSpline(x_array, ddtauddx)

        # Compute and spline visibility function and it derivatives
        # XXX TODO XXX
        # self.g_tilde_of_x_spline  = CubicSpline(x_array, g_tilde_of_x)
        # self.dgdx_tilde_of_x_spline = CubicSpline(x_array, dgdx_tilde_of_x)
        # self.ddgddx_tilde_of_x_spline = CubicSpline(x_array, ddgddx_tilde_of_x)

        return

    def _n_H(self, x):
        return self.cosmo.OmegaB0 * self.cosmo.rhoc0 / (const.m_H * np.exp(3 * x))

    def _T_b(self, x):
        return self.cosmo.TCMB0 / np.exp(x)

    def X_e_saha(self, x):
        """
        Solve the Saha equations for hydrogen and helium recombination
        Returns: Xe, ne with Xe = ne/nH beging the free electron fraction
        and ne the electon number density
        """
        a = np.exp(x)

        n_H = self._n_H(x)

        n_b = n_H  # Is this only allowed for Saha?

        T_b = self._T_b(x)  # T_b approx.= T_gamma

        Ca = 1 / n_b
        Cb = ((const.m_e * T_b) / (2 * np.pi)) ** (3 / 2)
        Cc = np.exp(-const.epsilon_0 / T_b)

        C = Ca * Cb * Cc

        # Solve Saha equation for Xe
        Xe = (-C + np.sqrt(C**2 + 4 * C)) / 2

        # Return Xe and ne
        return Xe

    # -------------- Peebles helpers -------------
    _peebles_alpha_2_coef = (
        24 / np.sqrt(27 * np.pi) * const.sigma_T * np.sqrt(const.epsilon_0)
    )

    _peebles_beta_coef = (const.m_e / (np.pi * 2)) ** (3 / 2)

    _peebles_lam_2s1s = 8.227

    _peebles_lam_alpha_coef = ((3 * const.epsilon_0) ** 3) / ((8 * np.pi) ** 2)

    def _peebles_dXedx(self, x, X_e):
        T_b = self._T_b(x)

        phi_2 = 0.448 * np.log(const.epsilon_0 / T_b)
        alpha_2 = self._peebles_alpha_2_coef * phi_2 / np.sqrt(T_b)
        beta = (
            self._peebles_beta_coef
            * alpha_2
            * T_b ** (3 / 2)
            * np.exp(-const.epsilon_0 / T_b)
        )
        beta_2 = 0.0 if beta == 0 else beta * np.exp(3 * const.epsilon_0 / (4 * T_b))

        n_H = self._n_H(x)
        n_1s = (1.0 - X_e) * n_H
        H = self.cosmo.H(x)
        lam_alpha = H * self._peebles_lam_alpha_coef / n_1s
        C_r = (self._peebles_lam_2s1s + lam_alpha) / (
            self._peebles_lam_2s1s + lam_alpha + beta_2
        )

        ret = (C_r / H) * (beta * (1 - X_e) - n_H * alpha_2 * X_e**2)

        return ret

    # -----------------------------------------------

    def X_e_peebles(self, x, X_e_0=Xe_saha_limit):
        sol = integrate.solve_ivp(
            self._peebles_dXedx,
            (x[0], x[-1]),
            [
                X_e_0,
            ],
            t_eval=x,
            method="BDF",
            rtol=1e-6,
            atol=1e-8,
        )
        assert sol.success, f"Peebles ivp failed, {sol.message}"
        return sol.y.flatten()

    def rhs_tau_ode(self, x, y):
        """
        Right hand side of the optical depth ODE dtaudx = RHS
        """

        # Set the right hand side
        # XXX TODO XXX
        dtaudx = 1.0
        return dtaudx
