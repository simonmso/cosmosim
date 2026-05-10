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

    def tau(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return self.tau_spline(x)

    def dtau(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return self.dtau_spline(x)

    def d2tau(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return self.d2tau_spline(x)

    def d3tau(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return self.d3tau_spline(x)

    def g_tilde(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return -self.dtau(x) * np.exp(-self.tau(x))

    def dg_tilde(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        return (self.dtau(x) ** 2 - self.d2tau(x)) * np.exp(-self.tau(x))

    def d2g_tilde(self, x):
        if not hasattr(self, "tau_spline"):
            raise NameError("The spline tau_spline has not been created")
        t = self.tau(x)
        dt = self.dtau(x)
        d2t = self.d2tau(x)
        d3t = self.d3tau(x)
        return (-(dt**3) + 3 * dt * d2t - d3t) * np.exp(-t)

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

        self.solve_for_optical_depth_tau()

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
        tau = self.tau(xarr)
        dtaudx = -self.dtau(xarr)
        ddtaudx = self.d2tau(xarr)

        g_tilde = self.g_tilde(xarr)
        dgdx_tilde = self.dg_tilde(xarr)
        ddgddx_tilde = self.d2g_tilde(xarr)

        # Reionization g_tilde
        # plt.xlim(-2.7, -2.0)
        # plt.ylim(-0.15, 0.15)
        # plt.title("Visibility function and derivatives close to reionization")
        # plt.plot(xarr, g_tilde, xarr, dgdx_tilde / 15.0, xarr, ddgddx_tilde / 300.0)
        # plt.show()

        # Recombination g_tilde
        fig, axs = plt.subplots(ncols=3, figsize=(2 * apsw, 0.7 * apsw))

        fig.suptitle("Visibility function and derivatives close to recombination")

        ax = axs[0]
        ax.set_title("g_tilde")
        ax.plot(xarr, g_tilde)

        ax = axs[1]
        ax.set_title("dgdx_tilde")
        ax.plot(xarr, dgdx_tilde)

        ax = axs[2]
        ax.set_title("ddgddx_tilde")
        ax.plot(xarr, ddgddx_tilde)

        # ax.legend()

        fig.savefig(path.join(url, "g_tilde"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
        ax.set_title("Tau and derivatives")
        ax.plot(xarr, tau, label="tau")
        ax.plot(xarr, dtaudx, label="dtaudx")
        ax.plot(xarr, ddtaudx, label="ddtaudx")
        ax.set_yscale("log")

        ax.legend()

        fig.savefig(path.join(url, "tau"))
        plt.close(fig)

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

    def _dtau_dx(self, x, tau):
        """
        Right hand side of the optical depth ODE -dtaudx = RHS
        """
        return -(const.c * self.ne_of_x(x) * const.sigma_T) / (self.cosmo.H(x))

    def solve_for_optical_depth_tau(self):
        """
        Solve for the optical depth tau(x) by integrating up
        dtaudx = -c sigmaT ne/H
        (PhD: Include the effects of reionization if z_reion > 0)
        """
        # reverse_derivative = lambda neg_x, tau: -self._dtau_dx(-neg_x, tau)

        x = np.linspace(0, self.x_start, num=self.npts)

        sol = integrate.solve_ivp(
            self._dtau_dx,
            # reverse_derivative,
            (0, self.x_start),
            [
                0,
            ],
            t_eval=x,
            rtol=1e-5,
            atol=1e-10,
        )

        assert sol.success, f"Tau ivp failed, {sol.message}"

        tau = sol.y.flatten()

        self.tau_spline = interpolate.make_interp_spline(x[::-1], tau[::-1])

        self.dtau_spline = self.tau_spline.derivative()
        self.d2tau_spline = self.tau_spline.derivative(2)
        self.d3tau_spline = self.tau_spline.derivative(3)
