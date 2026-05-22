import cython
from cython.cimports.FastSpline import FastSpline
from cython.cimports.BackgroundCosmology import BackgroundCosmology
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from os import path
import warnings

from Global import const
from Global import APS_COL_W as apsw


@cython.cclass
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
    """

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
        x_start=-15,
        x_end=0,
    ):

        # Settings for solver
        self.x_start = x_start
        self.x_end = x_end
        self.npts = 500
        self.npts_tau_before_reion = 1000
        self.npts_tau_during_reion = 1000
        self.npts_tau_after_reion = 1000
        self.Xe_saha_limit = 0.99

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
        return self.tau_spline(x)

    def dtau(self, x):
        return self.dtau_spline(x)

    @cython.cfunc
    def dtau_fast(self, x: cython.double) -> cython.double:
        return self.dtau_spline_fast.evaluate(x)

    def d2tau(self, x):
        return self.d2tau_spline(x)

    @cython.cfunc
    def d2tau_fast(self, x: cython.double) -> cython.double:
        return self.d2tau_spline_fast.evaluate(x)

    def d3tau(self, x):
        return self.d3tau_spline(x)

    def g_tilde(self, x):
        return -self.dtau(x) * np.exp(-self.tau(x))

    def dg_tilde(self, x):
        return (self.dtau(x) ** 2 - self.d2tau(x)) * np.exp(-self.tau(x))

    def d2g_tilde(self, x):
        t = self.tau(x)
        dt = self.dtau(x)
        d2t = self.d2tau(x)
        d3t = self.d3tau(x)
        return (-(dt**3) + 3 * dt * d2t - d3t) * np.exp(-t)

    def Xe(self, x):
        return np.exp(self.log_Xe_spline(x))

    def ne(self, x):
        return np.exp(self.log_ne_spline(x))

    def s(self, x):
        """Sound horizon"""
        return self.s_spline(x)

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

        self.solve_z_star()

        self.solve_sound_horizon()

    def plot(self, url):
        """
        Make some useful plots
        """
        npts = 10000
        xarr = np.linspace(self.x_start, self.x_end, num=npts)
        Xe = self.Xe(xarr)
        ne = self.ne(xarr)
        tau = self.tau(xarr)
        dtaudx = -self.dtau(xarr)
        dtau_fast = -np.array(list(map(self.dtau_fast, xarr)))
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

        fig.savefig(path.join(url, "g_tilde"))
        plt.close(fig)

        # Tau
        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))
        ax.set_title("Tau and derivatives")
        ax.plot(xarr, tau, label="tau")
        ax.plot(xarr, dtaudx, label="dtaudx")
        ax.plot(xarr, dtaudx, label="dtaudx")
        ax.plot(xarr, dtau_fast, label="dtau_fast")
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

        # Sound horizon
        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        ax.set_title("Sound horizon")
        ax.plot(xarr, self.s(xarr), label=r"$s(x)$")
        ax.axvline(np.log(1 / (1 + self.z_star)), label=r"$z_{*}$")

        ax.set_yscale("log")
        fig.savefig(path.join(url, "sound_horizon"))
        plt.close(fig)

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

        self.log_Xe_spline = interpolate.make_interp_spline(x, np.log(Xe))
        self.log_ne_spline = interpolate.make_interp_spline(x, np.log(ne))

    def _n_H(self, x):
        return self.cosmo.OmegaB0 * self.cosmo.rhoc0 / (const.m_H * np.exp(3 * x))

    def _T_b(self, x):
        return self.cosmo.TCMB0 / np.exp(x)

    def X_e_saha(self, x_inp):
        """
        Solve the Saha equations for hydrogen and helium recombination
        Returns: Xe, ne with Xe = ne/nH beging the free electron fraction
        and ne the electon number density
        """
        x = np.asarray(x_inp)

        n_H = self._n_H(x)

        n_b = n_H  # Is this only allowed for Saha?

        T_b = self._T_b(x)  # T_b approx.= T_gamma

        Ca = 1 / n_b
        Cb = ((const.m_e * T_b) / (2 * np.pi)) ** (3 / 2)
        Cc = np.exp(-const.epsilon_0 / T_b)

        C = Ca * Cb * Cc

        # Solve Saha equation for Xe
        Xe = (-C + np.sqrt(C**2 + 4 * C)) / 2

        # avoid huge - huge causing floating point errors
        if np.ndim(x) > 0:
            Xe[C > 1e11] = 1.0
        elif C > 1e11:
            Xe = 1.0

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
        beta_2 = (
            self._peebles_beta_coef
            * alpha_2
            * T_b ** (3 / 2)
            * np.exp(-const.epsilon_0 / (4 * T_b))
        )

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

    def X_e_peebles(self, x, X_e_0):
        sol = integrate.solve_ivp(
            self._peebles_dXedx,
            (x[0], x[-1]),
            [
                X_e_0,
            ],
            t_eval=x,
            rtol=1e-6,
            atol=1e-8,
        )
        assert sol.success, f"Peebles ivp failed, {sol.message}"
        return sol.y.flatten()

    def _dtau_dx(self, x, tau):
        """
        Right hand side of the optical depth ODE -dtaudx = RHS
        """
        return -(const.c * self.ne(x) * const.sigma_T) / (self.cosmo.H(x))

    def solve_for_optical_depth_tau(self):
        """
        Solve for the optical depth tau(x) by integrating up
        dtaudx = -c sigmaT ne/H
        (PhD: Include the effects of reionization if z_reion > 0)
        """
        x = np.linspace(0, self.x_start, num=self.npts)

        sol = integrate.solve_ivp(
            self._dtau_dx,
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

        self.tau_spline = interpolate.CubicSpline(x[::-1], tau[::-1])

        self.dtau_spline = self.tau_spline.derivative()
        self.d2tau_spline = self.tau_spline.derivative(2)
        self.d3tau_spline = self.tau_spline.derivative(3)

        self.dtau_spline_fast = FastSpline(self.dtau_spline)
        self.d2tau_spline_fast = FastSpline(self.d2tau_spline)

    def solve_z_star(self):
        """
        Find the redshift for tau = 1 (peak of the visibility function)
        """
        # Compute z_star (peak of visibility function or tau = 1)
        sol = optimize.root_scalar(
            lambda x: self.tau(x) - 1,
            method="brentq",
            bracket=[-12, -4],
            fprime=self.dtau,
            fprime2=self.d2tau,
            x0=-7,
            rtol=1e-8,
        )
        assert sol.converged, f"Failed to find z_star, {sol.flag}"
        self.z_star = (1 / np.exp(sol.root)) - 1

    def _dsdx(self, x, _, R0):
        R = R0 / np.exp(x)
        return (const.c * np.sqrt(R / (3 * (1 + R)))) / self.cosmo.Hp(x)

    def solve_sound_horizon(self):
        x = np.linspace(self.x_start, self.x_end, num=self.npts)

        R0 = (4 * self.cosmo.OmegaR0) / (3 * self.cosmo.OmegaB0)
        s_ini = self._dsdx(x[0], 0, R0)

        res = integrate.solve_ivp(
            self._dsdx,
            (self.x_start, self.x_end),
            y0=(s_ini,),
            args=(R0,),
            t_eval=x,
            dense_output=True,
        )
        assert res.success, f"Failed to find sound horizon, {res.message}"

        self.s_spline = lambda x: res.sol(x).flatten()
