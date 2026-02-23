from os import path
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate

from Global import const
from Global import APS_COL_W as apsw


class BackgroundCosmology:
    """
    This is a class for the cosmology at the background level.
    It holds cosmological parameters and functions relevant for the background.

    Input Parameters:
      h           (float): The little Hubble parameter h in H0 = 100h km/s/Mpc
      OmegaB0      (float): Baryonic matter density parameter at z = 0
      OmegaCDM0    (float): Cold dark matter density parameter at z = 0
      OmegaK0      (float,optional): Curvative density parameter at z = 0
      name        (float,optional): A name for describing the cosmology
      TCMB        (float,optional): The temperature of the CMB today in Kelvin. Fiducial value is 2.725K
      Neff        (float,optional): The effective number of relativistic neutrinos
      OmegaR0     (float,optional): Total radiation density. Overrides TCMB and Neff
      OmegaM0     (float,optional): Total matter density. Overrides OmegaB0 and OmegaCDM
      x_pts       ([float],optional): values of x to solve for

    Attributes:
      OmegaR0      (float): Radiation matter density parameter at z = 0
      OmegaNu0     (float): Massless neutrino density parameter at z = 0
      OmegaM0      (float): Total matter (CDM+b+mnu) density parameter at z = 0
      OmegaK0      (float): Curvature density parameter at z = 0

    Functions:
      eta             (float->float) : Conformal time times c (units of length) as function of x=log(a)
      H               (float->float) : Hubble parameter as function of x=log(a)
      dHdx            (float->float) : First derivative of hubble parameter as function of x=log(a)
      Hp              (float->float) : Conformal hubble parameter H*a as function of x=log(a)
      dHpdx           (float->float) : First derivative of conformal hubble parameter as function of x=log(a)
    """

    def __init__(
        self,
        h0=0.7,
        OmegaB0=0.046,
        OmegaCDM0=0.224,
        OmegaK0=0.0,
        name="FiducialCosmology",
        TCMB_in_K=2.725,
        Neff=3.046,
        OmegaR0=None,
        OmegaM0=None,
        x_pts=None,
    ):
        self.OmegaK0 = OmegaK0
        self.h0 = h0
        self.H0 = const.H0_over_h * h0
        self.name = name
        self.Neff = Neff

        # Set the constants
        self.rhoc0 = (3.0 * self.H0**2) / (
            8.0 * np.pi * const.G
        )  # Critical density today

        if OmegaR0 is not None:
            self.OmegaR0 = OmegaR0
            self.TCMB0 = None
        else:
            self.TCMB0 = TCMB_in_K * const.K
            self.OmegaR0 = (
                2
                * (np.pi**2.0 / 30.0)
                * ((const.k_b * self.TCMB0) ** 4 / ((const.hbar**3) * (const.c**5)))
                * (8.0 * np.pi * const.G / (3.0 * self.H0**2))
            )  # Radiation

        if OmegaM0 is not None:
            self.OmegaM0 = OmegaM0  # Total matter
            self.OmegaB0 = None
            self.OmegaCDM0 = None
        else:
            self.OmegaM0 = OmegaB0 + OmegaCDM0  # Total matter
            self.OmegaB0 = OmegaB0
            self.OmegaCDM0 = OmegaCDM0

        self.OmegaNu0 = (
            self.Neff * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * self.OmegaR0
        )  # Neutrino radiation
        self.OmegaR0tot = self.OmegaR0 + self.OmegaNu0  # Total radiation
        self.OmegaLambda0 = (
            1.0 - self.OmegaR0tot - self.OmegaM0 - self.OmegaK0
        )  # Dark energy (from Sum Omega_i = 1)

        # calculate equalities from analytic expressions
        self.x_rm = np.log(self.OmegaR0tot / self.OmegaM0)
        self.x_mlam = np.log(self.OmegaM0 / self.OmegaLambda0) / 3

        # Settings for integration and splines of eta
        if x_pts is not None:
            self.x_pts = x_pts
        else:
            self.x_pts = np.linspace(np.log(1e-8), np.log(1.0), num=1000)

        self.x_start = min(self.x_pts)
        self.x_end = max(self.x_pts)

    # =========================================================================
    # Methods availiable after solving
    # =========================================================================

    def eta(self, x):
        if not hasattr(self, "eta_ode_sol"):
            raise NameError("The spline eta_ode_sol has not been created")
        return self.eta_ode_sol(x).flatten()

    def t(self, x):
        if not hasattr(self, "t"):
            raise NameError("The spline t_ode_sol has not been created")
        return self.t_ode_sol(x).flatten()

    def H(self, x):
        a = np.exp(x)
        M = self.OmegaM0 * (a ** (-3))
        R = self.OmegaR0tot * (a ** (-4))
        K = self.OmegaK0 * (a ** (-2))
        L = self.OmegaLambda0

        return self.H0 * np.sqrt(M + R + K + L)

    def Hp(self, x):
        a = np.exp(x)
        return a * self.H(x)

    def dHdx(self, x):
        a = np.exp(x)
        M = self.OmegaM0 * (a ** (-3))
        R = self.OmegaR0tot * (a ** (-4))
        K = self.OmegaK0 * (a ** (-2))
        L = self.OmegaLambda0

        return (self.H0 / (2.0 * np.sqrt(M + R + K + L))) * (
            -3.0 * M - 4.0 * R - 2.0 * K
        )

    def dHpdx(self, x):
        a = np.exp(x)
        return a * (self.H(x) + self.dHdx(x))

    def d2Hdx2(self, x):
        a = np.exp(x)
        M = self.OmegaM0 * (a ** (-3))
        R = self.OmegaR0tot * (a ** (-4))
        K = self.OmegaK0 * (a ** (-2))
        L = self.OmegaLambda0
        H0 = self.H0
        H = self.H(x)
        dH = self.dHdx(x)

        A = -(1.0 / H**2) * dH * (-3.0 * M - 4.0 * R - 2.0 * K)
        B = (1.0 / H) * (9.0 * M + 16.0 * R + 4.0 * K)

        return (H0**2 / 2.0) * (A + B)

    def d2Hpdx2(self, x):
        a = np.exp(x)
        H = self.H(x)
        dH = self.dHdx(x)
        d2H = self.d2Hdx2(x)

        return a * (H + 2 * dH + d2H)

    def detadx(
        self,
        x,
        eta=None,  # unused param eta so the function can be passed directly to solve_ivp
    ):
        return const.c / self.Hp(x)

    def dtdx(self, x, t=None):
        return 1 / self.H(x)

    def OmegaK(self, x):
        a = np.exp(x)
        H = self.H(x)
        H0 = self.H0
        Om0 = self.OmegaK0
        return (Om0 * H0**2) / (a**2 * H**2)

    def OmegaCDM(self, x):
        return (self.OmegaCDM0 * self.H0**2) / (np.exp(x * 3) * self.H(x) ** 2)

    def OmegaB(self, x):
        return (self.OmegaB0 * self.H0**2) / (np.exp(x * 3) * self.H(x) ** 2)

    def OmegaR(self, x):
        return (self.OmegaR0 * self.H0**2) / (np.exp(x * 4) * self.H(x) ** 2)

    def OmegaNu(self, x):
        return (self.OmegaNu0 * self.H0**2) / (np.exp(x * 4) * self.H(x) ** 2)

    def OmegaLambda(self, x):
        return (self.OmegaLambda0 * self.H0**2) / (self.H(x) ** 2)

    def OmegaM(self, x):
        return (self.OmegaM0 * self.H0**2) / (np.exp(x * 3) * self.H(x) ** 2)

    def OmegaRtot(self, x):
        return (self.OmegaR0tot * self.H0**2) / (np.exp(x * 4) * self.H(x) ** 2)

    def chi(self, x):
        return self.eta(0) - self.eta(x)

    def r(self, x):
        chi = self.chi(x)
        if self.OmegaK0 == 0:
            return chi
        sqrt_k = np.sqrt(np.abs(self.OmegaK0)) * self.H0 / const.c
        if self.OmegaK0 < 0:
            return np.sin(sqrt_k * chi) / sqrt_k
        return np.sinh(sqrt_k * chi) / sqrt_k

    def dA(self, x):
        return np.exp(x) * self.r(x)

    def dL(self, x):
        return self.r(x) / np.exp(x)

    def info(self):
        """
        Print some useful info about the class
        """
        print("")
        print(f"Background Cosmology [{self.name}]:")
        print("OmegaB0:        %8.7f" % self.OmegaB0)
        print("OmegaCDM0:      %8.7f" % self.OmegaCDM0)
        print("OmegaLambda0:   %8.7f" % self.OmegaLambda0)
        print("OmegaR0:        %8.7e" % self.OmegaR0)
        print("OmegaNu0:       %8.7e" % self.OmegaNu0)
        print("OmegaK0:        %8.7f" % self.OmegaK0)
        if self.TCMB0 is not None:
            print("TCMB (K):      %8.7f" % (self.TCMB0 / const.K))
        print("h:             %8.7f" % self.h0)
        print("H0:            %8.7e" % self.H0)
        print("H0 (km/s/Mpc): %8.7f" % (self.H0 / (const.km / const.s / const.Mpc)))
        print("Neff:          %8.7f" % self.Neff)
        print("OmegaM0:        %8.7f" % self.OmegaM0)
        print("OmegaR0tot:     %8.7e" % self.OmegaR0tot)

    def solve(self, rtol=1e-6):
        """
        Main driver for all the solving.
        For LCDM we only need to solve for the conformal time eta(x)
        """
        # Compute and spline conformal time eta = Int_0^t dt/a = Int da/(a^2 H(a)) =  Int dx/[ exp(x) * H(exp(x)) ] where x = log a
        eta_out = integrate.solve_ivp(
            self.detadx,
            t_span=(self.x_start, self.x_end),
            y0=(const.c / self.Hp(self.x_start),),
            t_eval=self.x_pts,  # explicitly compute eta at these x values
            dense_output=True,  # create spline
            rtol=rtol,
        )

        # same for t
        t_out = integrate.solve_ivp(
            self.dtdx,
            t_span=(self.x_start, self.x_end),
            y0=(1 / (2 * self.H(self.x_start)),),
            t_eval=self.x_pts,  # explicitly compute eta at these x values
            dense_output=True,  # create spline
            rtol=rtol,
        )

        self.eta_ode_sol = eta_out.sol
        self.t_ode_sol = t_out.sol

    def plot(self, url):
        """
        Plot some useful quantities
        """
        npts = 2000
        x = np.linspace(self.x_start, self.x_end, num=npts)
        eta = self.eta(x)
        # eta = self.eta(x) * self.H(x) * np.exp(x) / const.c

        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        ax.plot(x, eta)
        ax.set_title(r"$\eta(x)$")
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\eta(x)$ (Mpc.)")

        fig.savefig(path.join(url, "eta"))
        plt.close(fig)
