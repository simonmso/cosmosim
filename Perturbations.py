from os import path
import numpy as np
import cython

from scipy import integrate
from scipy import interpolate
from scipy import optimize
from matplotlib import pyplot as plt

from Global import const
from Global import APS_COL_W as apsw
from cython.cimports.FastSpline import FastSpline
from cython.cimports.BackgroundCosmology import BackgroundCosmology
from cython.cimports.RecombinationHistory import RecombinationHistory
from cython.cimports.libc.math import exp
from cython.cimports.cpython.ref import PyObject
from cython.cimports.libc.string import memcpy
from cython.cimports.libcpp.utility import move
from cython.cimports.libcpp.vector import vector
from cython.cimports.CyRK import (
    cysolve_ivp_gil,
    DiffeqFuncType,
    WrapCySolverResult,
    CySolveOutput,
    PreEvalFunc,
    ODEMethod,
    Event,
)

# A system to have control over where each
# quantity is in the ODE arrays (optional to use)


i_deltaCDM = cython.declare(cython.Py_ssize_t, 0)
i_vCDM = cython.declare(cython.Py_ssize_t, 1)
i_deltaB = cython.declare(cython.Py_ssize_t, 2)
i_vB = cython.declare(cython.Py_ssize_t, 3)
i_Phi = cython.declare(cython.Py_ssize_t, 4)
i_theta = cython.declare(cython.Py_ssize_t, 5)

c = cython.declare(cython.double, const.c)


@cython.cclass
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
    npts_x = 2500
    # npts_tight = 13
    npts_tight = int(0.4 * npts_x)

    # Tight coupling hard max (start of recomb.)
    # _tight_hard_stop = -8.3
    _tight_hard_stop: cython.double

    cosmo: BackgroundCosmology
    rec: RecombinationHistory
    k_min = cython.declare(cython.double, visibility="public")
    k_max = cython.declare(cython.double, visibility="public")
    npts_k = cython.declare(cython.int, visibility="public")
    n_ell_theta: cython.Py_ssize_t
    n_tot_tight: cython.Py_ssize_t
    n_tot_full: cython.Py_ssize_t

    R0: cython.double
    H0: cython.double

    results = cython.declare(object, visibility="public")
    results_x = cython.declare(object, visibility="public")
    splines: object
    psi_spline: object
    source_spline: object

    source_splines = cython.declare(object, visibility="public")

    x_start: cython.double
    x_end: cython.double

    sol_tight = cython.declare(WrapCySolverResult, visibility="public")

    def __init__(
        self,
        BackgroundCosmology,
        RecombinationHistory,
        keta_max=3000.0,
        npts_k=100,
        n_ell_theta=10,
        x_start=-15,
        x_end=0,
        transition=-8.3,
    ):
        """
        Intitialize the object
        """
        self.cosmo = BackgroundCosmology
        self.rec = RecombinationHistory
        self.k_min = 1.0 / self.cosmo.eta(0.0)
        self.k_max = keta_max / self.cosmo.eta(0.0)
        self.npts_k = npts_k
        self.x_start = x_start
        self.x_end = x_end
        self._tight_hard_stop = transition

        # Number of ells to include and the total number of quantities in the ODE system
        self.n_ell_theta = n_ell_theta
        self.n_tot_tight = 5 + 2
        self.n_tot_full = 5 + n_ell_theta

        self.R0 = (4 * self.cosmo.OmegaR0) / (3 * self.cosmo.OmegaB0)

        # Compute x where tau' > 10
        # self._set_tight_limit_tau_only()

        # Unpack some values for speed
        self.H0 = self.cosmo.H0

        # PhD: you need to add polarization and neutrinos here and in the ODEs below
        # (and don't forget the differences between the tight coupling and full ODE)
        # ...

    # =========================================================================
    # Functions availiable after solving
    # =========================================================================

    # def sourceT(self, k, x):
    #     self.check_for_splines()
    #     return self.sourceT_spline(k, x)

    def deltaCDM(self, kx):
        return self.splines[i_deltaCDM](kx)

    def deltaB(self, kx):
        return self.splines[i_deltaB](kx)

    def vCDM(self, kx):
        return self.splines[i_vCDM](kx)

    def vB(self, kx):
        return self.splines[i_vB](kx)

    def Phi(self, kx):
        return self.splines[i_Phi](kx)

    def Psi(self, kx):
        return self.psi_spline(kx)

    def Theta(self, kx, ell):
        return self.splines[i_theta + ell](kx)

    def source(self, kx):
        return self.source_spline(kx)

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

    def plot(self, url):
        """
        Plot the perturbations and source function as function of x = log(a) for a single value of k
        """
        kval = self.k_min
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
        x_tc_end = self._tight_hard_stop
        x_tight = np.linspace(self.x_start, x_tc_end, self.npts_tight)
        x_full = np.linspace(x_tc_end, self.x_end, self.npts_x - self.npts_tight)
        x = np.concat((x_tight, x_full[1:]))

        # 3D array to store the data
        results = np.zeros((self.n_tot_full, self.npts_k, len(x)))

        # Psi isn't solved differentially,
        # so is stored seperately
        Psi = np.zeros((self.npts_k, len(x)))

        # Loop over all k-values
        for idx, k in enumerate(ks):
            # Compute IC for tight coupling
            y_tc_ic = self.get_ic(self.x_start, k)

            # Solve the tight coupling ODE
            sol_tight = self.solve_ivp(
                "tight", (self.x_start, x_tc_end), y_tc_ic, k, x_tight
            )

            assert (
                sol_tight.success
            ), f"Failed to find tight solution, {sol_tight.message}"

            results[: self.n_tot_tight, idx, : self.npts_tight] = sol_tight.y

            # calculate the non-differential tight values
            self.set_analytic_tight_values(results, x_tight, k, idx)

            y_full_ic = results[:, idx, self.npts_tight - 1]

            # Solve the full ODE
            sol_full = self.solve_ivp(
                "full", (x_tc_end, self.x_end), y_full_ic, k, x_full
            )
            assert sol_full.success, f"Failed to find full solution, {sol_full.message}"

            results[:, idx, self.npts_tight :] = sol_full.y[:, 1:]

            Psi[idx] = self.get_Psi(
                x, results[i_Phi, idx, :], results[i_theta + 2, idx, :], k
            )

        self.results = results
        self.results_x = x

        self.splines = []

        # Spline ode variables
        for result in self.results:
            self.splines.append(
                interpolate.RegularGridInterpolator((ks, x), result, method="cubic")
            )

        # Spline source function
        source = self.calc_source(ks, x, results, Psi)
        self.source_spline = interpolate.RegularGridInterpolator(
            (ks, x), source, method="cubic"
        )

        # Splines experiment
        splines = []
        for xi in range(len(x)):
            s = interpolate.CubicSpline(ks, source[:, xi])
            splines.append(FastSpline(poly_spline=s))
        self.source_splines = splines

        # Spline psi
        self.psi_spline = interpolate.RegularGridInterpolator(
            (ks, x), Psi, method="cubic"
        )

    def get_ic(self, x, k):
        """
        Set IC for the tight coupling system at given x = log(a) and wavenumber k
        """
        y = np.zeros(self.n_tot_tight)

        # Cosmological variables
        Hp = self.cosmo.Hp(x)
        ckHp = const.c * k / Hp

        # Compute IC
        Psi = -(2.0 / 3.0)
        y[i_deltaCDM] = -1.5 * Psi
        y[i_vCDM] = -0.5 * ckHp * Psi
        y[i_deltaB] = y[i_deltaCDM]
        y[i_vB] = y[i_vCDM]
        y[i_Phi] = -Psi
        y[i_theta + 0] = -0.5 * Psi
        y[i_theta + 1] = (1.0 / 6.0) * ckHp * Psi

        return y

    def solve_ivp(
        self,
        func: str,
        t_span: tuple,
        y0: cython.double[:],
        k: cython.double,
        x_eval: cython.double[:],
    ):
        # This is almost directly taken from the example function in the CyRK docs:
        # https://cyrk.readthedocs.io/en/latest/Demos/1_-_Getting_Started.html#cysolve_ivp-Example

        # Cast our diffeq to the accepted format
        dydt: DiffeqFuncType

        if func == "tight":
            dydt = rhs_tight_coupling
        else:
            dydt = rhs_full

        # Convert the python user input to pure C types
        num_y: cython.size_t = len(y0)
        num_x: cython.size_t = len(x_eval)
        t_start: cython.double = t_span[0]
        t_end: cython.double = t_span[1]
        y0_vec: vector[cython.double] = vector[cython.double](num_y)
        t_eval_vec: vector[cython.double] = vector[cython.double](num_x)
        yi: cython.size_t
        for yi in range(num_y):
            y0_vec[yi] = y0[yi]

        xi: cython.size_t
        for xi in range(num_x):
            t_eval_vec[xi] = x_eval[xi]

        args: RHS_args = RHS_args(k, cython.cast(cython.pointer(PyObject), self))

        args_vec: vector[cython.char] = vector[cython.char](cython.sizeof(RHS_args))

        memcpy(args_vec.data(), cython.address(args), cython.sizeof(RHS_args))

        result: CySolveOutput = cysolve_ivp_gil(
            dydt,
            t_start,
            t_end,
            y0_vec,
            method=ODEMethod.DOP853,
            rtol=1.0e-10,
            atol=1.0e-10,
            args_vec=args_vec,
            num_extra=0,
            max_num_steps=1000000,
            max_ram_MB=2000,
            dense_output=False,  # unused
            t_eval_vec=t_eval_vec,
            pre_eval_func=DummyPreEval,  # unused
            events_vec=vector[Event](),  # unused
            rtols_vec=vector[cython.double](),  # unused
            atols_vec=vector[cython.double](),  # unused
            max_step=1,
            first_step=1e-5,
        )

        pysafe_result: WrapCySolverResult = WrapCySolverResult()
        pysafe_result.set_cyresult_pointer(move(result))

        return pysafe_result

    def get_Psi(self, x, phi, theta2, k):
        return -phi - 12 * (self.cosmo.H0 / (const.c * k * np.exp(x))) ** 2 * (
            self.cosmo.OmegaR0 * theta2
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
        y[i_theta + 2, k_idx, : self.npts_tight] = (
            -(20.0 / 45.0)
            * (const.c * k / (self.cosmo.Hp(x) * self.rec.dtau(x)))
            * y[i_theta + 1, k_idx, : self.npts_tight]
        )

        for l in range(3, self.n_ell_theta):
            prev = y[i_theta + l - 1, k_idx, : self.npts_tight]
            y[i_theta + l, k_idx, : self.npts_tight] = (
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

    def calc_source(self, ks, x, results, Psi_full):
        ret = np.zeros(results.shape[1:])

        a = np.exp(x)

        g = self.rec.g_tilde(x)
        dg = self.rec.dg_tilde(x)
        d2g = self.rec.d2g_tilde(x)
        t = self.rec.tau(x)
        dt = self.rec.dtau(x)
        d2t = self.rec.d2tau(x)

        Hp = self.cosmo.Hp(x)
        dHp = self.cosmo.dHpdx(x)
        d2Hp = self.cosmo.d2Hpdx2(x)

        for idx, k in enumerate(ks):
            derivs = self.rhs_py(x, results[:, idx, :], k)

            ck = const.c * k

            O0 = results[i_theta, idx, :]
            O1 = results[i_theta + 1, idx, :]
            O2 = results[i_theta + 2, idx, :]
            O3 = results[i_theta + 3, idx, :]
            Psi = Psi_full[idx, :]
            vB = results[i_vB, idx, :]

            dPhi = derivs[i_Phi, :]
            dO1 = derivs[i_theta + 1, :]
            dO2 = derivs[i_theta + 2, :]
            dO3 = derivs[i_theta + 3, :]
            dvB = derivs[i_vB, :]

            dPsi = (
                -dPhi
                - 12
                * (self.cosmo.H0 / ck) ** 2
                * self.cosmo.OmegaR0
                * (dO2 - 2 * O2)
                / a**2
            )

            d2O2 = ck / (5 * Hp) * (
                -(2 * dHp / Hp * O1) + (2 * dO1) + (3 * dHp / Hp * O3) - (3 * dO3)
            ) + (9 / 10) * (d2t * O2 + dt * dO2)

            rightmost_der = Hp * (dHp * g * O2 + Hp * dg * O2 + Hp * g * dO2) + Hp * (
                (d2Hp * g * O2 + dHp * dg * O2 + dHp * g * dO2)
                + (dHp * dg * O2 + Hp * d2g * O2 + Hp * dg * dO2)
                + (dHp * g * dO2 + Hp * dg * dO2 + Hp * g * d2O2)
            )

            base = g * (O0 + Psi + 0.25 * O2)
            int_sachs_wolfe = np.exp(-t) * (dPsi - dPhi)
            doppler = -1 / ck * ((dHp * g * vB) + (Hp * dg * vB) + (Hp * g * dvB))
            thompson_prefered = (3 / 4) / ck**2 * rightmost_der

            ret[idx, :] = base + int_sachs_wolfe + doppler + thompson_prefered

        return ret

    # Wrapper for the cython rhs eq.
    def rhs_py(self, xs, y: cython.double[:, :], k):
        """
        xs: array-like
        """
        # Arrange y in fortran
        y_view = y.copy_fortran()

        # Store results here
        res = np.zeros_like(
            y_view, order="F"
        )  # order="F" since we want the array to be contiguous in the first index

        res_view: cython.double[::1, :] = res

        # Package up k and self
        args: RHS_args = RHS_args(k, cython.cast(cython.pointer(PyObject), self))
        args_vec: vector[cython.char] = vector[cython.char](cython.sizeof(RHS_args))
        memcpy(args_vec.data(), cython.address(args), cython.sizeof(RHS_args))

        i: cython.size_t
        for i, x in enumerate(xs):
            if x < self._tight_hard_stop:
                rhs_tight_coupling(
                    cython.address(res_view[0, i]),
                    x,
                    cython.address(y_view[0, i]),
                    cython.address(args_vec[0]),
                    DummyPreEval,
                )
            else:
                rhs_full(
                    cython.address(res_view[0, i]),
                    x,
                    cython.address(y_view[0, i]),
                    cython.address(args_vec[0]),
                    DummyPreEval,
                )

        return res


RHS_args = cython.struct(k=cython.double, _self=cython.pointer(PyObject))


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
# @cython.boundscheck(False)
# @cython.cdivision(True)
def rhs_full(
    dydx_view: cython.pointer(cython.double),
    x: cython.double,
    y: cython.pointer(cython.double),
    args: cython.pointer(cython.char),
    pre_eval_func: PreEvalFunc,
    # dy: x: cython.double, y: cython.double[:], k: cython.double
) -> cython.void:
    """
    Set the right hand side of the full ODE system dy/dx = RHS
    for a given value of x. The wavenumber k is set in the global variable k_current
    """
    args_unpacked: cython.pointer(RHS_args) = cython.cast(
        cython.pointer(RHS_args), args
    )
    k: cython.double = args_unpacked.k

    with cython.gil:
        _self: Perturbations = cython.cast(
            Perturbations, cython.cast(object, args_unpacked._self)
        )

        Hp: cython.double = _self.cosmo.Hp_fast(x)
        dt: cython.double = _self.rec.dtau_fast(x)
        eta: cython.double = _self.cosmo.eta_fast(x)

        H0: cython.double = _self.H0
        R0: cython.double = _self.R0

        OmegaR0: cython.double = _self.cosmo.OmegaR0
        OmegaB0: cython.double = _self.cosmo.OmegaB0
        OmegaCDM0: cython.double = _self.cosmo.OmegaCDM0

        n_ell_theta: cython.size_t = _self.n_ell_theta
        lmax_idx: cython.size_t = i_theta + n_ell_theta - 1

    O2: cython.double = y[i_theta + 2]

    ckHp = (c * k) / Hp
    a: cython.double = exp(x)
    R: cython.double = R0 / a
    O0 = y[i_theta + 0]
    O1 = y[i_theta + 1]
    vB = y[i_vB]
    Phi = y[i_Phi]

    Psi: cython.double = -Phi - 12 * (H0 / (c * k * a)) ** 2 * (
        OmegaR0 * O2
    )  # N_2 = 0; i.e. no neutrinos
    dPhi: cython.double = (
        Psi
        - (1 / 3) * ckHp * ckHp * Phi
        + 0.5
        * (H0 / Hp) ** 2
        * (
            OmegaCDM0 * y[i_deltaCDM] / a
            + OmegaB0 * y[i_deltaB] / a
            + 4 * OmegaR0 / (a * a) * O0
        )
    )
    dO0: cython.double = -ckHp * O1 - dPhi

    dydx_view[i_deltaCDM] = ckHp * y[i_vCDM] - 3 * dPhi
    dydx_view[i_vCDM] = -y[i_vCDM] - ckHp * Psi
    dydx_view[i_deltaB] = ckHp * vB - 3 * dPhi
    dydx_view[i_theta + 0] = dO0
    dydx_view[i_Phi] = dPhi
    # ------------------------------------

    dydx_view[i_theta + 1] = ckHp / 3 * (O0 - 2 * O2 + Psi) + dt * (O1 + vB / 3)
    dydx_view[i_vB] = -vB - ckHp * Psi + dt * R * (3 * O1 + vB)

    # Thetas 2 < ell < lmax
    l: cython.size_t
    for l in range(2, n_ell_theta - 1):
        idx: cython.size_t = i_theta + l
        Pi = O2 if l == 2 else 0
        dydx_view[i_theta + l] = ckHp / (2 * l + 1) * (
            l * y[idx - 1] - (l + 1) * y[idx + 1]
        ) + dt * (y[idx] - Pi / 10)

    # Theta lmax
    Olmax = y[lmax_idx]
    dydx_view[lmax_idx] = (
        ckHp * (y[lmax_idx - 1] - (l + 1) / (k * eta) * Olmax) + dt * Olmax
    )


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def rhs_tight_coupling(
    dydx_view: cython.pointer(cython.double),
    x: cython.double,
    y: cython.pointer(cython.double),
    args: cython.pointer(cython.char),
    pre_eval_func: PreEvalFunc,
) -> cython.void:
    """
    Set the right hand side of the tight coupling ODE system dy/dx = RHS
    for a given value of x. The wavenumber k is set in the global variable k_current
    """
    args_unpacked: cython.pointer(RHS_args) = cython.cast(
        cython.pointer(RHS_args), args
    )
    k: cython.double = args_unpacked.k
    with cython.gil:

        _self: Perturbations = cython.cast(
            Perturbations, cython.cast(object, args_unpacked._self)
        )

        Hp: cython.double = _self.cosmo.Hp_fast(x)
        dHp: cython.double = _self.cosmo.dHpdx_fast(x)

        H0: cython.double = _self.cosmo.H0
        dt: cython.double = _self.rec.dtau_fast(x)
        d2t: cython.double = _self.rec.d2tau_fast(x)
        OmegaR0: cython.double = _self.cosmo.OmegaR0
        OmegaCDM0: cython.double = _self.cosmo.OmegaCDM0
        OmegaB0: cython.double = _self.cosmo.OmegaB0
        R0: cython.double = _self.R0

    ckHp = (c * k) / Hp
    a = exp(x)
    R = R0 / a
    O0 = y[i_theta + 0]
    O1 = y[i_theta + 1]
    O2 = -(20.0 / 45.0) * ckHp / dt * O1  # w/o polarization
    vB = y[i_vB]

    Phi = y[i_Phi]

    Psi = -Phi - 12.0 * (H0 / (c * k * a)) ** 2 * (
        OmegaR0 * O2
    )  # N_2 = 0; i.e. no neutrinos

    dPhi = (
        Psi
        - (1.0 / 3.0) * ckHp * ckHp * Phi
        + 0.5
        * (H0 / Hp) ** 2
        * (
            OmegaCDM0 * y[i_deltaCDM] / a
            + OmegaB0 * y[i_deltaB] / a
            + 4 * OmegaR0 / (a * a) * O0
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
    dydx_view[i_deltaCDM] = ckHp * y[i_vCDM] - 3 * dPhi
    dydx_view[i_vCDM] = -y[i_vCDM] - ckHp * Psi
    dydx_view[i_deltaB] = ckHp * vB - 3 * dPhi
    dydx_view[i_vB] = dvB
    dydx_view[i_Phi] = dPhi
    dydx_view[i_theta + 0] = dO0
    dydx_view[i_theta + 1] = 1.0 / 3.0 * (q - dvB)


# Necessary for debuging the RHS eqs.
@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def DummyPreEval(
    a: cython.pointer(cython.char),
    b: cython.double,
    c: cython.pointer(cython.double),
    d: cython.pointer(cython.char),
) -> cython.void:
    pass
