import cython
from FastSpline cimport FastSpline
from BackgroundCosmology cimport BackgroundCosmology


cdef class RecombinationHistory:
    # Settings for solver
    cdef double x_start
    cdef double x_end
    cdef Py_ssize_t npts
    cdef Py_ssize_t npts_tau_before_reion
    cdef Py_ssize_t npts_tau_during_reion
    cdef Py_ssize_t npts_tau_after_reion
    cdef double Xe_saha_limit

    cdef BackgroundCosmology cosmo

    cdef double Yp

    cdef double reionization
    cdef double z_reion
    cdef double delta_z_reion
    cdef double helium_reionization
    cdef double z_helium_reion
    cdef double delta_z_helium_reion
    cdef double z_star

    # Splines
    cdef object tau_spline
    cdef object dtau_spline
    cdef FastSpline dtau_spline_fast
    cdef object d2tau_spline
    cdef FastSpline d2tau_spline_fast
    cdef object d3tau_spline
    cdef object log_Xe_spline
    cdef object log_ne_spline
    cdef object s_spline

    cdef double dtau_fast(self, double x)
    cdef double d2tau_fast(self, double x)



