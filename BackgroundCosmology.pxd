from FastSpline cimport FastSpline

cdef class BackgroundCosmology:
    # Public
    cdef public double OmegaK0
    cdef public double h0
    cdef public double H0
    cdef public str name
    cdef public double Neff
    cdef public double rhoc0
    cdef public double TCMB0
    cdef public double OmegaM0
    cdef public double OmegaB0
    cdef public double OmegaCDM0
    cdef public double OmegaNu0
    cdef public double OmegaR0
    cdef public double OmegaR0tot
    cdef public double OmegaLambda0
    cdef public double x_rm
    cdef public double x_mlam
    cdef public double x_accel

    # Private
    cdef object x_pts

    cdef double x_start
    cdef double x_end
    
    # Scipy splines
    cdef object t_ode_sol 
    cdef object eta_sol

    # Custom spline
    cdef FastSpline eta_fast_sol

    cdef double eta_fast(self, double x)
    cdef double H_fast(self, double x)
    cdef double Hp_fast(self, double x)
    cdef double dHdx_fast(self, double x)
    cdef double dHpdx_fast(self, double x)
    
