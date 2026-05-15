cdef class FastSpline:
    cdef double[:] x
    cdef Py_ssize_t nx
    cdef double[:, :] c
    cdef Py_ssize_t nk

    cdef double evaluate(self, double x)