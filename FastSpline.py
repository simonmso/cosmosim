import cython


@cython.cclass
class FastSpline:
    def __init__(self, poly_spline=None):
        if poly_spline is not None:
            self.x = poly_spline.x
            self.nx = self.x.shape[0]
            self.c = poly_spline.c
            self.nk = self.c.shape[0]

    @cython.cfunc
    def evaluate(self, x_inp: cython.double) -> cython.double:

        # Bounds check
        if (x_inp > self.x[self.nx - 1]) or (x_inp < self.x[0]):
            raise ValueError(
                f"Input {x_inp} not within [{self.x[0]}, {self.x[self.nx - 1]}]"
            )

        # Find interval
        i: cython.Py_ssize_t = 0
        while self.x[i + 1] < x_inp:
            i += 1

        # Evaluate polynomial
        arg: cython.float = x_inp - self.x[i]
        k: cython.Py_ssize_t
        f: cython.double = 0.0
        for k in range(self.nk):
            f = f * arg + self.c[k, i]
        return f
