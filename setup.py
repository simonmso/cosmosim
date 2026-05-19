from setuptools import setup, Extension
from Cython.Build import cythonize
import CyRK
import numpy as np
import os

# Grab include directories so the C++ compiler can locate header files
cyrk_include = os.path.dirname(CyRK.__file__)
numpy_include = np.get_include()

setup(
    ext_modules=cythonize(
        [
            # "./Perturbations.py",
            "./RecombinationHistory.py",
            "./BackgroundCosmology.py",
            "./FastSpline.py",
            Extension(
                "Perturbations",  # Output module name
                sources=["Perturbations.py"],  # Source file
                language="c++",  # Enforce C++ compilation
                include_dirs=[cyrk_include, numpy_include],
                extra_compile_args=["-std=c++17"],
            ),
        ],
        annotate=True,
    ),
    compiler_directives={"language_level": "3"},
)
