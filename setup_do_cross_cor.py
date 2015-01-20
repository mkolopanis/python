from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
	ext_modules = cythonize("do_cross_cor.pyx"),
	include_dirs=[numpy.get_include()]
)
