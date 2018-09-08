from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "cpp_mstar",                                
           sources=["cython_od_mstar.pyx"], 
           extra_compile_args=["-std=c++11"]
      )))
