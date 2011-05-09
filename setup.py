from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
sourcefiles = ['sample.pyx', 'fast_sample.c']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("sample", sourcefiles)]
)

