from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

libs = ['m', 'fftw3']
args = ['-std=c99', '-O3']
sources = ['src/noisegate.pyx', 'src/fft_stuff.c']
include = ['include', numpy.get_include()]
linkerargs = ['-Wl,-rpath,lib']
libdirs = ['lib']


extensions = [
    Extension("noisegate",
              sources=sources,
              include_dirs=include,
              libraries=libs,
              library_dirs=libdirs,
              extra_compile_args=args,
              extra_link_args=linkerargs)
]

setup(name='noisegate',
      packages=['noisegate'],
      ext_modules=cythonize(extensions, annotate=True),
      )
