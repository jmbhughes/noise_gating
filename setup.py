from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='noise_gating',
    version='0.0.1',
    packages=[],
    url='',
    license='',
    author='jmbhughes',
    author_email='hughes.jmb@gmail.com',
    description='noise gate data',
    setup_requires=['cython',
                    'numpy'],
    install_requires=['numpy',
                      'astropy',
                      'tqdm'],
    ext_modules=cythonize("noisegate.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
