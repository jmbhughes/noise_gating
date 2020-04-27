#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='noisegate',
    python_requires='>=3.7',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    install_requires=["numpy", "deepdish", "astropy"],
    version='0.0.1',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    packages=find_packages(),
    url='',
    description='Noise Gating',
    license='LICENSE.txt',
    long_description=open('readme.md').read(),
    long_description_content_type="text/markdown",
)
