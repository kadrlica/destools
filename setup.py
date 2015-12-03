import sys
import os
try: from setuptools import setup
except ImportError: from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    return open(os.path.join(here,filename)).read()

setup(
    name='destools',
    version="0.1.0",
    url='https://github.com/kadrlica/destools',
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = ['bin/authlist.py'],
    install_requires=[
        'python >= 2.7.0',
        'numpy >= 1.6.1',
    ],
    packages=['destools'],
    description="Set of simple tools for working in DES.",
    long_description=read('README.md'),
    platforms='any',
    keywords='latex des',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
    ]
)
