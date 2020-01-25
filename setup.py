from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

import glob
import subprocess
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

sys_includes = []

if __name__ == "__main__":
    setup(name="badlands_companion",
          author            = "Tristan Salles",
          author_email      = "tristan.salles@sydney.edu.au",
          url               = "https://github.com/badlands-model",
          version           = "1.0.2",
          description       = "Pre and post processing scripts for Badlands",
          long_description = long_description,
          long_description_content_type = "text/markdown",
          ext_modules       = [],
          packages          = ['badlands_companion'],
          package_data      = {'badlands_companion':sys_includes},
          data_files=[('badlands_companion',sys_includes)],
          include_package_data = True,
          install_requires  = [
                        'tribad',
                        'numpy>=1.15.0',
                        'six>=1.11.0',
                        'setuptools>=38.4.0',
                        'gFlex>=1.1.0',
                        'scikit-image>=0.15',
                        'pandas>=0.24',
                        'scipy>=1.2',
                        'h5py>=2.8.0',
                        'matplotlib>=3.0',
                        'plotly',
                        'cmocean',
                        'pyevtk',
                        'colorlover'
                        ],
          python_requires   = '>=3.5',
          classifiers       = ['Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6']
          )
