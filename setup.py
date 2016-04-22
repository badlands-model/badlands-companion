#!/usr/bin/env python

from setuptools import setup

"""
setup.py for pyBadlands Companion
"""

setup(
    name="pybadlands_companion",
    version="1.0",
    author="Tristan Salles",
    author_email="tristan.salles@sydney.edu.au",
    description=("Pre and post processing scripts for pyBadlands"),
    url='https://github.com/badlands-model/pyBadlands-Companion',
    long_description=open('README.md').read(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 2.7',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    packages=['pybadlands_companion'],
    install_requires=[
        'pyBadlands'
    ]
)
