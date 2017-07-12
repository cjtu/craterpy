""" Use setuptools to prepare acerim package.
"""
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

PACKAGES = find_packages()

# Get version from acerim/version.py
version = {}
with open(path.join(here, 'acerim','version.py')) as f:
    exec(f.read(), version)

setup(
    name='acerim',
    version=version['__version__'], # From /acerim/version.py
    description='A package for analyzing impact crater ejecta',
    long_description="""
        Acerim
        =======
        The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python 
        package for analyzing impact crater ejecta.

        This package provides a variety of tools for:
        - loading crater data
        - loading image data
        - locating craters in an image
        - extracting a Region of Interest (ROI) around a crater from an image
        - analyzing ROIs using the included or user-defined statistical methods

        To get started using this software, please go to the repository README_.

        .. _README: https://github.com/cjtu/acerim/master/README.md

        License
        =======
        ``acerim`` is licensed under the terms of the MIT license. See the file
        "LICENSE" for information on the history of this software, terms & conditions
        for usage, and a DISCLAIMER OF ALL WARRANTIES.

        All trademarks referenced herein are property of their respective holders.

        Copyright (c) 2017--, Christian Tai Udovicic.
    """,    
    maintainer='Christian Tai Udovicic',
    maintainer_email='cj.taiudovicic@gmail.com',     
    url='http://github.com/cjtu/acerim',
    license='MIT',
    classifiers=['Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='planetary-science craters ejecta data-analysis',
    author='Christian Tai Udovicic',
    author_email='cj.taiudovicic@gmail.com',
    platforms='OS Independent',     
    packages=PACKAGES,
    package_data={'examples': ['craters.csv', 'moon.tif']},
    install_requires=['numpy', 'scipy', 'matplotlib', 'gdal', 'pandas'],
    extras_require={'testing': ['pytest'],
        'citation': ['duecredit'],
    }
    python_requires='>=2.7'
)