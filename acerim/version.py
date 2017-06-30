from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev' # make it '' for full release

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 2 - Pre-Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "acerim: a package for analyzing impact crater ejecta"
# Long description will go up on the pypi page
long_description = """

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
"""

NAME = "acerim"
MAINTAINER = "Christian Tai Udovicic"
MAINTAINER_EMAIL = "cj.taiudovicic@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/cjtu/acerim"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Christian Tai Udovicic"
AUTHOR_EMAIL = "cj.taiudovicic@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'acerim': [pjoin('data', '*')]}
REQUIRES = ["numpy", "scipy", "matplotlib", "gdal", "pandas"]
