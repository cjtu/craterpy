ACERIM |ZenodoBadge|_ |TravisBadge|_ |AppveyorBadge|_ |RtdBadge|_ |PyPiBadge|_ |CodecovBadge|_
==============================================================================================
.. |ZenodoBadge| image:: https://zenodo.org/badge/88457986.svg
.. _ZenodoBadge: https://zenodo.org/badge/latestdoi/88457986

.. |TravisBadge| image:: https://travis-ci.org/cjtu/acerim.svg?branch=master
.. _TravisBadge: https://travis-ci.org/cjtu/acerim

.. |AppveyorBadge| image:: https://ci.appveyor.com/api/projects/status/7r7f4lbj6kgguhtw/branch/master?svg=true
.. _AppveyorBadge: https://ci.appveyor.com/project/cjtu/acerim/branch/master

.. |RtdBadge| image:: http://readthedocs.org/projects/acerim/badge/?version=latest
.. _RtdBadge: http://acerim.readthedocs.io/en/latest/?badge=latest

.. |PyPiBadge| image:: https://badge.fury.io/py/acerim.svg
.. _PyPiBadge: https://badge.fury.io/py/acerim

.. |CodecovBadge| image:: https://codecov.io/gh/cjtu/acerim/branch/master/graph/badge.svg
.. _CodecovBadge: https://codecov.io/gh/cjtu/acerim

Overview
--------

Welcome to ACERIM!

Please note: this package is actively under development. You can direct any questions or report any bugs to Christian at cj.taiudovicic@gmail.com. 

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python package for planetary scientists that simplifies crater data analysis. If you have image data and a list of crater locations, ACERIM will help you extract data from those locations and analyze that data with statistics of your choosing.

Use ACERIM if you want to do one or more of the following:

  - import crater databases and image datasets into easily queried python DataFrames
  - extract image data from regions around craters or other features given lat, lon and radius
  - mask your data arays (e.g. to remove pixels within a crater floor or in a ring on the ejecta blanket)
  - compute statistics on the extracted crater and/or ejecta image data
  - save and plot images and statistics.

New users can head to the `Tutorial Jupyter notebook <https://github.com/cjtu/acerim/blob/master/acerim/sample/Tutorial.ipynb>`_ for a step-by-step walkthrough (with sample data) of how to use ACERIM in a research workflow.

Note: This package was written with the Moon in mind, but is applicable to any cratered planetary body. However, ACERIM currently only supports image data in the simple cylindrical projection. For assistance reprojecting images to simple cylindrical format in python, see `GDAL <http://www.gdal.org/>`_.


Dependencies
------------

ACERIM is compatible with python versions 2.7, 3.4 and 3.5. It requires the following packages:

  - numpy
  - scipy
  - pandas
  - matplotlib
  - gdal=2.1.0


Installation
------------

**Warning**: Acerim depends on the GDAL (Geospatial Data Abstraction Library) python package which requires certain C++ binaries. It is recommended that you follow the `gdal installation instructions <https://pypi.python.org/pypi/GDAL>`_ before installing ACERIM.


Quick Installation with Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to install ACERIM is with `Anaconda <https://www.continuum.io/Anaconda-Overview>`_. See `Continuum Analytics <https://www.continuum.io/downloads>`_ for installation instructions.  Anaconda is useful because it:

1) Provides easy to use virtual environments, and
2) Resolves package dependencies before install.

The following section will describe how to create and activate a conda virtual environment to run ACERIM. For more on Anaconda virtual environments, see `Managing Environments <https://conda.io/docs/using/envs>`_. 

With *anaconda and gdal installed*, open a terminal window and create a new conda environment with the following command (replace **your_env_name** and choose desired python version):: 

  conda create --name your_env_name python=3.5

Activate the environment (OS X, Unix, or powershell users may need the *source*)::

  (source) activate your_env_name

Now install the dependencies. Ensure to specify the gdal and libgdal versions to avoid a known bug being tracked `here <https://github.com/ContinuumIO/anaconda-issues/issues/1687>`_::

  conda install numpy scipy pandas matplotlib gdal=2.1.0 libgdal=2.1.0

With the environment active, install the latest stable release of ACERIM from the Cheese Shop (Python Package Index) using pip (python install package)::

  pip install acerim

Now that you have ACERIM installed, head over to the `Tutorial <https://github.com/cjtu/acerim/blob/master/acerim/sample/Tutorial.ipynb>`_ to get started!

**Note**: Remember to activate your virtual environment each time you use ACERIM.


Cloning this repository
^^^^^^^^^^^^^^^^^^^^^^^

You can clone this repository by navigating to your target directory and issuing the command::

  git clone https://github.com/cjtu/acerim.git

If you then switch to the *acerim* root directory and activate your environment, you can install the package from its source using::

  python setup.py install



Organization
------------

The project has the following structure::

    acerim/
      |- acerim/
         |- aceclasses.py
         |- acefunctions.py
         |- acestats.py
         |- sample
            |- craters.csv
            |- moon.tif
            |- Tutorial
         |- tests
            |- test_classes.py
            |- test_functions.py
         |- version.py
      |- docs/
      |- LICENSE.txt
      |- README.rst
      |- setup.py
      |- setup.cfg

The main modules are located in **acerim/acerim/**. To get started, see the examples given in `Tutorial <https://github.com/cjtu/acerim/blob/master/acerim/sample/Tutorial.ipynb>`_. API documentation is available at `readthedocs <https://readthedocs.org/projects/acerim/>`_.


Testing ACERIM
--------------

A suite of unittests are located in the **/acerim/tests**. They use the sample data included in **/acerim/sample**. To troubleshoot possible errors you can install the pytest module and run it.::

  conda install pytest

Then from the root acerim directory::

    py.test

A summary of test results will appear in the shell. 


Support and Bug Reporting
-------------------------

Any bugs or errata can be reported to Christian at cj.taiudovicic@gmail.com. Please include your operating system and details of your python environment (e.g. using conda list).


Citing ACERIM
-------------

For convenience, this project uses the OSI-certified MIT open access liscence for warranty-free ease of use and distribution. The author simply asks that you cite the project. The citable DOI can be found at Zenodo by clicking the button below. To read more about citable code, check out `Zenodo <http://help.zenodo.org/features>`_.

.. image:: https://zenodo.org/badge/88457986.svg
    :target: https://zenodo.org/badge/latestdoi/88457986


License
-------

Copyright (c) 2017- Christian Tai Udovicic. Released under the MIT license. This software comes with no warranties. See LICENSE.txt for details.