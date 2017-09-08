ACERIM
======

Overview
--------

Welcome to ACERIM!

Please note: this package is actively under development. You can direct any questions to Christian at cj.taiudovicic@gmail.com. 

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python package for planetary scientists that simplifies crater data analysis. If you have image data and a list of crater locations, ACERIM will help you extract data from those locations and analyze that data with statistics of your choosing.

Use ACERIM if you want to do one or more of the following:

  - import crater databases and image datasets into easily queried python DataFrames
  - extract image data from regions around craters or other features given lat, lon and radius
  - mask your data arays (e.g. to remove pixels within a crater floor or in a ring on the ejecta blanket)
  - compute statistics on the extracted crater and/or ejecta image data
  - save and plot images and statistics.

New users can head to acerim/sample/tutorial.py for a step-by-step walkthrough (with sample data) of how to use ACERIM in a research workflow.

Note: This package was written with the Moon in mind, but is applicable to any cratered planetary body. However, ACERIM currently only supports image data in the simple cylindrical projection. For assistance reprojecting images to simple cylindrical format in python, see `GDAL <http://www.gdal.org/>`_.


Dependencies
------------

ACERIM is compatible with python versions 2.7-3.3. Additionally, the following packages are required for ACERIM to run::

  - gdal
  - numpy
  - scipy
  - pandas
  - matplotlib


Installation
------------

**PLEASE NOTE**: Acerim depends on the GDAL (Geospatial Data Abstraction Library) python package. Since GDAL in turn depends on C++, it is highly recommended that you follow the installation instructions for gdal on PyPI `here <https://pypi.python.org/pypi/GDAL>`_. Once GDAL is successfully isntalled, proceed with installing ACERIM.


Quick Installation with Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended way to install ACERIM is using the `anaconda <https://www.continuum.io/Anaconda-Overview>`_ platform which can be downloaded from `Continuum Analytics <https://www.continuum.io/downloads>`_.  Anaconda is preferred because it

1) automatically solves package dependency conflicts, and 
2) conveniently manages your virtual environments, allowing you to separate packages with finicky dependencies from your main python installation. 

The following section will describe how to create and activate a conda virtual environment to run ACERIM. For more on Anaconda virtual environments, see `Managing Environments <https://conda.io/docs/using/envs>`_. 

With *anaconda and gdal installed*, open a terminal/command line window and create a new conda environment using:: 

  conda create --name env_name python=3.3 anaconda gdal

Replace env_name with your desired environment name. Listing anaconda will install all anaconda packages into the environment (including the required numpy, scipy, pandas, etc). This aooears to be the simplest way to avoid dependency issues in gdal, but does install unnecessary packages into the environment. For a lightweight install, see `Manual Installation`_. Note that the python version in your environment must be between 2.7 to 3.3 for gdal to function. 

Activate your new environment::

  activate env_name

OS X, Unix or powershell users may need to type **source**::

  source activate env_name

With the environment active, install ACERIM from the Python Package Index (PyPI) using pip (python install package)::

  pip install acerim

If the installer completes without any errors, you can test that ACERIM and its dependencies are successfully install. If the following command runs error-free, then ACERIM was successfully installed!

::

  python -c "import acerim"

Now that you have ACERIM installed, head over to /acerim/sample/tutorial.py to get started with your crater analytics!

Note: Remember to activate your virtual environment before using ACERIM.


Manual Installation with Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install gdal and the other acerim Dependencies_ manually, start by creating a blank environment with conda::

  conda create --name your_env python=3.3

Activate the environment::

  activate your_env

Or::
  
  source activate your_env

Then install GDAL and all of the Dependencies_ using conda.

  conda install gdal numpy scipy pandas matplotlib

Then proceed with installing ACERIM with pip::

  pip install acerim


Manual Installation without Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are familiar with installing packages without Anaconda, you can install ACERIM and its dependencies using pip. Or head to this link `acerim <https://pypi.python.org/simple/acerim>`_ to download the latest distribution from PyPI. You can then unzip it in your desired directory and from the root project directory run:

::

  python setup.py install

But you knew that already... 


Organization
------------

The project has the following structure::

    acerim/
      |- README.rst
      |- acerim/
         |- aceclasses.py
         |- acefunctions.py
         |- acestats.py
         |- sample
            |- craters.csv
            |- moon.tif
            |- tutorial.rst
            |- tutorial.py
         |- tests
            |- test_classes.py
            |- test_functions.py
         |- version.py
      |- docs/
      |- setup.py
      |- setup.cfg
      |- LICENSE.txt

The core of this project is located in /acerim. To get started using ACERIM, see tutorial.py in /acerim/sample. API documentation is listed in /docs and is also available at `readthedocs <https://readthedocs.org/projects/acerim/>`_. A suite of unittests is located in /acerim/tests.


Testing ACERIM
--------------

A suite of unittests are located in the /acerim/tests. They use the sample data included in /acerim/sample to test all ACERIM classes and functions. You can verify that ACERIM is working correctly on your machine by installing the pytest module (using *conda install pytest* or *pip install pytest*) and following these steps::

  1) open a shell/cmd window
  2) navigate to the parent ACERIM directory (e.g.'/Users/cjtu/Desktop/acerim')
  3) run the command:

::

    py.test acerim

A summary of test results will appear in the shell. 


Support and Bug Reporting
-------------------------

Any bugs or errors can be reported to Christian at cj.taiudovicic@gmail.com. Please include your operating system and details of your python environment (e.g. using conda list).


Citing ACERIM
-------------

For convenience, this project uses the OSI-certified MIT open access liscence for ease of use and distribution. The author simply asks that you cite the project if you use it in your work. Citation information can be found and easily exported in your desired format at: 

.. image:: https://zenodo.org/badge/88457986.svg
   :target: https://zenodo.org/badge/latestdoi/88457986


License
-------

Copyright (c) 2017- Christian Tai Udovicic. Released under the MIT license. This software comes with no warranties. See LICENSE.txt for details.