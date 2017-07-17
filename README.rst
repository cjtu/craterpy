ACERIM
======

Overview
--------

Welcome to ACERIM!

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python package for planetary scientists that simplifies crater data analysis. If you have image data and a list of crater locations, ACERIM will help you extract data from those locations and analyze that data with statistics of your choosing.

*Use ACERIM if you want to do one or more of the following*::

  - import crater databases and image datasets into easily queried python objects;
  - extract image data from regions around craters given their latitude, longitude, 
      radius, and the desired ROI window size;
  - mask your data arays to isolate pixels within the crater, on the ejecta blanket, 
      or within a user-provided shapefile;
  - compute statistics on the extracted crater/ejecta data;
  - save and plot your ROIs and statistics.

New users can head to acerim/sample/tutorial.py for a step-by-step walkthrough (with sample data) of how to use ACERIM in a research workflow.

Note: This package was written with the Moon in mind, but is applicable to any cratered planetary body. However, ACERIM currently only supports image data in the simple cylindrical projection. For assistance reprojecting images in python, see GDAL_.

.. _GDAL: http://www.gdal.org/


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

**PLEASE NOTE**: Acerim depends on the GDAL (Geospatial Data Abstraction Library) python package. Since GDAL in turn depends on C++, it is highly recommended that you follow the installation instructions on PyPI_ to properly install gdal and its dependencies before you install ACERIM.

.. _PyPI: https://pypi.python.org/pypi/GDAL


Using Anaconda
^^^^^^^^^^^^^^

The recommended way to install ACERIM is using the `anaconda <https://www.continuum.io/Anaconda-Overview>`_ platform which can be downloaded from `Continuum Analytics <https://www.continuum.io/downloads>`_.  Anaconda is preferred because it

1) automatically solves package dependency conflicts, and 
2) conveniently manages your virtual environments, allowing you to separate packages with finicky dependencies from your main python installation. 

The following section will describe how to create and activate a virtual environment which is compatible with ACERIM. For more on Anaconda virtual environments, see `Managing Environments <https://conda.io/docs/using/envs>`_. 

With anaconda and gdal installed, open a terminal/command line window and create a new conda environment using:: 

  conda create --name env_name python=3.3 gdal numpy scipy pandas matplotlib

Replace env_name with your desired environment name. Listing gdal and the other dependencies outright forces conda to resolve dependency conflicts while building the environment and can avoid dependency headaches in the future. Note that python must be version 2.7 to 3.3 for gdal to function. You can activate the environment with::

  source activate env_name

And then install ACERIM from the Python Package Index (PyPI) using the pip (python install package) command::

  pip install acerim

If the installer completes without any errors, a simple import statement can verify that ACERIM was successfully installed::

  python -c "import acerim"

Now that you have ACERIM installed, head over to /acerim/sample/tutorial.py to get started with your crater analytics!


Manual installation
^^^^^^^^^^^^^^^^^^^
First ensure that GDAL is installed, then install the following dependencies::

- numpy
- scipy
- pandas
- matplotlib
- setuptools (to build the package)

Download the most recent distribution from `PyPI <https://pypi.python.org/simple/acerim>`_ and unzip it in the desired directory. From the root project directory run::

  python setup.py install


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

A suite of unittests are located in the ./acerim/tests. They use the sample data included in /acerim/sample to test all ACERIM classes and functions. To test if ACERIM is working as it should on your machine, install the pytest module (using conda or pip) and follow the following steps::

  1) open a terminal/shell/cmd window
  2) navigate to the parent ACERIM directory (e.g.'/Users/cjtu/Desktop/acerim')
  3) run the following command:

::

    py.test acerim

A summary of test results will appear in the shell. 


Support and Bug Reporting
-------------------------

Any bugs or errors can be reported to Christian at cj.taiudovicic@gmail.com. Please include your operating system and details of your python environment (e.g. using conda list).


Citing ACERIM
-------------

For convenience, this project uses the OSI-certified MIT open access liscence for ease of use and distribution. The author simply asks that you cite the project, which can be found at: 

.. image:: https://zenodo.org/badge/88457986.svg
   :target: https://zenodo.org/badge/latestdoi/88457986


LICENSE
-------

The MIT License (MIT)

Copyright (c) 2017 Christian Tai Udovicic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.