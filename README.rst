ACERIM
======

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python 
package that simplifies impact crater ejecta analysis workflows. It provides functions for loading crater data into easily queried pandas DataFrames and uses crater locations to automatically pull data from gdal Datasets. 

The major features of acerim are:

- loading, storing, and querying crater data
- loading image data
- locating craters in an image
- extracting a Region of Interest (ROI) around a crater from an image
- analyzing ROIs using the included or user-defined statistical methods

See ./examples/example.py for a worked example of how to use acerim in a research workflow.

Note: This package was written with the Moon in mind, but is applicable to any 
cratered planetary body. Currently acerim only supports image data in simple cylindrical projection. For assistance reading and reprojecting images in python, see GDAL_.

.. _GDAL: http://www.gdal.org/


Dependencies
------------

- gdal
- numpy
- scipy
- pandas


Installation
------------

PLEASE NOTE: Acerim depends on the GDAL (Geospatial Data Abstraction Library) python package. Since GDAL in turn depends on C++ classes, it is highly recommended that you follow the instructions here_ to properly install gdal on your python environment before you install acerim.

.. _here: https://pypi.python.org/pypi/GDAL


Using Anaconda
^^^^^^^^^^^^^^

The recommended way to install acerim is using the anaconda_ platform which can be downloaded here_.  Anaconda is useful because it will automatically solve package dependency conflicts and allows you to create virtual environments to separate packages with finicky dependencies from your main python installation. To create and activate a conda virtual environment, see `Managing Environments <https://conda.io/docs/using/envs>`_. 

.. _Anaconda: https://www.continuum.io/Anaconda-Overview

.. _here: https://www.continuum.io/downloads

With anaconda and gdal installed, open a terminal/command line window and create a new conda environment. Listing gdal and the other dependencies outright forces conda to resolve dependency conflicts outright. Note that python must be <= 3.3 for gdal to function correctly::

  conda create --name env_name python=3.3 gdal numpy scipy pandas matplotlib

You can activate the environment with::

  source activate env_name

And then install acerim from PyPI using pip::

  pip install acerim


Manual installation
^^^^^^^^^^^^^^^^^^^
First ensure that GDAL is installed. Running python <= 3.3, install the following dependencies::

- numpy
- scipy
- pandas
- matplotlib

Download the most recent distribution from `PyPI <https://pypi.python.org/simple/acerim>`_ and in the main project directory run

  python setup.py install


Organization
------------

The project has the following structure::

    acerim/
      |- README.md
      |- acerim/
         |- aceclasses.py
         |- acefunctions.py
         |- acestats.py
         |- examples
            |- README.md
            |- example.py
         |- tests
            |- craters.csv
            |- moon.tif
            |- test_classes.py
            |- test_functions.py
         |- version.py
      |- setup.py
      |- LICENSE

The core of this project is located in ./acerim. To get started using acerim, see example.py in ./acerim/examples. Documentation is listed in ./docs and is also available at `readthedocs <https://readthedocs.org/projects/acerim/>`_. A suite of tests is located in ./acerim/tests.


Testing acerim
--------------

A suite of unittests are located in the ./acerim/tests. They use the sample data included in ./acerim/examples to test all acerim classes and functions. To test if acerim is working as it should on your machine, install the pytest module (using conda or pip) and follow the following steps:

1) open a terminal/shell/cmd window
2) navigate to the parent acerim directory (e.g.'/Users/cjtu/code/acerim')
3) run the command
    | py.test acerim

A summary of test results will appear in the shell. 


Support and Bug Reporting
-------------------------

Any bugs or errors can be reported to Christian at cj.taiudovicic@gmail.com. Please include your operating system and printout of your python environment (e.g. using conda list).


Citing acerim
-------------

For convenience, this project uses the OSI-certified MIT open liscence for ease of use and distribution. The author simply asks that you cite the associated thesis_ for this project which can be found at DOI_. 

.. _thesis: https://thesislink.com
.. _DOI: https://doi.com


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