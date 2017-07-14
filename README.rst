======
ACERIM
======

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python 
package for analyzing impact crater ejecta.

This package provides a variety of tools for:

- loading crater data
- loading image data
- locating craters in an image
- extracting a Region of Interest (ROI) around a crater from an image
- analyzing ROIs using the included or user-defined statistical methods

For a worked example of how to use acerim in a research workflow, see:
    /acerim/examples/example.py

Note: This package was written with the Moon in mind, but is applicable to any 
cratered planetary body, as long as the image data is in a simple cylindrical
projection. For assistance reading and reprojecting images in other projections
see GDAL_.

.. _GDAL: http://www.gdal.org/


Dependencies
------------

This project uses the Anaconda platform developed by continuum.io. The easiest
way to install this package is to install anaconda [here](anacondalink)
and use the command "conda install acerim". This will automatically fetch and 
build the necessary dependencies (NOTE: package not yet hosted on PIP so this
won't work yet).

For non-anaconda installations, ensure the folllowing depencdencies are 
installed and up to date:

- numpy
- scipy
- gdal
- pandas
- matplotlib
    
Other useful (but non-essential) packages:

- duecredit
- pytest


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

The core of this project is located in /acerim. A worked example of a research
workflow is given in /acerim/example.py. Some sample data and test cases are 
included in /acerim/tests


Testing acerim
--------------

A suite of unittests and sample data are located in the /acerim/tests 
directory. Unittesting can be used to ensure that acerim is properly installed
and working correctly on your system. It is also useful to ensure that any 
changes to the source code does not break the back-end code. The easiest way to
run all unittests automatically is by using pytest in the shell:

1) open a terminal/shell/cmd window
2) navigate to the parent acerim directory (e.g.'/Users/cjtu/code/acerim')
3) run the command
    | py.test acerim

A summary of test results will appear in the shell.


Citing acerim
-------------

For convenience, this project uses an MIT open liscence and duecredit for ease
of use and citation. Make sure duecredit is installed and then simply run your 
code with the duecredit flag::
|	python -m duecredit your_acerim_branch/your_analysis.py

All modules and functions invoked by you_analysis.py will be stored in the 
duecredit.p log file. To output a Latex summary of this logfile, type the 
following command::
|	duecredit summary --format=bibtex
    
Alternatively, the associated thesis_ for this project can be cited at DOI_.

.. _thesis: https://thesislink.com
.. _DOI: https://doi.com