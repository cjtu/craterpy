## acerim

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) is a python 
package for analyzing impact crater ejecta.

It provide a variety of tools for:
- loading crater data into CraterDataFrames 
- loading image data into CraterDatasets
- locating craters in a CraterDataset given their locations in a CraterDataFrame
- reading image data in a Region of Interest (ROI) around a crater
- analyzing ROIs using included or user-defined statistical methods

This package was written with the Moon in mind, but is applicable to any 
cratered planetary body, as long as the image data is in a simple cylindrical
projection. 


### Dependencies

This project uses the Anaconda platform developed by continuum.io. The easiest
way to install this package is to install anaconda and use "conda install acerim".
This will automatically fetch and build the necessary dependencies.

For non-anaconda installations, ensure the folllowing depencdencies are 
installed and updated:
	numpy
	scipy
	gdal
	pandas
   matplotlib


### Organization

The project has the following structure:

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

The core of this project is located in /acerim and a 
### Testing

A suite of unittests and sample data are located in the /tests directory. These
can be used to ensure acerim is working correctly on your system, or that any 
changes you have made did not interfere with the back-end code. To automatically
run all tests from the shell:

1) navigate to the parent acerim directory (e.g.'/Users/cjtu/code/acerim')
2) run the command:	py.test acerim

A summary of test results will appear in the shell


### Citing this project

For convenience, this project uses a MIT open liscence and duecredit for ease
of use and citation. Simply run your code with the duecredit flag:
	python -m duecredit your_acerim_branch/your_analysis.py

All invoked modules and functions will be stored by the duecredit.p file.
To output a Latex summary of the used files and functions, use the following 
command:
	duecredit summary --format=bibtex
    
Alternatively, the associated thesis for this project can be viewed and cited 
<here> and its DOI is <DOI>.