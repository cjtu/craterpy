## acerim

Acerim is a package for identifying and analyzing impact crater ejecta.

The Automated Crater Ejecta Region of Interest Mapper (ACERIM) package provides
a variety of tools for loading crater locations and image data in order to 
characterize ejecta in a regions of interest around known crater locations. 

The package was written generally to be applicable to cratered, roughly 
spherical bodies. Input data must be in simple cylindrical coordinates in a 
format recognizable by the gdal image processing library (see GDAL for full list
of supported data types). Crater lists are handled by custom DataFrame objects 
built on the pandas library. Most standard spreadsheet file formats (e.g. csv, 
Excel) are supported. 


### Dependencies

This project uses the continuum.io platform Anaconda. If anaconda is installed,
this package will automatically fetch and build the necessary dependencies.
For manual install, the dependent libraries are listed below:
	numpy
	scipy
	gdal
	pandas


### Organization



### Testing

A suite of unittests and sample data are located in the /tests directory. To
run all tests from the command line, navigate to the parent acerim directory
(e.g. '/Users/cjtu/code/acerim') and run the command:
	py.test acerim


### Citing this project

For convenience, this project uses a MIT open liscence and duecredit for ease
of use and citation. Simply run your code with the duecredit flag:
	python -m duecredit your_acerim_branch/your_analysis.py

All invoked modules and functions will be stored by the duecredit.p file.
To get a summary of the used files and functions, use the following command:
	duecredit summary --format=bibtex