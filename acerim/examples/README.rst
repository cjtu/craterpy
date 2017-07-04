===============
Worked Examples
===============

This README will provide step by step examples of how to use acerim to analyze crater ejecta and plot the results. Acerim provides a suite of functions to:

1) Import crater data
2) Import image data
3) Extract regions of interest (ROIs) from the image data around selected craters from crater data
4) Conduct basic statistics on the ROIs and tabulate the results
5) Plot crater statistics and ROIs
6) Export tabulated statistics, plots, and ROI images


1) Importing Crater Data
========================

Acerim uses pandas, a spreadsheet handling library, to read and store crater data. Pandas supports a variety of spreadsheet types, including plain text tab-delimited (.txt), comma separated values (.csv), and excel (.xls) spreadsheets. The pandas dataset is centered on the pandas.DataFrame object which conveniently stores tabular data in a matrix-like, dictionary-like object which allows the use of column names as keys, but can also be indexed similarly to a 2D array. Acerim extends the basic DataFrame with the acerim.CraterDataFrame object. This object includes all of the functionality of the pandas DataFrame with a few extra features implemented for convenience.

A CraterDataFrame can be initialized as follows:

>>> import acerim as ace
>>> crater_csv = '/acerim/tests/crater_data.csv'
>>> cdf = ace.CraterDataFrame(crater_csv)

The first 5 lines of data can be printed with:

>>> print(cdf[0:5])
       Name    Rad    Lat   Lon
0  Patricia  5.040  24.91  0.50
1   Ukert R  9.130   7.93  0.69
2   Chladni  6.535   3.99  1.12
3   Ukert N  8.565   7.58  2.01
4      Bela  5.035  24.67  2.27

The CraterDataFrame requires that latitude, longitude, and radius (or diameter) columns are defined in the header row of the imported data. To manually set the column names or if a header is not included in the data, use the columns argument as follows:

>>> cdf = ace.CraterDataFrame(crater_csv, columns=['Name', 'Diam', 'Lat', 'Lon'])
>>> print(cdf[0:5])
       Name    Rad    Lat   Lon   _Rad
0  Patricia  5.040  24.91  0.50  2.520
1   Ukert R  9.130   7.93  0.69  4.565
2   Chladni  6.535   3.99  1.12  3.268
3   Ukert N  8.565   7.58  2.01  4.283
4      Bela  5.035  24.67  2.27  2.518

This will assign the given column names to the columns in the order that they appear in the sheet.


2) Importing Image Data
=======================


3) Extracting ROIs
==================


4) ROI Statistics
=================


5) Plotting Statistics and ROIs
===============================


6) Exporting Statistics and Saving Plots
========================================