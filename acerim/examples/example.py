# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:20:20 2017

@author: Christian
"""
import sys
import os.path
d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, d)
import numpy.random as rnd

#%% Import acerim modules
from acerim import classes as ac
from acerim import functions as af
from acerim import acestats as acs

# Import dataset from tests
ds_path = os.path.join(d, 'moon.tif')
ads = ac.AceDataset(ds_path, radius=1737)

# Import crater csv from tests
csv_path = os.path.join(d, 'craters.csv')
cdf = ac.CraterDataFrame(csv_path)

# Check crater data
cdf.head()

# Make the crater names in the first column the index 
cdf = ac.CraterDataFrame(csv_path, index_col=0)
cdf.head()

# Find Humboldt. Note the use of square brackets.
cdf.loc['Humboldt']

# Get only the radius of Humboldt
cdf['Rad'].loc['Humboldt']

# Or
cdf.loc['Humboldt']['Rad']

# The difference is that cdf['Rad'] takes the radius series out of the
# dataframe and then searches through it for 'Humboldt' 
len(cdf['Rad'])

# Alternatively, the cdf.loc['Humboldt'] find Humboldt first and needs to only
# search this row to find the radius. 
len(cdf.loc['Humboldt'])

# The second indexing is usually more efficient but the most efficient way to
# return a single cell is using the at method.
cdf.at['Humboldt', 'Rad']

# DataFrames also support fancy (boolean) indexing. To get all crater larger 
# than 9 km
cdf[cdf['Rad'] > 9]

# Or to get all craters with longitude between 0 and 10 degrees
cdf[(cdf['Lon'] > 0) & (cdf['Lon'] < 10)]

# Conditions ca be strung together to subset or "clean" the data
cdf = cdf[(cdf['Lat'] > -80) & (cdf['Lat'] < 80)]
#-----------------------------------------------------------------------------
# To get an ROI from the AceDataset, the crater lat, lon and rad is required
lat, lon, rad = cdf.cloc('Copernicus')
lat, lon, rad = [9.62, 20.08, 93]
roi = ads.getROI(lat, lon, rad, wsize=5)

# To plot the roi
ads.plotROI(roi)


# Lets get a few random lines from cdf. Setting the seed produces the same 
# random numbers on each run
rnd.seed(5)
sample = rnd.randint(0, len(cdf), 5)

# Loop through the names in the cdf index then extract the lat, lon, radius 
# then get and plot the rois
for name in cdf.index[sample]:
    lat = cdf.at[name, 'Lat']
    lon = cdf.at[name, 'Lon']
    rad = cdf.at[name, '_Rad']
    roi = ads.getROI(lat, lon, rad, wsize=4)
    ads.plotROI(roi, figsize=(4,4))
    
    
#----------------------------------------------------------------------------
# Statistics can be performed on an roi using the acestats module

    
    
