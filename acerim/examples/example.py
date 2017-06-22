# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:20:20 2017

@author: Christian
"""
import sys
import os.path
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, d)

#%% Import acerim modules
import classes as ac
import functions as af
import acestats as acs

# Import dataset from tests
ds_path = os.path.join(d, 'tests', 'moon.tif')
ads = ac.AceDataset(ds_path, radius=1737)

# Import crater csv from tests
csv_path = os.path.join(d, 'tests', 'craters.csv')
cdf = ac.CraterDataFrame(csv_path)

# Check crater data
cdf.head()

# Make the crater names in the first column the index 
cdf = ac.CraterDataFrame(csv_path, index_col=0)
cdf.head()

# Find Aratus. Note the use of square brackets.
cdf.loc['Aratus']

# Get only the radius of Aratus
cdf['Rad'].loc['Aratus']

# Or
cdf.loc['Aratus']['Rad']

# The difference is that cdf['Rad'] takes the radius series out of the
# dataframe and then searches through it for 'Aratus' 
len(cdf['Rad'])

# Alternatively, the cdf.loc['Aratus'] find aratus first and needs to only
# search this row to find the radius. 
len(cdf.loc['Aratus'])

# The second indexing is usually more efficient but the most efficient way to
# return a single cell is using the at method.
cdf.at['Aratus', 'Rad']

# DataFrames also support fancy (boolean) indexing. To get all crater larger 
# than 9 km
cdf[cdf['Rad'] > 9]

# Or to get all craters with longitude between 0 and 10 degrees
cdf[(cdf['Lon'] > 0) & (cdf['Lon'] < 10)]

#-----------------------------------------------------------------------------
# To get an ROI from the AceDataset, the crater lat, lon and rad is required
lat, lon, rad = cdf.cloc('Copernicus')
lat, lon, rad = [9.62, 20.08, 93]
roi = ads.getROI(lat, lon, rad, wsize=5)

# To plot the roi
ads.plotROI(roi)


# Lets get a few lines from cdf and plot their rois
sample = cdf[(cdf['Lon'] > 170) & (cdf['Lat'] < 80)]#cdf.head()#cdf.sample(5)
lats = sample['Lat']
lons = sample['Lon']
rads = sample['_Rad']


for name in sample.index:
    roi = ads.getROI(lats[name], lons[name], rads[name], wsize=5)
    ads.plotROI(roi, figsize=(4,4))
