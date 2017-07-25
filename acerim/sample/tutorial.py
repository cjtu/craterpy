"""
Welcome to the ACERIM tutorial!
===============================

This will walk you through an example of a research script using ACERIM to 
introduce and explain the various features of the package. Full API 
documentation is also available at www.readthedos.org/projects/acerim.

For the purpose of this tutorial, we first need to import os to get the 
absolute path to our sample data directory. We then store the path to our 
sample crater sheet and sample image (craters.csv and moon.tif, respectively).
"""
import os.path
data_dir = os.path.dirname(os.path.abspath(__file__))
image_data_path = os.path.join(data_dir, 'moon.tif')
crater_sheet_path = os.path.join(data_dir, 'craters.csv')

"""In general, you can specify data_dir directly with your desired file path."""
# data_dir = /path/to/your/data

"""A typical acerim workflow will begin by importing the 3 main acerim modules:"""
from acerim import aceclasses as ac
from acerim import acefunctions as af
from acerim import acestats as acs

"""
aceclasses.AceDataset 
---------------------

The AceDataset class is used to import and manipulate image data. Note that 
acerim assumes that supplied image data is in a simple cylindrical projection. 
If a geotiff is supplied, the AceDataset will attempt to read and store the 
geographical information automatically. Importing moon.tif from image_data_path:
"""

ads = ac.AceDataset(image_data_path)
print(ads)
# AceDataset object with bounds (90.0N, -90.0S), (-180.0E, 180.0E), radius 6378.137 km, and 4.0 ppd resolution

"""It seems that the geospatial information supplied the wrong radius for the 
Moon (it should be 1737.4 km). We can set any of the geographical bounds, the 
radius of the planetary body, or the dataset resolution by specifying the 
desired parameters when initializing an AceDataset.
"""

ads = ac.AceDataset(image_data_path, nlat=90, slat=-90, wlon=-180, elon=180, 
                    radius=1737.4, ppd=4)
print(ads)
# AceDataset object with bounds (90N, -90S), (-180E, 180E), radius 1737.4 km, and 4 ppd resolution

"""Now we can explore some features of the AceDataset"""
# TODO: AceDataset features
# To get an ROI from the AceDataset, the crater lat, lon and rad is required

lat, lon, rad = [9.62, 20.08, 93]
roi = ads.get_roi(lat, lon, rad, wsize=5)

# To plot the roi
ads.plot_roi(roi)


"""
aceclasses.CraterDataFrame
--------------------------

In this section we will look at the other major ACERIM class, the CraterDataFrame.
"""

cdf = ac.CraterDataFrame(crater_sheet_path)

# Check crater data
cdf.head()

# Make the crater names in the first column the index 
cdf = ac.CraterDataFrame(crater_sheet_path, index_col=0)
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


# Loop through the names in the cdf index then extract the lat, lon, radius 
# then get and plot the rois
for name in cdf.index[:5]:
    lat = cdf.at[name, 'Lat']
    lon = cdf.at[name, 'Lon']
    rad = cdf.at[name, '_Rad']
    roi = ads.get_roi(lat, lon, rad, wsize=4)
    ads.plot_roi(roi, figsize=(4,4))
    
    
    
"""
The acestats module
-------------------

Statistics can be performed on an roi using the acestats module.
"""
# TODO: acestats examples
    
    
