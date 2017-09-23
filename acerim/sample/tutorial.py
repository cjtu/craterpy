"""
Welcome to the ACERIM tutorial!
===============================

**IN PROGRESS**

This will walk you through an example of a research script using ACERIM to 
introduce and explain the various features of the package. Full API 
documentation is also available at www.readthedos.org/projects/acerim.
"""

""" Housekeeping ""
For this tutorial to run on your system without you specifying a path to your 
ACERIM installation, we will need to import the built-in module "os". You 
should not require os for your general ACERIM workflow because you can simply
specify the path to your directory.
"""
import os.path


""" Importing classes from ACERIM """
"""A typical acerim workflow will begin by importing the 2 main ACERIM classes,
the AceDataset and the CraterDataFrame. Here we shorten these to Ads and Cdf, 
respectively, to avoid typing acerim.aceclasses.AceDataset each time we want to
load an image.
"""
from acerim.aceclasses import AceDataset as Ads
from acerim.aceclasses import CraterDataFrame as Cdf


""" Path to your data """
"""In general, you can specify data_dir directly with your desired file path.
"""
data_dir = '/path/to/your/data'

"""Here, we will use os to find this file's directory (where our sample data is
located):
"""
sample_dir = os.path.dirname(os.path.abspath(__file__))


""" The AceDataset """

""" The AceDataset class is used to import and manipulate image data. Note that 
acerim assumes that supplied image data is in a simple cylindrical projection. 
If a geotiff is supplied, the AceDataset will attempt to read and store the 
geographical information automatically. Importing moon.tif from image_data_path:
"""
moon_image_path = sample_dir + '/moon.tif'
moon = Ads(moon_image_path)
print(moon)

""" Printing the AceDataset (called "moon") is a good way to test that the image 
was imported, but it also gives a breakdown of the geospatial information 
it contains. In this case, moon printed the wrong radius for the Moon (it 
should be 1737.4 km, not 6378 km). All geospatial information is saved
in moon's attributes. In this case the offending attribute can be checked with: 
"""
moon.radius

""" Since moon has the wrong radius, we will have to make a new AceDataset, 
this time specifying the correct radius. Any of the geospatial parameters can 
be set by specifying them in the constructor:
"""
moon = Ads(moon_image_path, nlat=90, slat=-90, wlon=-180, elon=180, 
		   radius=1737.4, ppd=4)
print(moon)

""" The main feature of the AceDataset is extracting 2D image arrays (dubbed 
ROIs) from crater locations in the dataset. Say we want to see Copernicus 
crater on the Moon. We will need to supply the AceDataset.get_roi method with
the latitude, longitude and radius of Copernicus.
"""
cop_lat, cop_lon, cop_rad = [9.62, 20.08, 93]

""" We will also need to supply a window size (in crater radii). Let's say we 
want to see 5 crater radii from the center of the crater:
"""
cop_wsize = 5

""" Now we have all we need to extract an ROI from moon:
"""
cop_roi = moon.get_roi(cop_lat, cop_lon, cop_rad, cop_wsize)

""" ROIs take the form of numpy 2D arrays. We can check the size of our ROI
with:
"""
cop_roi.shape

""" Now let's take a look at Aristarchus, but plot the ROI on the fly:
"""
arist_roi = moon.get_roi(23.7, -47.4, 40, 5, plot_roi=True)

""" To have more plotting options you can pass the roi to the plot_roi method 
of AceDataset:
"""
moon.plot_roi(arist_roi, figsize=(5,5), title='Aristarchus', vmin=0, vmax=75,
			  cmap='jet')


""" The CraterDataFrame """

""" In this section we will explore how to use the CraterDataFrame to store and
access crater data in this tabular data structure.
"""
crater_sheet_path = sample_dir + '/craters.csv'

""" Importing a crater spreadsheet """
""" A CraterDataFrame can be initialized by passing a string to the 
constructor. Doing so assumes that the string is a file path to a csv (comma 
separated values) file:
"""
craters_fromcsv = Cdf(crater_sheet_path)

""" Data can also be read from a properly formatted dictionary:
"""
cdict = {'Lat' : [10, -20., 80.0], 
         'Lon' : [14, -40.1, 317.2],
         'Diam' : [2, 12., 23.7]}
craters_fromdict = Cdf(cdict)

""" The CraterDataFrame also accepts pandas.DataFrame objects, so that if 
other import options are needed (importing from an Excel spradsheet, specifying
index columns, etc), this can be handled with pandas and then passed to the
CraterDataFrame like so:
"""
import pandas
dataframe = pandas.read_csv(crater_sheet_path)
craters_frompandas = Cdf(dataframe)

""" A final import option that is useful for simplying queries is the index
option. If you have a column in your data with crater names or unique indices,
you can specify its position to make it the default DataFrame index. In this 
case it is the first column of data (position 0 in python):
"""
craters = Cdf(crater_sheet_path, index_col=0)

""" Visualizing and Querying the CraterDataFrame """
""" Methods that are available to the pandas.DataFrame are also available to
the CraterDataFrame. For instance to give a quick check of the first few lines 
of data, we can use the pandas.DataFrame method "head":
"""
craters.head()

""" To query a crater assuming we set the index to the Name column, we use 
loc['Crater Name']:
"""
craters.loc['Humboldt']

""" There are different way to get a value from the CraterDataFrame. This will 
return the longitude of the crater:
"""
craters.loc['Humboldt']['Lon']

""" But using ".at[]" is the most efficient way to get a single cell of data:
"""
craters.at['Humboldt', 'Lon']

""" DataFrames also support fancy (boolean) indexing. To get all crater larger 
than 9 km in diameter:
"""
craters[craters['Diam'] > 9]

""" Or to get all craters with longitude between 0 and 10 degrees:
"""
craters[(craters['Lon'] > 0) & (craters['Lon'] < 10)]

""" Conditions ca be strung together to filter your data:
"""
craters = craters[(craters['Lat'] > -80) & (craters['Lat'] < 80)]


""" Advanced Examples """

""" Loop through the first 5 names in the craters index, 
extract the lat, lon, radius and plot the ROIs
"""
for name in craters.index[:5]:
    lat = craters.at[name, 'Lat']
    lon = craters.at[name, 'Lon']
    rad = craters.at[name, '_Rad']
    roi = moon.get_roi(lat, lon, rad, wsize=4, plot_roi=True)

        
""" The acestats module """

""" Statistics can be performed on an roi using the acestats module.
"""
# TODO: acestats examples
    
    
