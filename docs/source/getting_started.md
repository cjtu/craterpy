# Getting Started

The basis of `craterpy` is the `CraterDatabase` which reads, digitizes, plots, and extracts statistics from supplied crater locations.

```python
from craterpy import CraterDatabase
cdb = CraterDatabase("/path/to/craters.csv", body="Moon")
```

## Importing a list of craters

Instantiating a `CraterDatabase` requires a table of craters, the planetary body, and the length units of the radius/diameter column ("m" or "km"). Data can be supplied as a file path or `pandas.DataFrame` that must contain the following named columns in any order (case-insensitive):

- `Latitude` or `Lat`
- `Longitide` or `Lon`
- `Radius` or `Rad` or `Diameter` or `Diam`

When multiple columns match, `craterpy` uses the first column found.

```{note}
To choose specific columns or filter to desired rows, first read the table into a `pandas.DataFrame` and pass the filtered subset dataframe to `CraterDatabase`.
```

Here we read a sample list of lunar craters, filter it to larger than 55 km and import it into a `CraterDatabase`:

```python
from craterpy import CraterDatabase, sample_data as sd
import pandas as pd
df = pd.read_csv(sd["moon_craters.csv"])
df = df[df['Rad'] > 55]
cdb = CraterDatabase(df, body="Moon", units="km")
print(cdb)
# CraterDatabase of length 203 with attributes lat, lon, rad, center.
```

Planetary `body` is required to ensure the correct coordinate reference system (CRS) is used. Supported bodies can be printed with:

```python
import craterpy
print(craterpy.all_bodies)
# ['moon', 'mars', 'mercury', 'venus', 'europa', 'ceres', 'vesta']
```

```{note}
Crater sizes can be given as radius or diameter, however, the length units should be specified as meters (`m`) or kilometers (`km`). Default is `units="m"`.
```

## Adding geometries to a CraterDatabase

A new geometry can be defined for each crater in the database. Circular and annular geometries have sizes specified in units of number of crater radii from the center of the crater (e.g. `size=1` is a circle at the crater rim, `size=2` is one radius beyond the rim, size=3 is one diameter beyond the rim, etc). Circles enclose all area within `size` radii from each crater. Annuli cut out a circle at the center and enclose the area from `inner` to `outer`.

```python
# Cicles centeres on each crater
cdb.add_circles("crater", size=1)
# Annulus from rim to 1 crater diameter past the rim (3 radii from the center)
cdb.add_annuli("ejecta", inner=1, outer=3)  
```

Geometries are stored under the supplied name as `geopandas.GeoSeries` objects. These geometries contain map projection information and can be saved and read from a GeoJSON shapefile with `.to_geojson()`, a fomat that is widely recognized by most GIS applications.

```python
# Save to shapefile
cdb.to_geojson("lunar_craters.geojson", "crater")
# Read from shapefile
new_cdb = CraterDatabase("lunar_craters.geojson", body="Moon", units="km")
```

Geometries can be shown on a plot with with or without an image (images should be converted to a georeferenced raster format before use with `craterpy`).

```python
cdb.plot()
# <circular crater outlines on blank white background>

cdb.plot(sd["moon.tif"])
# <circular crater outlines plotted on supplied raster map image moon.tif>

# Plot the craters on a larger image with higher resoltion
cdb.plot(sd["moon.tif"], "crater", size=10, dpi=200)
```

Individual regions of interest ROIs can also be shown as an array of subplots.

```python
cdb.plot_rois(sd["moon.tif"], "crater")
# <subplot of craters>

cdb.add_annuli("rim", 0.5, 1.56)
cdb.plot_rois(sd["moon.tif"], "rim", color='black', cmap="cividis", grid_kw={'alpha': 0})
```

## Extracting statistics

Stats can be computed from the crater geometries from a supplied georeferenced raster with:

```python
cdb = CraterDatabase(sd["moon_craters.csv"], "Moon", units="m")

# Define regions for central peak, crater floor, and rim (sizes in crater radii)
cdb.add_annuli("peak", 0, 0.1)
cdb.add_annuli("floor", 0.3, 0.6)
cdb.add_annuli("rim", 1.0, 1.2)

stats = cdb.get_stats(sd["moon.tif"], regions=['floor', 'peak', 'rim'], stats=['mean', 'std'])
print(stats.head())
```

For a list of valid statistics, see the [rasterstats documentation](https://pythonhosted.org/rasterstats/manual.html#zonal-statistics).

The input raster file must be georeferenced and importable by `gdal`.

## All craters on the moon > 2km

To plot all > 2km craters on the moon, first download the [Lunar Crater Database](https://pdsimage2.wr.usgs.gov/Individual_Investigations/moon_lro.kaguya_multi_craterdatabase_robbins_2018/data/) (Robbins, 2018) in CSV format.

Then run the following code to read in, filter, and ingest the database as a `CraterDatabase`, and then visualize it. **Note:** visualization of so many craters may take several minutes, or even tens of minutes depending on the machine, to appear.

```python
from craterpy import CraterDatabase
import pandas as pd
df = pd.read_csv('lunar_crater_database_robbins_2018.csv')
df = df[df['DIAM_CIRC_IMG'] > 2]  # Filter out craters < 2 km diameter 
cdb = CraterDatabase(df, "Moon", units="km")
print('Plotting... (this may take several minutes)')
cdb.plot(linewidth=0.25, alpha=0.25, color='gray')
```

![Lunar craters plot](https://github.com/cjtu/craterpy/raw/main/docs/_images/readme_moon_robbins.png)

## More help

If you encounter any bugs or issues, please check out the [Issues board](https://github.com/cjtu/craterpy/issues) to see if others had the same questions or to report what you found. We also welcome feature requests and contributions!
