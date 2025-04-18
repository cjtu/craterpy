# Getting Started

The basis of `craterpy` is the `CraterDatabase` which reads, digitizes, plots, and extracts statistics from supplied crater locations.

```python
from craterpy import CraterpyDatabase
cdb = CraterpyDatabase("my_craters.csv", body="Moon", units="km")
```

## Importing a list of craters

Instantiating a `CraterDatabase` requires a table of craters (either a file or `pandas.DataFrame`) with the following columns (case-insensitive):

- `Latitude` or `Lat`
- `Longitide` or `Lon`
- `Radius` or `Rad` or `Diameter` or `Diam`

If there are multiple columns that match, `craterpy` assumes the first matching column is desired.

To specify specific columns or rows to use for the `lon`, `lat`, and `diam` or `rad` columns, first read the table into a `pandas.DataFrame` and pass in the desired columns. You may also filter the crater list. Here, we take only craters larger than 5 km:

```python
from craterpy import sample_data as sd
import pandas as pd
df = pd.read_csv(sd["craters.csv"], usecols=[4, 8, 9], names=["lat", "lon", "diam"])
df = df[df.diam > 5]
cdb = CraterpyDatabase(df, body="Moon", units="km")
```

The `CraterDatabase` will work with either a radius or diameter column. However, the length units of the craters must be specified as meters or kilometers (default is `units=m`).

The planetary `body` must also be specified to ensure the correct coordinate reference system (CRS) is used.

## Adding geometries to a CraterDatabase

A new geometry can be defined for each crater in the database. Circular and annular geometries have sizes specified in units of number of crater radii from the center of the crater (e.g. size=1 is a circle at the crater rim, size=2 is one radius beyond the rim, size=3 is one diameter beyond the rim, etc). Circles enclose all area within `size` radii from each crater. Annuli cut out a circle at the center and enclose the area from `inner` to `outer`.

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
new_cdb = CraterDatabase("lunar_craters.geojson")
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

## More help

If you encounter any bugs or issues, please check out the [Issues board](https://github.com/cjtu/craterpy/issues) to see if others had the same questions or to report what you found. We also welcome feature requests and contributions!
