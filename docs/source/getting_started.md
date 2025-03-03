# Getting Started

The `craterpy` module provides the `CraterDatabase` class which can be imported as:

```python
from craterpy import CraterpyDatabase
cdb = CraterpyDatabase("craters.csv", body="Moon", units="km")
```

## Importing a list of craters

Instantiating a `CraterDatabase` requires a table of craters (either a file or `pandas.DataFrame`) with the following columns (case-insensitive):

- `Latitude` or `Lat`
- `Longitide` or `Lon`
- `Radius` or `Rad` or `Diameter` or `Diam`

If there are multiple columns that match, `craterpy` assumes the first matching column is desired.

To specify specific columns or rows to use, first read the table into a `pandas.DataFrame`, filter it and pass in the subset DataFrame instead:

```python
import pandas as pd
df = pd.read_csv("craters.csv", usecols=[4, 8, 9], names=["lat", "lon", "diam"])
df = df[df.diam > 5]
cdb = CraterpyDatabase(df, body="Moon", units="km")
```

The `CraterDatabase` will convert crater sizes in diameter to radius for its computations automatically. However, the length units of the craters must be specified as meters or kilometers (default is "m").

The planetary `body` must also be specified to ensure the correct coordinate reference system (CRS) is used.

## Adding geometries to a CraterDatabase

A new geometry can be defined for each crater in the database. Circular and annular geometries have sizes specified in units of number of crater radii from the center of the crater (e.g. size=1 is a circle at the crater rim, size=2 is one radius beyond the rim, size=3 is one diameter beyond the rim, etc). Circles enclose all area within `size` radii from each crater. Annuli cut out the center and enclose the area from `inner` to `outer`.

```python
# Cicles centeres on each crater
cdb.add_circles("crater", size=1)
# Annulus from rim to 1 crater diameter past the rim (3 radii from the center)
cdb.add_annuli("ejecta", inner=1, outer=3)  
```

Geometries are stored as `geopandas.GeoSeries` objects and can be accessed by the suplied name. Operations on `GeoSeries` are possible, including saving the geometries to a geojson shapefile that is importable to most GIS applications.

```python
cdb.craters.to_file("lunar_craters.geojson")
cdb.ejecta.to_file("lunar_ejecta.geojson")
```

Geometries can be shown on a plot with:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax = cdb.plot(name="ejecta", alpha=0.5, color='tab:green')
ax.set_xlim(0, 180)  # Subset to E hemisphere
ax.set_ylim(0, 90)  # Subset to N hemisphere
```

An image with coordinates in degrees can be supplied to the same axis, or if the bounds are known you may supply an extent (see `GeoSeries.plot()` and `matplotlib` docs for more plotting info):

```python
im = plt.imread('moon.tif')
ax = cdb.plot(alpha=0.5, color='tab:green')
ax.imshow(im, extent=(-180, 180, -90, 90), cmap='gray')
```

## Extracting statistics

Stats can be computed from the crater geometries from a supplied georeferenced raster with:

```python
cdb = CraterDatabase(df, "Moon", units="km")

# Define regions for central peak, crater floor, and rim (sizes in crater radii)
cdb.add_annuli("peak", 0, 0.1)
cdb.add_annuli("floor", 0.3, 0.6)
cdb.add_annuli("rim", 1.0, 1.2)

# Here, DEM is a geotiff with elevation relative to reference Moon
stats = cdb.get_stats("dem.tif", regions=['floor', 'peak', 'rim'], stats=['mean', 'std'])
```

For a list of valid statistics, see the [rasterstats documentation](https://pythonhosted.org/rasterstats/manual.html#zonal-statistics).

The input raster file must be georeferenced and importable by `gdal`.

## More help

If you encounter any bugs or issues, please check out the [Issues board](https://github.com/cjtu/craterpy/issues) to see if others had the same questions or to report what you found. We also welcome feature requests and contributions!
