# Getting Started

The basis of `craterpy` is the `CraterDatabase` which reads crater data, manages coordinate reference systems (CRS), digitizes regions of interest, and plots or extracts statistics from supplied crater locations.

## Importing Craters

The `CraterDatabase` takes a table of craters with at least columns for latitude, longitude, and radius or diameter of the craters. The exact names of the columns are not strict, ex 'Lat' and 'rad' are recognized, but if multiple columns match, `craterpy` will assume the first is correct.

```{note}
To choose specific columns or filter to desired rows, first read the table into a `pandas.DataFrame` and pass the filtered subset dataframe to `CraterDatabase`.
```

Here we read a sample list of lunar craters, optionally filter it to keep only the largest and then improt it into a `CraterDatabase`:

```python
from craterpy import CraterDatabase, sample_data
import pandas as pd
crater_file = sample_data["moon_craters_km.csv"]
cdb_all = CraterDatabase(crater_file, body="Moon", units="km")
print(cdb_all)
# Moon CraterDatabase (N=786)

# Or filter/clean data with pandas first
df = pd.read_csv(sample_data["moon_craters_km.csv"])
df = df[df["Diameter (km)"] > 100]
cdb = CraterDatabase(df, body="Moon", units="km")
print(cdb)
# Moon CraterDatabase (N=222)
```

Planetary `body` is a required `CraterDatabase` parameter to ensure the correct coordinate reference system (CRS) is used (read more about CRS [here](https://docs.qgis.org/3.40/en/docs/gentle_gis_introduction/coordinate_reference_systems.html)). Supported bodies are given in `craterpy.all_bodies`. The CRS defaults to the `IAU_2015` CRS most commonly used for the particular body, see list [here](https://planetarynames.wr.usgs.gov/TargetCoordinates).  A custom input CRS for the crater coordinates can be specified as follows:

```python
from craterpy import all_bodies, CraterDatabase, sample_data
print(all_bodies)
# ['mercury', 'venus', 'moon', 'earth', 'mars', 'ceres', 'vesta', 'europa', 'ganymede', 'callisto', 'enceladus', 'tethys', 'dione', 'rhea', 'iapetus', 'pluto']

custom_crs = "IAU_2015:200000100"  # Can be any CRS string recognized by pyproj
cdb = CraterDatabase(sample_data["ceres_craters_km.csv"], "ceres", input_crs=custom_crs, units="km")
```

```{note}
Make sure to specify `units="m"` (meters) or `units="km"` (kilometers) for crater size (default is `m`). Size will be determined by the first radius or diameter column in the dataframe with the given units.
```

## Adding regions 

New regions produce shapefile geometries for every crater in the `CraterDatabase`. Circular and annular regions are defined in units of crater radii from the center:

- `size=1`: 1 radius from the center (the circular crater rim)
- `size=2`: 2 raddii from the center (one radius beyond the rim)
- `size=3`: 3 radii from the center (one diameter beyond the rim)
- etc... 

Circles cover all enclosed area within `size` radii. Annuli are defined from the `inner` to `outer` radius and exclude the inner circle at the center.

```python
from craterpy import CraterDatabase, sample_data as sd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
cdb = CraterDatabase(sd["moon_craters_km.csv"], body="Moon", units="km")

# Default plot is circular ROI at 1 crater radius
cdb.plot()

# Explicitly add circular ROIs and plot on a raster
cdb.add_circles("crater", size=1)
cdb.plot(sd["moon.tif"], "crater")

# Annular ROIs with some plot customization
cdb.add_annuli("ejecta", inner=1, outer=3)
ax = cdb.plot(sd["moon.tif"], "ejecta", linestyle="--", color="tab:green", alpha=0.4, size=8, dpi=200)
ax.set_ylim(-30, 70)
ax.set_xlim(-90, 90)
plt.title("New title here")
plt.show()
```

## Plotting ROIs

In addition to plotting the full `CraterDatabase` as above, individual crater ROIs can be shown if a georeferenced raster image is given.


```python
from craterpy import CraterDatabase, sample_data as sd
cdb = CraterDatabase(sd["tethys_craters_km.csv"], body="tethys", units="km")

cdb.add_circles("crater", size=2)
cindex = cdb.data.iloc[10:19].index
cdb.plot_rois(sd["tethys.tif"], "crater", range(1,10))

cdb.add_annuli("ejecta", 1, 2)
cdb.plot_rois(sd["tethys.tif"], "ejecta", range(1,10), color='black', cmap="cividis", grid_kw={'alpha': 0})
```


```{note}
See other sample planetary crater data that can be plotted with `craterpy` in {doc}`planetary_body_examples`.
```

## Exporting ROIs to GIS

Regions of interest can be saved as GeoJSON shapefiles for use in any GIS application (e.g. ArcGIS or QGIS) using `.to_geojson()`.

```python
from craterpy import CraterDatabase, sample_data as sd

cdb = CraterDatabase(sd["mars_craters_km.csv"], body="Mars", units="km")
cdb.add_circles("crater", size=1)
print(cdb)
# Mars CraterDatabase (N=352)

# Save to shapefile
cdb.to_geojson("mars_craters.geojson", "crater")

# Read from shapefile
new_cdb = CraterDatabase("mars_craters.geojson", body="Mars", units="km")
print(new_cdb)
# Mars CraterDatabase (N=352)
```

## Extracting statistics

Stats can be computed from the crater geometries from a supplied georeferenced raster with:

```python
from craterpy import CraterDatabase, sample_data as sd
import pandas as pd

df = pd.read_csv(sd["moon_craters_km.csv"])
cdb = CraterDatabase(df[df["Diameter (km)"] > 20], "Moon", units="km")

# Define regions for crater floor, rim, and ejecta (sizes in crater radii)
cdb.add_annuli("floor", 0.4, 0.6)  # crater floor, excluding possible central peak
cdb.add_annuli("rim", 0.99, 1.01)  # thin annulus at rim
cdb.add_annuli("background", 5, 5.5) # outside the continuous ejecta

# Pull statistics from Lunar Digital Elevation Model (DEM)
stats = cdb.get_stats(sd["moon_dem.tif"], regions=['floor', 'rim', 'background'], stats=['mean'])

# Use mean elevations to compute depth (rim to floor) and height of rim above background
stats['crater_depth'] = (stats.mean_rim - stats.mean_floor)
stats['rim_height'] = (stats.mean_rim - stats.mean_background)
print(stats.head())
ax = stats[["crater_depth", "rim_height"]].plot(kind="hist", bins=100, alpha=0.8)
ax.set_xlabel('Height (m)')
# Expect depths to be greater than rim height
```

For a list of valid statistics, see the [rasterstats documentation](https://pythonhosted.org/rasterstats/manual.html#zonal-statistics).

The input raster file must be georeferenced and importable by `gdal`.

## Example: Plot all lunar craters near Orientale Basin

First download the [Lunar Crater Database](https://pdsimage2.wr.usgs.gov/Individual_Investigations/moon_lro.kaguya_multi_craterdatabase_robbins_2018/data/) (Robbins, 2018) in CSV format. Currently `craterpy` works on any tabular data with a diam, lat, and lon column. Since crater databases are often not supplied in a standardized and labelled PDS format, some manual data cleaning may be necessary to ensure the correct columns, planetary body, and units are used.

Here we read the data with `pandas` and filter to the desired columns and area. We rename the diam, lat, and lon columns for convenience, then pass only those columns to `CraterDatabase` to ensure `craterpy` uses the correct values (the Robbins CSV contains mulitple columns corresponding to each). Pre-filtering the region desired saves on computation time.

Here, taking only craters larger than 5km results in ~12500 craters vs. all craters larger than 1 km results in ~175k craters. On `craterpy v0.9.5` these took ~20s and ~5 minutes to plot, respectively (when tested on a Ryzen 7 laptop with 8 cores at 1.9 GHz and 24GB RAM).

```python
from craterpy import CraterDatabase, sample_data as sd
import pandas as pd
df = pd.read_csv('/home/cjtu/projects/craterpy/tmp/lunar_crater_database_robbins_2018.csv')
df = df.rename(columns={'DIAM_CIRC_IMG':'diam', 'LAT_CIRC_IMG':'lat', 'LON_CIRC_IMG':'lon'})
df = df[['diam', 'lat', 'lon']]
df = df[(-70 < df.lat) & (df.lat < 30)
        & (220 < df.lon) & (df.lon < 320)
        & (1 < df.diam) & (df.diam < 300)]

# Subset to only 5 km and larger craters, plot on DEM
df5 = df[(5 < df.diam)]
cdb5 = CraterDatabase(df5, "Moon", units="km")
cdb5.plot(sd['moon_dem.tif'], alpha=0.3, color='k', savefig='readme_orientale_robbins_gt5.png')
```

![Lunar craters plot](https://github.com/cjtu/craterpy/raw/main/docs/_images/readme_orientale_robbins_gt5.png)


```python
# Plot all craters >1km, this may take some time!
cdb = CraterDatabase(df, "Moon", units="km")
cdb.plot(sd['moon_dem.tif'], alpha=0.3, color='k', savefig='readme_orientale_robbins.png')
```

![Lunar craters plot](https://github.com/cjtu/craterpy/raw/main/docs/_images/readme_orientale_robbins.png)

For more plotting customization options using `cartopy`, see the this page in their [docs](https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/intro.html).

## More help

If you encounter any bugs or issues, please check out the [Issues board](https://github.com/cjtu/craterpy/issues) to see if others had the same questions or to report what you found. We also welcome feature requests and contributions!
