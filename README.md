<div align="center">
  <strong>Craterpy:</strong><em> Impact crater data science in Python.</em>
</div>

<div align="center">
  <!-- PYPI version -->
  <a href="https://badge.fury.io/py/craterpy">
    <img src="https://badge.fury.io/py/craterpy.svg"
      alt="PYPI version" />
  </a>
  <!-- Code quality and testing (CI) -->
  <a href="https://github.com/cjtu/craterpy/actions">
    <img src="https://github.com/cjtu/craterpy/workflows/Code%20Quality%20Checks/badge.svg"
      alt="Code Quality Checks" />
  </a>
  <!-- Test Coverage (codecov) -->
  <a href="https://codecov.io/gh/cjtu/craterpy">
    <img src="https://codecov.io/gh/cjtu/craterpy/branch/trunk/graph/badge.svg?token=9K567x0YUJ"
      alt="Code Coverage" />
  </a>
</div>
<div align="center">
  <!-- Zenodo citation -->
  <a href="https://zenodo.org/badge/latestdoi/88457986">
    <img src="https://zenodo.org/badge/88457986.svg"
      alt="Cite on Zenodo" />
  </a>
  <!-- ReadTheDocs -->
  <a href="http://craterpy.readthedocs.io/en/latest/?badge=latest">
    <img src="http://readthedocs.org/projects/craterpy/badge/?version=latest"
      alt="Cite on Zenodo" />
  </a>
  <!-- Code Style -->
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg"
      alt="Code Style: Black" />
      </a>
</div>

# Overview

Craterpy makes it easier to work with impact crater data in Python. Highlights:

- convert a table of crater data to a GeoDataFrame or GIS-ready shapefile
- extract zonal statistics associated with each crater in circlular or annular regions (rasterstats)
- eliminate some pain points of planetary GIS analysis (antimeridian wrapping, projection conversions, etc.)

Note: craterpy is not a detection algorithm (e.g., [PyCDA](https://github.com/AlliedToasters/PyCDA)), nor is it a crater count age dating tool (see [craterstats](https://github.com/ggmichael/craterstats)).

**Note:** *Craterpy is in beta. We appreciate bug reports and feature requests on the [issues board](https://github.com/cjtu/craterpy/issues).*

## Examples

All craters on the Moon > 2km.

```python
from craterpy import CraterDatabase
cdb = CraterDatabase('lunar_crater_database_robbins_2018.csv', "Moon", units="km")
cdb.plot(linewidth=0.25, alpha=0.25, color='gray')
```

![Lunar craters plot](https://github.com/cjtu/craterpy/raw/trunk/craterpy/data/_images/readme_moon_robbins.png)


Define cicular/annular regions and export to GIS-ready shapefiles:

```python
cdb.add_circles("craters", size=1)  # Crater interiors
cdb.add_annuli("ejecta", inner=1, outer=3)  # Annulus from rim to 2 radii past the rim (excludes interior)
cdb.craters.to_file("lunar_craters.geojson")
cdb.ejecta.to_file("lunar_ejecta.geojson")
```

Plot on a raster

```python
cdb = CraterDatabase('vesta_craters.csv', "Vesta", units="m")
im = plt.imread('vesta.tif')
ax = cdb.plot(alpha=0.5, color='tab:green')
ax.imshow(im, extent=(-180, 180, -90, 90), cmap='gray')
```

![Vesta map plot](https://github.com/cjtu/craterpy/raw/trunk/craterpy/data/_images/readme_vesta_cdb.png)


Compute raster image statistics on each crater geometry.

```python
import pandas as pd
df = pd.DataFrame({'Name': ["Orientale", "Copernicus", "Tycho"],
                    'Lat': [-19.9, 9.62, -43.35],
                    'Lon': [-94.7, -20.08, -11.35],
                    'Rad': [250., 48., 42.]})
cdb = CraterDatabase(df, "Moon", units="km")

# Define regions for central peak, crater floor, and rim (sizes in crater radii)
cdb.add_annuli("peak", 0, 0.1)
cdb.add_annuli("floor", 0.3, 0.6)
cdb.add_annuli("rim", 1.0, 1.2)

# DEM is a geotiff with elevation relative to reference Moon in meters
stats = cdb.get_stats("dem.tif", regions=['floor', 'peak', 'rim'], stats=['mean', 'std'])
print(stats)
```

| **Name** | **Lat** | **Lon** | **Rad** | **mean_floor** | **std_floor** | **mean_peak** | **std_peak** | **mean_rim** | **std_rim** |
|---|---|---|---|---|---|---|---|---|---|
| Orientale | -19.90 | -94.70 | 250.0 | -2400.0 | 400.0 | -2800.0 | 100.0 | 400.0 | 1100.0 |
| Compernicus | 9.62 | -20.08 | 48.0 | -3400.0 | 200.0 | -3400.0 | 100.0 | -0.0 | 200.0 |
| Tycho | -43.35 | -11.35 | 42.0 | -3200.0 | 400.0 | -2100.0 | 500.0 | 900.0 | 400.0 |


See the full [craterpy documentation](https://readthedocs.org/projects/craterpy/) on Read the Docs.


## Installation

With pip:

```bash
pip install craterpy
```

From the repo with [poetry](https://python-poetry.org/docs/) (for latest version & to contribute). First fork and clone the repository, then:

```bash
# Install craterpy with poetry
$ cd craterpy
$ poetry install

# Check installation version
poetry version

# Activate the venv 
$ poetry shell
$ which python

# Or open a Jupyter notebook
$ poetry run jupyter notebook
```
- **Note**: Craterpy is currently only tested on Ubuntu and OS X. If you would like to use craterpy on Windows, check out the Windows Subsystem for Linux ([WSL](https://docs.microsoft.com/en-us/windows/wsl/install)). 

Trouble installing craterpy? Let us know on the [issues](https://github.com/cjtu/craterpy/issues) board.


## Documentation

Full API documentation and usage examples are available at [readthedocs](https://readthedocs.org/projects/craterpy/).

## Contributing

There are two major ways you can help improve craterpy:

### Bug Reporting and Feature Requests

You can report bugs or request new features on the [issues](https://github.com/cjtu/craterpy/issues) board.

### Contributing Directly

Want to fix a bug / implement a feature / fix some documentation? We welcome pull requests from all new contributors! You (yes you!) can help us make craterpy as good as it can be! See [CONTRIBUTING.rst](https://github.com/cjtu/craterpy/blob/trunk/CONTRIBUTING.rst) for details on how to get started - first time GitHub contributors welcome - and encouraged!

## Citing craterpy

Craterpy is [MIT Licenced](https://github.com/cjtu/craterpy/blob/master/LICENSE.txt) and is free to use with attribution. Citation information can be found [here](https://zenodo.org/badge/latestdoi/88457986).
