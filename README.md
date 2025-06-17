<h1 align="center">
  <strong>Craterpy:</strong><em> Impact crater data science in Python.</em>
</h1>

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
  <!-- Code of Conduct -->
  <a href="CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"
      alt="Contributor Covenant" />
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

Note: Craterpy is not a crater detection algorithm (e.g. [PyCDA](https://github.com/AlliedToasters/PyCDA)), nor is it a crater count age dating tool (see [craterstats](https://github.com/ggmichael/craterstats)).

**Note:** *Craterpy is in development. We appreciate bug reports and feature requests on the [issues board](https://github.com/cjtu/craterpy/issues).*


## Quickstart

Install with `pip install craterpy` then follow the full worked example in the docs [Getting Started](https://craterpy.readthedocs.io/en/latest/getting_started.html).

## Demo

Quickly import tabluar crater data from a CSV and visualize it on a geotiff in 2 lines of code:

```python
from craterpy import CraterDatabase, sample_data

cdb = CraterDatabase(sample_data['vesta_craters.csv'], 'Vesta', units='m')
cdb.plot(sample_data['vesta.tif'], alpha=0.5, color='tab:green')
```

![Vesta map plot](https://github.com/cjtu/craterpy/raw/trunk/craterpy/data/_images/readme_vesta_cdb.png)

Clip and plot targeted regions around each crater from large raster datasets.

```python
cdb.add_circles('crater_rois', 3)
cdb.plot_rois(sample_data['vesta.tif'], 'crater_rois')
cdb.plot_rois(sample_data['vesta.tif'], 'crater_rois', range(1500, 1503))
```

![Vesta plot rois](https://github.com/cjtu/craterpy/raw/trunk/craterpy/data/_images/readme_vesta_rois.png)

Extract zonal statistics for crater regions of interest.

```python
# Import lunar crater and define the floor and rim
cdb = CraterDatabase(sample_data['moon_craters.csv'], 'Moon', units='km')
cdb.add_annuli("floor", 0.4, 0.8)  # Crater floor (exclude central peak and rim)
cdb.add_annuli("rim", 0.9, 1.1)  # Thin annulus at crater rim

# Compute summary statistics for every ROI see docs for supported stats
stats = cdb.get_stats(sample_data['moon_dem.tif'], regions=['floor', 'rim'], stats=['median'])

# Compute crater depth as rim elevation - floor elevation
stats['depth (m)'] = (stats.median_rim - stats.median_floor)
print(stats.head(3).round(2))

# Name     Rad      Lat     Lon     median_floor  median_rim  depth (m)
# Olbers D  50.015  10.23  -78.03      -1452.50    -1322.88     129.62
# Schuster   50.04   4.44  146.42        445.58     1976.97    1531.39
# Gilbert  50.125  -3.20   76.16      -2213.66     -731.64    1482.02
```

## Documentation

Full API documentation and usage examples are available at [readthedocs](https://readthedocs.org/projects/craterpy/).


## Installation

We recommend pip installing craterpy into a virtual environment, e.g. with `conda` or `venv`:

```bash
pip install craterpy
```
- **Note**: Craterpy is currently only tested on Ubuntu and OS X but may work on some versions of Windows. 

## Contributing

There are two major ways you can help improve craterpy:

- Report bugs or request new features on the [issues](https://github.com/cjtu/craterpy/issues) board.

- Contributing directly. See [CONTRIBUTING.rst](https://github.com/cjtu/craterpy/blob/trunk/CONTRIBUTING.rst) for full details. First time open source contributors are welcome!

## Citing craterpy

Craterpy is [MIT Licenced](https://github.com/cjtu/craterpy/blob/master/LICENSE.txt) and is free to use with attribution. Citation information can be found [here](https://zenodo.org/badge/latestdoi/88457986).
