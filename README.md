<h1 align="center">
  <strong>Craterpy:</strong><em> Impact crater data science in Python</em>
</h1>

<div align="center">
  <!-- PYPI version -->
  <a href="https://pypi.org/project/craterpy">
    <img alt="PyPI badge" src="https://img.shields.io/pypi/v/craterpy">
  </a>
  <!-- Tests (GitHub Actions CI) -->
  <a href="https://github.com/cjtu/craterpy/actions/workflows/test.yml">
    <img src="https://github.com/cjtu/craterpy/actions/workflows/test.yml/badge.svg"
      alt="Tests badge" />
  </a>
  <!-- Test Coverage (codecov) -->
  <a href="https://codecov.io/gh/cjtu/craterpy">
    <img src="https://codecov.io/gh/cjtu/craterpy/branch/main/graph/badge.svg?token=9K567x0YUJ"
      alt="Codecov badge" />
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
      alt="Ruff badge" />
  </a>
</div>
<div align="center">
  <!-- ReadTheDocs -->
  <a href="http://craterpy.readthedocs.io/latest/?badge=latest">
    <img src="http://readthedocs.org/projects/craterpy/badge/?version=latest"
      alt="Docs badge" />
  </a>
  <!-- Code of Conduct -->
  <a href="CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"
      alt="Contributor Covenant" />
      </a>
</div>
<div align="center">
  <!-- JOSS badge -->
  <a style="border-width:0" href="https://doi.org/10.21105/joss.08663">
    <img src="https://joss.theoj.org/papers/10.21105/joss.08663/status.svg" alt="DOI badge" >
  </a>
</div>

# Overview

Craterpy makes it easier to work with impact crater data in Python. Highlights:

- convert a table of crater coordinates and sizes to a GeoDataFrame or GIS-ready shapefile
- extract zonal statistics associated with each crater in circlular or annular regions (with [rasterstats](https://pythonhosted.org/rasterstats/))
- eliminate some pain points of planetary GIS analysis (antimeridian wrapping, projection conversions, etc.)
- supports all roughly spherical cratered bodies ([examples](https://craterpy.readthedocs.io/latest/planetary_body_examples.html))

Note: Craterpy is not a crater detection algorithm (e.g. [PyCDA](https://github.com/AlliedToasters/PyCDA)), nor is it a crater count age dating tool (see [craterstats](https://github.com/ggmichael/craterstats)).

**Note:** *Craterpy is in development. We appreciate bug reports and feature requests on the [issues board](https://github.com/cjtu/craterpy/issues).*


## Quickstart

Install with `pip install craterpy` then see example usage at [Getting Started](https://craterpy.readthedocs.io/latest/getting_started.html).

## Demo

Quickly import tabluar crater data from a CSV and visualize it on a geotiff in 2 lines of code:

```python
from craterpy import CraterDatabase, sample_data as sd

cdb = CraterDatabase(sd['vesta_craters_km.csv'], 'Vesta', units='km')
cdb.plot(sd['vesta.tif'], alpha=0.5, color='tab:green', savefig='readme_vesta_cdb.png')
```

![Vesta map plot](https://github.com/cjtu/craterpy/raw/main/docs/_images/readme_vesta_cdb.png)

Clip and plot targeted regions around each crater from large raster datasets.

```python
cdb.add_circles('crater_roi', 1.5)
cdb.plot_rois(sd['vesta.tif'], 'crater_roi', range(3, 12))
```

![Vesta plot rois](https://github.com/cjtu/craterpy/raw/main/docs/_images/readme_vesta_rois.png)

Extract zonal statistics for crater regions of interest.

```python
import pandas as pd
from craterpy import CraterDatabase, sample_data as sd
df = df = pd.read_csv(sd["moon_craters_km.csv"])
cdb = CraterDatabase(df[df["Diameter (km)"] < 60], "Moon", units="km")

# Define regions for crater floor, rim (sizes in crater radii)
cdb.add_annuli("floor", 0.4, 0.6)  # crater floor, excluding possible central peak
cdb.add_annuli("rim", 0.99, 1.01)  # thin annulus at rim

# Pull statistics from a Lunar Digital Elevation Model (DEM) GeoTiff
stats = cdb.get_stats(sd["moon_dem.tif"], regions=['floor', 'rim'], stats=['mean'])

# Use mean elevations to compute depth (rim to floor)
stats['crater_depth (m)'] = (stats.mean_rim - stats.mean_floor)
print(stats.head().to_string(float_format='%.1f', index=False))

#  Diameter (km)  Latitude  Longitude  mean_floor  mean_rim  crater_depth (m)
#           60.0      19.4     -146.5      6070.0   10792.9            4722.9
#           60.0      44.2      145.3      -976.4    3114.0            4090.4
#           60.0     -43.6       -7.5     -3617.5     186.8            3804.4
#           60.0      -9.6      134.7      1843.4    6127.9            4284.4
#           59.9     -25.3        2.4     -2634.2    -945.0            1689.1
```

## Cite This Repository

If you use this project in your research, please cite the [JOSS paper](https://doi.org/10.21105/joss.08663) as below:

> Tai Udovicic et al., (2025). Craterpy: Impact crater data science in Python. Journal of Open Source Software, 10(113), 8663, https://doi.org/10.21105/joss.08663

```bibtex
@article{craterpy2025, 
doi = {10.21105/joss.08663},
author = {Tai Udovicic, Christian J. and Essunfeld, Ari and Costello, Emily S.},
title = {Craterpy: Impact crater data science in Python},
journal = {Journal of Open Source Software},
url = {https://doi.org/10.21105/joss.08663},
year = {2025},
publisher = {The Open Journal},
volume = {10},
number = {113},
pages = {8663}}
```

## Documentation

Full API documentation and usage examples are available at [ReadTheDocs](https://craterpy.readthedocs.io/).


## Installation

We recommend pip installing craterpy into a virtual environment, e.g. with `conda` or `venv`:

```bash
pip install craterpy
```
- **Note**: Craterpy is tested on latest long-term support versions of Windows, OS X and Ubuntu, and Python version 3.10 and up.

## Contributing

There are two major ways you can help improve craterpy:

- Report bugs or request new features on the [issues](https://github.com/cjtu/craterpy/issues) board.

- Contributing directly. See [CONTRIBUTING.rst](https://github.com/cjtu/craterpy/blob/main/CONTRIBUTING.rst) for full details. First time open source contributors are welcome!
