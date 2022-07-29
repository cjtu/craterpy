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

Craterpy simplifies the extraction and statistical analysis of impact craters in planetary datasets. It can:

- work with tables of crater data in Python (using pandas)
- load and manipulate planetary image data in Python (using rasterio)
- extract, mask, filter, and compute stats on craters located in planetary imagery
- plot crater regions of interest

Craterpy currently only supports simple cylindrical images and requires you to provide a table of crater locations and sizes (e.g. it isn't a crater detection program). See the example below!

**Note:** *Craterpy is in alpha. We appreciate bug reports and feature requests on the [issues board](https://github.com/cjtu/craterpy/issues).*

## Example

Craterpy in action:

```python
import pandas as pd
from craterpy import dataset, stats
df = pd.DataFrame({'Name': ["Orientale", "Langrenus", "Compton"],
                    'Lat': [-19.9, -8.86, 55.9],
                    'Lon': [-94.7, 61.0, 104.0],
                    'Rad': [147.0, 66.0, 82.3]})
moon = dataset.CraterpyDataset("moon.tif")
stat_df = cs.ejecta_stats(df, moon, 4, ['mean', 'median', 'std'], plot=True)
```

![ejecta image](https://raw.githubusercontent.com/cjtu/craterpy/trunk/craterpy/data/_images/readme_crater_ejecta.png)

```python
stats_df.head()
```

![crater stats](https://raw.githubusercontent.com/cjtu/craterpy/trunk/craterpy/data/_images/readme_stat_df.png)

New users should start with the Jupyter notebook [tutorial](https://gist.github.com/cjtu/560f121049b342aa0b2bf70e038358b7) for typical usage with examples. See also [craterpy documentation](https://readthedocs.org/projects/craterpy/) on Read the Docs.

**Note**: This package currently **only accepts image data in simple-cylindrical (Plate Caree) projection**. If your data is in another projection, please reproject it to simple-cylindrical before importing it with craterpy. If you would like add reprojection functionality to craterpy, consider [Contributing](https://github.com/cjtu/craterpy/blob/trunk/CONTRIBUTING.rst).

## Installation

With pip:

```bash
pip install craterpy
python -c "import craterpy; print(craterpy.__version__)"
```

In a new [conda environment](https://conda.io/docs/using/envs):

```bash
# Create and activate a new conda environment called "craterpy"
conda create -n craterpy python=3.9
conda activate craterpy

# Install craterpy with pip
pip install craterpy
python -c "import craterpy; print(craterpy.__version__)"
```

With [git](https://git-scm.com) and [poetry](https://python-poetry.org/docs/) (for latest version & development):

```bash
# Clone this repository
$ cd ~
$ git clone https://github.com/cjtu/craterpy.git

# Enter the repository
$ cd craterpy

# Configure poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# Install craterpy with poetry
$ poetry install

# Check installation
poetry version

# Either open a Jupyter server
$ poetry run jupyter notebook

# Or activate the venv from your Python editor of choice
# The venv is path is ~/craterpy/.venv/bin/python
```

On Windows (see [rasterio installation for Windows](https://rasterio.readthedocs.io/en/latest/installation.html#windows)):

- **Note**: Craterpy is tested on Ubuntu and OS X. If you would like to use craterpy on Windows, we recommend getting the Windows Subsystem for Linux ([WSL](https://docs.microsoft.com/en-us/windows/wsl/install)) and installing into a Linux distribution. However, the following may also work for a native Windows installation and depends on a working installation of rasterio from pre-compiled binaries (see link above).

```bash
# Windows requires gdal binaries specific to the OS (32/64-bit) and python version
# First download the rasterio and GDAL binaries for your system (link above)
# If rasterio imports with no error then craterpy should be pip installable
pip install GDAL-X.Y.Z-...-win.whl
pip install rasterio-X.Y.Z-...-win.whl
python -c "import rasterio"
pip install craterpy
python -c "import craterpy; print(craterpy.__version__)"
```

Trouble installing craterpy? Let us know on the [issues](https://github.com/cjtu/craterpy/issues) board.

## Dependencies

Craterpy requires python >3.7.7 and is tested on Ubuntu and OS X. It's core dependencies are:

- rasterio
- pandas
- numpy
- matplotlib

## Documentation

Full API documentation is available at [readthedocs](https://readthedocs.org/projects/craterpy/).

## Contributing

There are two major ways you can help improve craterpy:

### Bug Reporting and Feature Requests

You can report bugs or request new features on the [issues](https://github.com/cjtu/craterpy/issues) board.

### Contributing Directly

Want to fix a bug / implement a feature / fix some documentation? We welcome pull requests from all new contributors! You (yes you!) can help us make craterpy as good as it can be! See [CONTRIBUTING.rst](https://github.com/cjtu/craterpy/blob/trunk/CONTRIBUTING.rst) for details on how to get started - first time GitHub contributors welcome - and encouraged!

## Citing craterpy

Craterpy is [MIT Licenced](https://github.com/cjtu/craterpy/blob/master/LICENSE.txt) and is free to use with attribution. Citation information can be found [here](https://zenodo.org/badge/latestdoi/88457986).

## Contact

If you have comments/question/concerns or just want to get in touch, you can email Christian at cj.taiudovicic@gmail.com or follow [@TaiUdovicic](https://twitter.com/TaiUdovicic) on Twitter.
