craterpy |TravisBadge|_ |AppveyorBadge|_ |RtdBadge|_ |PyPiBadge|_ |CodecovBadge|_ |ZenodoBadge|_
================================================================================================
.. |ZenodoBadge| image:: https://zenodo.org/badge/88457986.svg
.. _ZenodoBadge: https://zenodo.org/badge/latestdoi/88457986

.. |TravisBadge| image:: https://travis-ci.org/cjtu/craterpy.svg?branch=master
.. _TravisBadge: https://travis-ci.org/cjtu/craterpy

.. |AppveyorBadge| image:: https://ci.appveyor.com/api/projects/status/7r7f4lbj6kgguhtw/branch/master?svg=true
.. _AppveyorBadge: https://ci.appveyor.com/project/cjtu/craterpy/branch/master

.. |RtdBadge| image:: http://readthedocs.org/projects/craterpy/badge/?version=latest
.. _RtdBadge: http://craterpy.readthedocs.io/en/latest/?badge=latest

.. |PyPiBadge| image:: https://badge.fury.io/py/craterpy.svg
.. _PyPiBadge: https://badge.fury.io/py/craterpy

.. |CodecovBadge| image:: https://codecov.io/gh/cjtu/craterpy/branch/master/graph/badge.svg
.. _CodecovBadge: https://codecov.io/gh/cjtu/craterpy


Overview
--------

Welcome to **craterpy** (formerly *ACERIM*), your one-stop shop to crater data science in Python!

This package is in the alpha stage of development. You can direct any questions to Christian at cj.taiudovicic@gmail.com. Bug reports and feature requests can be opened as `issues <https://github.com/cjtu/craterpy/issues>`_ on GitHub.

You can use craterpy to:

  - import tabular crater data into DataFrames (extends pandas),
  - load image data into efficient Dataset objects (extends gdal),
  - easily extract, mask, filter, plot, and compute stats on crater image data.


Example
-------
A code-snippet and plot is worth a thousand words::

  import pandas as pd
  import craterpy as cp
  df = pd.DataFrame("craters.csv", index)
  ds = cp.Dataset("moon.tif")
  rois = cp.get_roi(ds, df["Crisium"], plot_roi=True)

*Images coming soon*

New users should start with the IPython notebook `tutorial <https://nbviewer.jupyter.org/github/cjtu/craterpy/blob/master/craterpy/sample/tutorial.ipynb>`_ for typical usage with examples.

**Note**: While craterpy is a tool used to query image data, it does reproject it. This package currently **only accepts image data in simple-cylindrical (Plate Caree) projection**. To reproject your images in Python, we suggest checking out `GDAL <http://www.gdal.org/>`_.


Dependencies
------------

craterpy supports python versions 2.7, 3.4 and 3.5. It depends on:

  - numpy
  - scipy
  - pandas
  - matplotlib
  - gdal=2.1.0


Quick Installation with Anaconda
--------------------------------

We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ package manager for data science in Python. 

First `install Anaconda <https://www.anaconda.com/download/>`_.

Then create a conda virtual environment (See `Managing Environments <https://conda.io/docs/using/envs>`_ for more info). Replace your_env_name, and set python equal to a compatible version listed in `Dependencies`_

:: 

  conda create --name your_env_name python=3.5

Activate the environment (OS X, Unix, or powershell users may need to use *source*)::

  (source) activate your_env_name

Now install the dependencies. Ensure to specify the gdal and libgdal versions to avoid a known bug being tracked `here <https://github.com/ContinuumIO/anaconda-issues/issues/1687>`_::

  conda install numpy scipy pandas matplotlib gdal=2.1.0 libgdal=2.1.0

Install the latest craterpy release with pip::

  pip install craterpy

Now that you have craterpy installed, head over to the `tutorial <https://nbviewer.jupyter.org/github/cjtu/craterpy/blob/master/craterpy/sample/tutorial.ipynb>`_ to get started!

**Note**: Remember to activate your virtual environment each time you use craterpy.


Forking this repository
^^^^^^^^^^^^^^^^^^^^^^^

You can fork craterpy from `GitHub <https://github.com/cjtu/>`_. You can then clone your fork locally, navigate to the root directory and install craterpy with:

::

  python setup.py install

**Warning**: This method installs the latest version of craterpy which may not be production stable. Installing from pip guarantees the previous stable release.

Documentation
-------------

Full documentation is available at `readthedocs <https://readthedocs.org/projects/craterpy/>`_.


Support
-------

Issues are tracked on GitHub at `craterpy issues <https://github.com/cjtu/craterpy/issues>`_. Feel free to open an issue about a bug, feature request, or general question about using craterpy! This will help guide the development process. All other inquiries can be directed to Christian at cj.taiudovicic@gmail.com.


Citing ACERIM
-------------

For convenience, this project uses the `MIT Liscence <https://github.com/cjtu/craterpy/blob/master/LICENSE.txt>`_ for warranty-free ease of use and distribution. The author simply asks that you cite the project. The `citable DOI <https://zenodo.org/badge/latestdoi/88457986>`_ can be found at Zenodo by clicking the badge below.

.. image:: https://zenodo.org/badge/88457986.svg
    :target: https://zenodo.org/badge/latestdoi/88457986

To read more about citable code, check out `Zenodo <http://help.zenodo.org/features>`_.


Contributing
------------

We are seeking contributers of all skill levels! If you are interested, please start by reading CONTRIBUTING.rst. Then you can check out the `issue tracker <https://github.com/cjtu/craterpy/issues>`_ for open issues or get in touch with Christian at cj.taiudovicic@gmail.com if you have questions about how to get started.


License
-------

Copyright (c) 2017- Christian Tai Udovicic. Released under the MIT license. This software comes with no warranties. See `LICENSE <https://github.com/cjtu/craterpy/blob/master/LICENSE.txt>`_ for details.
 