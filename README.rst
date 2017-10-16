craterpy |TravisBadge|_ |AppveyorBadge|_ |RtdBadge|_ |PyPiBadge|_ |CodecovBadge|_ |ZenodoBadge|_
================================================================================================
.. |ZenodoBadge| image:: https://zenodo.org/badge/88457986.svg
.. _ZenodoBadge: https://zenodo.org/badge/latestdoi/88457986

.. |TravisBadge| image:: https://travis-ci.org/cjtu/craterpy.svg?branch=master
.. _TravisBadge: https://travis-ci.org/cjtu/craterpy

.. |AppveyorBadge| image:: https://ci.appveyor.com/api/projects/status/kns2v4vn07r6h078?svg=true
.. _AppveyorBadge: https://ci.appveyor.com/project/cjtu/craterpy/branch/master

.. |RtdBadge| image:: http://readthedocs.org/projects/craterpy/badge/?version=latest
.. _RtdBadge: http://craterpy.readthedocs.io/en/latest/?badge=latest

.. |PyPiBadge| image:: https://badge.fury.io/py/craterpy.svg
.. _PyPiBadge: https://badge.fury.io/py/craterpy

.. |CodecovBadge| image:: https://codecov.io/gh/cjtu/craterpy/branch/master/graph/badge.svg
.. _CodecovBadge: https://codecov.io/gh/cjtu/craterpy


Overview
--------

Welcome to craterpy (formerly *ACERIM*), your one-stop shop to crater data science in Python!

This package is in the alpha stage of development. You can direct any questions to Christian at cj.taiudovicic@gmail.com. Bug reports and feature requests can be opened as issues at the `issue tracker`_ on GitHub.

You can use craterpy to:

  - work with tables of crater data in Python (using pandas),
  - load and manipulate image data in Python (using gdal),
  - easily extract, mask, filter, plot, and compute stats on craters located in your images.

.. `issue tracker`_: https://github.com/cjtu/craterpy/issues

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

**Note**: While craterpy read in image data, it does not reproject it. This package currently **only accepts image data in simple-cylindrical (Plate Caree) projection**. If your data is in another projection, please reproject it to simple-cylindrical before importing it with craterpy (check out `GDAL <http://www.gdal.org/>`_). If you would like add reprojection functionality, consider `contributing`_.

.. _`contributing`: https://github.com/cjtu/craterpy/blob/master/CONTRIBUTING.rst


Dependencies
------------

craterpy supports python versions 2.7, 3.4 and 3.5. It explicitly requires:

  - numpy
  - scipy
  - pandas
  - matplotlib
  - gdal=2.1.0


Quick Installation with Anaconda
--------------------------------

We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ package manager to work with craterpy. 

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

API documentation is available at `readthedocs <https://readthedocs.org/projects/craterpy/>`_.


Contributing
------------
There are two major ways you can help improve craterpy:

Bug Reporting and Feature Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can report bugs or request new features on the `issue tracker <https://github.com/cjtu/craterpy/issues>`_. If you are reporting a bug, please give a detailed description about how it came up and what your build environment is (e.g. with ``conda list``). 

Becoming a contributor
^^^^^^^^^^^^^^^^^^^^^^
We are looking for new contributors! If you are interested in open source and want to join a supportive learning environment - or if you want to use craterpy and make it better for everyone - consider contributing to the project. See `contributing`_ for details on how to get started!


Citing ACERIM
-------------

For convenience, this project uses the `MIT Licence <https://github.com/cjtu/craterpy/blob/master/LICENSE.txt>`_ for warranty-free ease of use and distribution. The author simply asks that you cite the project when using it in published research. The `citable DOI <https://zenodo.org/badge/latestdoi/88457986>`_ can be found at Zenodo by clicking the badge below.

.. image:: https://zenodo.org/badge/88457986.svg
    :target: https://zenodo.org/badge/latestdoi/88457986

To read more about citable code, check out `Zenodo <http://help.zenodo.org/features>`_.


Contact
-------
If you have comments/question/concerns or just want to get in touch, you can email Christian at cj.taiudovicic@gmail.com or follow `@TaiUdovicic <https://twitter.com/TaiUdovicic>`_ on Twitter.


License
-------

Copyright (c) 2017- Christian Tai Udovicic. Released under the MIT license. This software comes with no warranties. See `LICENSE <https://github.com/cjtu/craterpy/blob/master/LICENSE.txt>`_ for details.
 