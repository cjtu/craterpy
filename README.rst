craterpy |TravisBadge|_ |AppveyorBadge|_ |RtdBadge|_ |PyPiBadge|_ |CodecovBadge|_ |ZenodoBadge|_
==============================================================================================
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

Welcome to **craterpy** (formerly ACERIM), your one-stop shop to crater data science in Python!

Please note: this package is in the alpha stage of development. You can direct any questions to Christian at cj.taiudovicic@gmail.com. Feature requests and bugs tracking will be hosted on the GitHub `bug tracker <craterpy/bugtacker>`_.

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

..image:: get_roi.png

cp.compute_stats(ds, df, rois).head()

..image:: compute_stats.png

cp.ejecta_profiles(ds, df, rois, spacing=0.1)

..image:: ejecta_profiles.png

New users should start with the IPython notebook at `tutorial <https://nbviewer.jupyter.org/github/cjtu/craterpy/blob/master/craterpy/sample/tutorial.ipynb>`_ for typical usage with examples.

**Note**: While craterpy is a tool used to query image data, it does reproject it. This package **only acepts image data in simple-cylindrical (Plate Caree) projection**. Many tools exist to reproject your data. To do so in Python, we suggest checking out `GDAL <http://www.gdal.org/>`_.


Dependencies
------------

craterpy supports python versions 2.7, 3.3 and 3.4. It depends on:

  - numpy
  - scipy
  - pandas
  - matplotlib
  - gdal=2.1.0


Quick Installation with Anaconda
--------------------------------

We reccommend the `Anaconda <https://www.continuum.io/Anaconda-Overview>`_ package manager. See `Continuum Analytics <https://www.continuum.io/downloads>`_ for installation instructions.

The following section will describe how to create and activate a conda virtual environment to run ACERIM. For more on Anaconda virtual environments, see `Managing Environments <https://conda.io/docs/using/envs>`_. 

With *anaconda and gdal installed*, open a terminal window and create a new conda environment with the following command (replace **your_env_name** and choose desired python version):: 

  conda create --name your_env_name python=3.5

Activate the environment (OS X, Unix, or powershell users may need to use *source*)::

  (source) activate your_env_name

Now install the dependencies. Ensure to specify the gdal and libgdal versions to avoid a known bug being tracked `here <https://github.com/ContinuumIO/anaconda-issues/issues/1687>`_::

  conda install numpy scipy pandas matplotlib gdal=2.1.0 libgdal=2.1.0

Install the latest craterpy release with pip::

  pip install craterpy

Now that you have craterpy installed, head over to the `tutorial <https://nbviewer.jupyter.org/github/cjtu/craterpy/blob/master/craterpy/sample/tutorial.ipynb>`_ to get started!

**Note**: Remember to activate your virtual environment each time you use ACERIM.


Forking this repository
^^^^^^^^^^^^^^^^^^^^^^^

You can fork craterpy from `GitHub <https://github.com/cjtu/>`_. You can then clone your forked version, navigate to the root directory and install craterpy with:

::

  python setup.py install



Documentation
-------------

Full documentation is available at `readthedocs <https://readthedocs.org/projects/craterpy/>`_.


Support and Bug Reporting
-------------------------

Bugs will be tracked at `craterpy bug tracker <craterpy/bugtacker>`_. General questions can be directed to Christian at cj.taiudovicic@gmail.com.


Citing ACERIM
-------------

For convenience, this project uses the OSI-certified MIT open access liscence for warranty-free ease of use and distribution. The author simply asks that you cite the project. The citable DOI can be found at Zenodo by clicking the badge below. To read more about citable code, check out `Zenodo <http://help.zenodo.org/features>`_.

.. image:: https://zenodo.org/badge/88457986.svg
    :target: https://zenodo.org/badge/latestdoi/88457986


Contributing
------------

craterpy is seeking contributers of all skill levels! Please read CONTRIBUTING.rst if you are interested in supporting craterpy. Feel free to check the bug tracker for open issues or get in touch with Christian at cj.taiudovicic@gmail.com if you have any questions.


License
-------

Copyright (c) 2017- Christian Tai Udovicic. Released under the MIT license. This software comes with no warranties. See LICENSE.txt for details.
