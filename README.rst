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
    from craterpy import dataset, stats
    df = pd.DataFrame({'Name': ["Orientale", "Langrenus", "Compton"],
                       'Lat': [-19.9, -8.86, 55.9],
                       'Lon': [-94.7, 61.0, 104.0],
                       'Rad': [147.0, 66.0, 82.3]})
    moon = dataset.CraterpyDataset("moon.tif")
    stat_df = cs.ejecta_stats(df, moon, 4, ['mean', 'median', 'std'], plot=True)


.. image:: /craterpy/data/_images/readme_crater_ejecta.png

::

  stats_df.head()

.. image:: /craterpy/data/_images/readme_stat_df.png


New users should start with the IPython notebook `tutorial`_ for typical usage with examples.

**Note**: This package currently **only accepts image data in simple-cylindrical (Plate Caree) projection**. If your data is in another projection, please reproject it to simple-cylindrical before importing it with craterpy (check out `GDAL`_). If you would like add reprojection functionality to craterpy, consider `Contributing`_.

.. _`tutorial`: https://gist.github.com/cjtu/560f121049b342aa0b2bf70e038358b7
.. _`GDAL`: http://www.gdal.org/
.. _`contributing`: https://github.com/cjtu/craterpy/blob/master/CONTRIBUTING.rst


Requires
--------
craterpy currently supports the following python versions on Linux, OS X and Windows:

- 2.7
- 3.4
- 3.5

It's core dependencies are:

- pandas
- gdal
- numpy
- scipy
- matplotlib

We recommend using the `Anaconda`_ package manager to install craterpy. Anaconda automatically resolves dependency conflicts and allows you to get virtual environments working quickly.

.. _`Anaconda`: https://www.anaconda.com/distribution/

Quick Installation with Anaconda
--------------------------------

1. `Install Anaconda <https://www.anaconda.com/download/>`_.

2. Open a terminal window and create a conda virtual environment (name it anything you like, and set the python version to a compatible version in `Requires`_)::

    conda create --name your_env_name python=3.5

3. Activate the environment (Windows users can omit "source")::

    source activate your_env_name

4. Now install the dependencies::

    conda install numpy scipy pandas matplotlib gdal=2.1.0 libgdal=2.1.0

5. Install craterpy with pip::

    pip install craterpy

6. Check your installation with ``conda list``.

Now that you have craterpy installed, head over to the `tutorial`_ to get started!

**Note**: Remember to activate your virtual environment each time you use craterpy.

.. _`Managing Environments`: https://conda.io/docs/using/envs

Installing from a fork
^^^^^^^^^^^^^^^^^^^^^^

1. Fork this project from `craterpy on GitHub`_.
2. Clone your fork locally
3. Navigate to the craterpy root directory and install with::

    python setup.py install

**Warning**: This installs the newest craterpy updates which may not be production stable. Installing from pip automatically pulls the previous stable release.

.. _`craterpy on GitHub`: https://github.com/cjtu/craterpy

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

Development Environment
"""""""""""""""""""""""
The suggested development environment is specified in `.environment.yml`. It can be built automatically in a new conda environment in a few simple steps:

1. Fork the `craterpy on GitHub`_.
2. Create the ``craterpy-dev`` environment with::

    conda env create -f .environment.yml

3. Activate the dev environment with (ignore "source" on Windows)::

    source activate craterpy-env

4. Test the environment with ``conda list``, then hack away!

The dev environment comes pre-installed with craterpy and all of its dependencies, as well as some handy libraries like ``pytest``, ``pytest-cov``, and ``flake8``.

Updating .environment.yml
"""""""""""""""""""""""""
A new ``.environment.yml`` can be generated from within the activated craterpy-dev environment with::

   conda env export > .environment.yml


Citing craterpy
---------------

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

