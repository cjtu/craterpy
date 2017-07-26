""" Setuptools instructions for building acerim package and its metadata."""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get version from /acerim/version.py
version = {}
with open(path.join(here, 'acerim','version.py')) as f:
    exec(f.read(), version)
    VERSION = version['__version__']
# Get long description from /README.rst
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
# Get packages
PACKAGES = find_packages(exclude=['tests'])

setup(
    name='acerim',
    version=VERSION, # From /acerim/version.py
    description='A package for analyzing impact crater ejecta',
    maintainer='Christian Tai Udovicic',
    maintainer_email='cj.taiudovicic@gmail.com',     
    url='http://github.com/cjtu/acerim',
    license='MIT',
    classifiers=['Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3'
    ],
    keywords='planetary science crater ejecta data analysis',
    author='Christian Tai Udovicic',
    author_email='cj.taiudovicic@gmail.com',
    platforms='OS Independent',     
    packages=PACKAGES,
    package_data={'': ['*.csv', '*.tif']},
    install_requires=['numpy', 'scipy', 'matplotlib', 'gdal', 'pandas'],
    #extras_require={#'test': ['pytest'],
        #'cite': ['duecredit'],},
    python_requires='>=2.7, <=3.3',
    long_description=LONG_DESCRIPTION
)
