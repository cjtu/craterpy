""" 
Supplies instructions to setuptools to build acerim package distributions. 
Contains the metadata that appears on PyPI and automatically fetches version 
number from version.py. Also automatically builds docs using sphinx. 
"""
from setuptools import setup, find_packages
from codecs import open
from sphinx.setup_command import BuildDoc
from os import path
here = path.abspath(path.dirname(__file__))


def get_version():
    """ Fetch version from /acerim/version.py """
    version = {}
    with open(path.join(here, 'acerim', 'version.py')) as f:
        exec(f.read(), version)
    return version['__version__']

def get_readme():
    """ Fetch long winded description from /README.rst """
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        return f.read()


# Set metadata
NAME = 'acerim'
VERSION = get_version()
DESCRIPTION = 'A package for analyzing impact crater ejecta.'
LONG_DESCRIPTION = get_readme()
AUTHOR = 'Christian Tai Udovicic'
AUTHOR_EMAIL = 'cj.taiudovicic@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = 'http://github.com/cjtu/acerim'
LICENSE = 'MIT'
CLASSIFIERS = ['Development Status :: 2 - Pre-Alpha',
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
    ]
KEYWORDS = 'planetary science crater ejecta data analysis'
PLATFORMS = 'OS Independent'
PACKAGES = find_packages(exclude=['tests']) 
PACKAGE_DATA = {'': ['*.csv', '*.tif']}
INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib', 'gdal', 'pandas']
PYTHON_REQUIRES = '>=2.7, <=3.3'
EXTRAS_REQUIRE = None # {'test': ['pytest'], 'cite': ['duecredit'],}

# Setup Sphinx integration to automatically build documentation 
CMDCLASS = {'build_docs': BuildDoc}
COMMAND_OPTIONS = { # optional and override docs/conf.py settings
        'build_docs': {
            'project': ('setup.py', NAME),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', VERSION),
            'build_dir': ('setup.py', './docs/_build')}}

# Run setup() function with metadata above
setup(
    name=NAME,
    version=VERSION, 
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,     
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,     
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=PYTHON_REQUIRES,
    cmdclass=CMDCLASS, 
    command_options=COMMAND_OPTIONS,
)
