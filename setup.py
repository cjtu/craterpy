"""Installation script.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))


def get_version():
    """ Fetch version from version.py """
    version = {}
    with open(path.join(here, 'craterpy', 'version.py')) as f:
        exec(f.read(), version)
    return version['__version__']


def get_readme():
    """ Fetch long winded description from /README.rst """
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        return f.read()


def try_BuildDoc():
    """Try to import BuildDoc (incompatible with py2)"""
    try:
        from sphinx.setup_command import BuildDoc
        return {'build_docs': BuildDoc}
    except ImportError:
        print("Warning: sphinx.setup_command unavailable. Docs not built")
        return {}


# Set package metadata
NAME = 'craterpy'
VERSION = get_version()
DESCRIPTION = 'A package for impact crater data science.'
LONG_DESCRIPTION = get_readme()
AUTHOR = 'Christian Tai Udovicic'
AUTHOR_EMAIL = 'cj.taiudovicic@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = 'http://github.com/cjtu/craterpy'
LICENSE = 'MIT'
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5']
KEYWORDS = 'python crater data analysis planetary science'
PLATFORMS = 'OS Independent'
PACKAGES = find_packages(exclude=['tests'])
PACKAGE_DATA = {'': ['*.csv', '*.tif']}
INSTALL_REQUIRES = ['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'gdal']
# PYTHON_REQUIRES = '>=2.7, <=3.3'
EXTRAS_REQUIRE = None  # {'test': ['pytest'], 'cite': ['duecredit'],}

# Setup Sphinx integration to automatically build documentation
CMDCLASS = try_BuildDoc()
COMMAND_OPTIONS = {  # Overrides docs/conf.py settings
        'build_docs': {
            'project': ('setup.py', NAME),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', VERSION),
            'source_dir': ('setup.py', './docs/source'),
            'build_dir': ('setup.py', './docs/build')}}

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
    # python_requires=PYTHON_REQUIRES,
    cmdclass=CMDCLASS,
    command_options=COMMAND_OPTIONS,
)
