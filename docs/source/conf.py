# Configure docs with Sphinx, sphinx-autodoc, and MyST
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --

import os
import sys
import toml

# Allow sphinx-autodoc to access craterpy contents
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information --

def get_version():
    with open('../../pyproject.toml') as pyproj:
        f = pyproj.read()
    return toml.loads(f)['tool']['poetry']['version']

project = "craterpy"
copyright = "2021, Christian J. Tai Udovicic"
author = "Christian J. Tai Udovicic"

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration --

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
]

source_suffix = ['.rst', '.md']

# Preserve :members: order.
autodoc_member_order = "bysource"

# -- HTML configuration --

html_theme = "furo"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
