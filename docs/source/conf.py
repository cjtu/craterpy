# Configure docs with Sphinx, sphinx-autodoc, and MyST
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --

import importlib.metadata
import os
import sys

# Allow sphinx-autodoc to access craterpy contents
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information --


def get_version():
    return importlib.metadata.version("craterpy")


project = "craterpy"
copyright = "2025, Christian J. Tai Udovicic"
author = "Christian J. Tai Udovicic"

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration --

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

nb_execution_mode = "cache"
nb_execution_timeout = 300

myst_enable_extensions = [
    "colon_fence",
]

source_suffix = [".rst", ".md", ".ipynb"]

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
