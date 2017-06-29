# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:29:08 2017

@author: Christian
"""

from __future__ import absolute_import, division, print_function
from .version import __version__
from .acerim import *

if __name__ == '__main__':
    import gdal
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import classes as ac
    import functions as af
    import acestats as acs