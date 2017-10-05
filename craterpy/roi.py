"""Contains the CraterRoi 2D array object"""
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import gdal
from acerim import acefunctions as af

class CraterRoi(np.ndarray):
	"""The CraterRoi is a window of image data with geographic information.

	The CraterRoi is a numpy 2Darray of data centered on a crater. It contains
	the geographic information of the crater and the bounds of the window to 
	simplify processing and plotting.

	CraterRoi inherits all attributes and methods from numpy.

	Attributes
    ----------
    lat, lon, radius : int or float
    	Center latitude and longitude of the crater and this roi [degrees]. 
    radius : int or float
        Radius of the crater [km].
    wsize : int or float
    	Size of window around crater [crater radii].
    extent : list of float
        North, south, west, and east bounds of CraterRoi [degrees].

    Methods
    -------
    filter(vmin, vmax, strict=False, fillvalue=np.nan)
		Replaces values outside the range (vmin, vmax) with fillvalue.
	mask(type, outside=False, fillvalue=np.nan)
		Applies mask of fillvalue according to type.
    plot(*args, **kwargs)
        Plot this CraterRoi. See plotting.plot_CraterRoi()

    See Also
    --------
    numpy.ndarray
    
    Examples
    --------
    >>> import os.path as p
    >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
    >>> dsfile = p.join(datadir, 'moon.tif')
    >>> ds = CpyDataset(dsfile, radius=1737)
    >>> roi = ds.get_roi(-27.2, 80.9, 207)  # Humboldt crater
    """
    def __init__(self, lat, lon, rad, wsize=2, plot=False):
    	pass

    def __repr__(self):
    	pass


    def filter(self, vmin=float('-inf'), vmax=float('inf'), strict=False, 
               fillvalue=np.nan):
    	"""Replaces values outside the range (vmin, vmax) with fillvalue.
		
		Parameters
	    ----------
	    vmin : int or float
	    	Minimum value to keep (default -inf, no lower bound).
	    vmax : int or float
	    	Maximum value to keep (default inf, no upper bound).
		strict : bool
			Keep values strictly greater and strictly less than vmin, vmax
			(default False).
		fillvalue : int or float
			Number to fill in filtered values (default np.nan).
    	"""
    	roi = self.copy()
	    nanmask = ~np.isnan(roi)  # build nanmask with pre-existing nans, if any
	    if not strict:
	        nanmask[nanmask] &= roi[nanmask] > vmax  # Add values > vmax to nanmask
	        nanmask[nanmask] &= roi[nanmask] < vmin  # Add values < vmin to nanmask
	    else:  # if strict, exclude values equal to vmin, vmax
	        nanmask[nanmask] &= roi[nanmask] >= vmax
	        nanmask[nanmask] &= roi[nanmask] <= vmin
	    roi[nanmask] = fillvalue
	    self = roi

    def mask(self, mask, outside=False, fillvalue=np.nan):
    	"""Fill pixels in bool mask from masking.py with fillvalue.

		Parameters
	    ----------
	    mask : 2D array
	    	Mask of this roi from masking.py
	    outside: bool
	    	Mask outside the area in mask (default False).
		fillvalue : int or float
			Number to fill in masked values (default np.nan).

		Examples
		--------

    	"""
    	pass

    def plot(self, *args, **kwargs):
    	"""Plot this CraterRoi. See plotting.plot_CraterRoi()"""
    	craterpy.plotting.plot_CraterRoi(self, *args, **kwargs)
