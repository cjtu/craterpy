"""Contains the CraterRoi object"""
from __future__ import division, print_function, absolute_import
import numpy as np
import craterpy.helper as ch


class CraterRoi:
    """The CraterRoi contains a region of interest of image data for a crater.

    The CraterRoi reads in a 2D numpy array of data from the CraterpyDataset
    cds and stores it in the roi attribute. The roi is centered on lat, lon of
    the specified crater, and extends to rad*wsize from the center.

    Attributes
    ----------
    cds : CraterpyDataset
        CraterDataset to read roi from.
    lat, lon, radius : int or float
        Center latitude and longitude of the crater and this roi [degrees].
    radius : int or float
        Radius of the crater [km].
    wsize : int or float
        Size of window around crater [crater radii].
    roi : numpy.ndarray
        2D numpy array region of interest centered on crater.
    extent : list of float
        North lat, south lat, west lon, and east lon bounds of roi [degrees].

    Methods
    -------
    filter(vmin, vmax, strict=False, fillvalue=np.nan)
        Replaces values outside the range (vmin, vmax) with fillvalue.
    mask(type, outside=False, fillvalue=np.nan)
        Applies craterpy.masking mask of fillvalue.
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
    >>> cds = CraterpyDataset(dsfile, radius=1737)
    >>> croi = CraterRoi(cds, -27.2, 80.9, 207)  # Humboldt crater
    """
    def __init__(self, cds, lat, lon, rad, wsize=2, plot=False):
        self.cds = cds
        self.lat = lat
        self.lon = lon
        self.rad = rad
        self.wsize = wsize
        self.extent = self._get_extent()
        self.roi = self._get_roi()
        if plot:
            self.plot()

    def __repr__(self):
        pass

    def _get_extent(self):
        """Return the bounds of this roi in degrees.

        Extent is specified as a tuple of (minlon, maxlon, minlat, maxlat).

        Returns
        -------
        extent : tuple
            Bounds of this roi (minlon, maxlon, minlat, maxlat) [degrees].

        Examples
        --------

        """
        width = ch.km2deg(2*self.wsize*self.rad, self.cds.calc_mpp(self.lat),
                          self.cds.ppd)
        height = ch.km2deg(2*self.wsize*self.rad, self.cds.calc_mpp(),
                           self.cds.ppd)
        minlon = self.lon-(height/2)
        maxlon = minlon + width
        minlat = self.lat-(width/2)
        maxlat = minlat + height
        return (minlon, maxlon, minlat, maxlat)

    def _get_roi(self):
            """Return 2D numpy array with data from self.cds.

            This roi is centered on (self.lat, self.lon). The window extends
            wsize crater radii from the crater center.

            Use _wrap_lon() if cds roi crosses lon extent of global dataset.

            Parameters
            ----------
            mask_crater : bool
                Masks the crater floor from the resulting ROI by replacing the
                crater with an ellipse of NaN.
            plot_roi : bool
                Plots the returned ROI.

            Returns
            --------
            roi: 2Darray
                Numpy 2D array of data centered on the specified crater and
                extending wsize*rad distance from the crater centre.
            """
            # If lon > 180, switch lon convention from (0, 360) -> (-180, 180)
            if self.lon > 180:
                self.lon -= 360
            return self.cds.get_roi(*self.extent)

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
        nanmask = ~np.isnan(roi)  # build nanmask with pre-existing nans
        if not strict:
            nanmask[nanmask] &= roi[nanmask] > vmax  # Add values > vmax
            nanmask[nanmask] &= roi[nanmask] < vmin  # Add values < vmin
        else:  # if strict, also exclude values equal to vmin, vmax
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
        self.roi[np.where(mask)] = fillvalue
        return self.roi

    def plot(self, *args, **kwargs):
        """Plot this CraterRoi. See plotting.plot_CraterRoi()"""
        from craterpy.plotting import plot_CraterRoi
        plot_CraterRoi(self, *args, **kwargs)
