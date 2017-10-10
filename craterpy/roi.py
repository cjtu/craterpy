"""Contains the CraterRoi object"""
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import gdal
from craterpy.helper import km2deg


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
    >>> ds = CraterpyDataset(dsfile, radius=1737)
    >>> roi = ds.get_roi(-27.2, 80.9, 207)  # Humboldt crater
    """
    def __init__(self, cds, lat, lon, rad, wsize=2, plot=False):
        self.cds = cds
        self.lat = lat
        self.lon = lon
        self.rad = rad
        self.extent = extent

        # self.__masked_where_pixels = []

    def __repr__(self):
        pass

    def _wrap360(cds, minlon, maxlon, topind, height):
        """Return ROI that crosses the 360 degree edge of a global dataset.

        Parameters
        ----------
        cds : CraterpyDataset
            Crater latitude, centre latitude of ROI.
        lon : int, float
            Crater longitude, centre longitude of ROI.
        rad : int, float
            Crater radius.
        wsize : int, float
            Window size in crater radii. Side length of ROI around crater is
            2*wsize*rad (E.g., ROI with wsize=1 gives smallest square around
            crater, making the side lengths 1 diam).
        mask_crater : bool
            Masks the crater floor from the resulting ROI by replacing the
            crater with an ellipse of NaN.
        plot_roi : bool
            Plots the returned ROI.
        get_extent : bool
            Reuturns the ROI window extent as (minlon, maxlon, minlat, maxlat)

        Returns
        --------
        roi: 2Darray
            Numpy 2D array of data centered on the specified crater and
            extending wsize*rad distance from the crater centre.

        """
        if minlon < cds.wlon:
            leftind = af.get_ind(minlon, lonarr - 360)
            leftwidth = af.deg2pix(cds.wlon - minlon, cds.ppd)
            rightind = af.get_ind(cds.wlon, lonarr)
            rightwidth = af.deg2pix(maxlon - cds.wlon, cds.ppd)
        elif maxlon > cds.elon:
            leftind = af.get_ind(minlon, lonarr)
            leftwidth = af.deg2pix(cds.elon - minlon, cds.ppd)
            rightind = af.get_ind(cds.elon, lonarr + 360)
            rightwidth = af.deg2pix(maxlon - cds.elon, cds.ppd)
        left_roi = cds.ReadAsArray(leftind, topind, leftwidth, height)
        right_roi = cds.ReadAsArray(rightind, topind, rightwidth, height)
        return np.concatenate((left_roi, right_roi), axis=1)


    def get_roi(self, lat, lon, rad, wsize=1, mask_crater=False,
                plot_roi=False, get_extent=False):
            """
            Return square 2D numpy array containing region of interest (ROI)
            centered on (lat,lon). The window size is given by 2*wsize*rad and
            extends wsize crater radii from the crater center.

            The crater at the center given by the ellipse at lat, lon, rad is
            excluded from the ROI with the mask_crater flag. This replaces pixels
            in the crater rim with NaN.

            If the requested ROI crosses the lon extent of a global dataset, use
            wrap_lon() to concatenate the parts of the roi on either side of the
            boundary. Otherwise, raise error if lat or lon out of bounds.

            Parameters
            ----------
            lat : int, float
                Crater latitude, centre latitude of ROI.
            lon : int, float
                Crater longitude, centre longitude of ROI.
            rad : int, float
                Crater radius.
            wsize : int, float
                Window size in crater radii. Side length of ROI around crater is
                2*wsize*rad (E.g., ROI with wsize=1 gives smallest square around
                crater, making the side lengths 1 diam).
            mask_crater : bool
                Masks the crater floor from the resulting ROI by replacing the
                crater with an ellipse of NaN.
            plot_roi : bool
                Plots the returned ROI.
            get_extent : bool
                Reuturns the ROI window extent as (minlon, maxlon, minlat, maxlat)

            Returns
            --------
            roi: 2Darray
                Numpy 2D array of data centered on the specified crater and
                extending wsize*rad distance from the crater centre.

            roi, extent : (2Darray, tuple)
                If get_extent flag is True, return both the 2D roi and the extent
                tuple.
            """
            # If lon > 180, switch lon convention from (0, 360) -> (-180, 180)
            if lon > 180:
                lon -= 360

            # Get window extent in degrees
            dwsize = af.km2deg(wsize*rad, self.calc_mpp(lat), self.ppd)
            minlat = lat-dwsize
            maxlat = lat+dwsize
            minlon = lon-dwsize
            maxlon = lon+dwsize
            extent = (minlon, maxlon, minlat, maxlat)
            # Throw error if window bounds are not in lat bounds.
            if minlat < self.slat or maxlat > self.nlat:
                raise ImportError('Latitude ({},{}) out of dataset bounds \
                                  ({},{})'.format(minlat, maxlat,
                                                  self.slat, self.nlat))
            # Make dataset pixel arrays
            latarr = np.linspace(self.nlat, self.slat, self.RasterYSize)
            lonarr = np.linspace(self.wlon, self.elon, self.RasterXSize)
            # Get top index and height of ROI
            topind = af.get_ind(maxlat, latarr)
            height = af.deg2pix(2*dwsize, self.ppd)
            # Check if ROI crosses lon extent of global ROI and needs to be wrapped
            if self.is_global() and (minlon < self.wlon or maxlon > self.elon):
                roi = wrap_lon(self, minlon, maxlon, topind, height)
            else:
                # Get left index and width of ROI and read data from dataset
                leftind = af.get_ind(minlon, lonarr)
                width = af.deg2pix(2*dwsize, self.ppd)
                roi = self.ReadAsArray(leftind, topind, width, height)
            if roi is None:
                raise ImportError('GDAL could not read dataset into array')
                return
            if mask_crater:
                cmask = af.crater_floor_mask(self, roi, lat, lon, rad)
                roi = af.mask_where(roi, cmask)
            if plot_roi:
                self.plot_roi(roi, extent=extent)
            if get_extent:
                return roi, extent
            else:
                return roi

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
        mask = np.array(np.ones(ndarray.shape))  # Same shape array of ones
        mask[np.where(condition)] = np.nan
        return ndarray * mask

    def plot(self, *args, **kwargs):
        """Plot this CraterRoi. See plotting.plot_CraterRoi()"""
        craterpy.plotting.plot_CraterRoi(self, *args, **kwargs)
