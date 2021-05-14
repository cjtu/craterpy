"""Contains the CraterRoi object"""
import numpy as np
import rasterio as rio
import craterpy.helper as ch
from craterpy.plotting import plot_CraterRoi


def get_extent(cds, lat, lon, rad, wsize=1):
    """
    Return the extent of the CraterRoi in cds at lat, lon and size.

    Parameters
    ----------
    cds : CraterpyDataset
        CraterDataset to read roi from.
    lat: float
        The latitude of the crater center [degrees].
    lon: float
        The longitude of the crater center [degrees].
    rad: float
        The radius of the crater [km].
    size: float
        The window size to read in crater radii (default 1 crater radius).

    Returns
    -------
    extent : tuple
        Bounds of this roi (minlon, maxlon, minlat, maxlat) [degrees].
    """
    width = ch.km2deg(
        2 * wsize * rad,
        cds.calc_mpp(lat),
        cds.xres,
    )
    height = ch.km2deg(2 * wsize * rad, cds.calc_mpp(), cds.yres)
    minlon = lon - (width / 2)
    maxlon = minlon + width
    minlat = lat - (height / 2)
    maxlat = minlat + height
    return (minlon, maxlon, minlat, maxlat)


def wrap_roi_360(cds, minlon, maxlon, topind, height):
    """Return roi that is split by the edge of a global dataset.

    Read the left and right sub-arrays and then concatenate them into
    the full roi.

    Parameters
    ----------
    cds : CraterpyDataset
        CraterDataset to read roi from.
    minlon : int or float
        Western longitude bound [degrees].
    maxlon : int or float
        Eastern longitude bound [degrees].
    topind : int
        Top index of returned roi.
    height : int
        Height of returned roi.

    Returns
    --------
    roi: 2Darray
        Concatenated roi wrapped around lon bound.
    """
    elon, minlon, maxlon = ch.lon360(np.array([cds.elon, minlon, maxlon]))

    # The # of pixels roi to the right of elon
    rightwidth = ch.deg2pix(maxlon - elon, cds.xres)
    rightind = 0
    # The # of pixels left of the elon
    leftwidth = ch.deg2pix(elon - minlon, cds.xres)
    leftind = cds.width - leftwidth
    # Read left and right ROIs and concatenate around elon
    w_left = rio.windows.Window(leftind, topind, leftwidth, height)
    w_right = rio.windows.Window(rightind, topind, rightwidth, height)
    left_roi = cds.read(1, window=w_left)
    right_roi = cds.read(1, window=w_right)
    return np.concatenate((left_roi, right_roi), axis=1)


def get_roi_latlon(cds, minlon, maxlon, minlat, maxlat):
    """Return numpy array of data specified by its geographical bounds

    Parameters
    ----------
    cds : CraterpyDataset
        The CraterpyDataset from which to extract the roi.
    minlon : int or float
        Western longitude bound [degrees].
    maxlon : int or float
        Eastern longitude bound [degrees].
    minlat : int or float
        Southern latitude bound [degrees].
    maxlat : int or float
        Northern latitude bound [degrees].

    Returns
    -------
    roi : numpy 2D array
        Numpy array specified by extent given.

    Examples
    --------
    >>> import os.path as p
    >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
    >>> dsfile = p.join(datadir, 'moon.tif')
    >>> ds = CraterpyDataset(dsfile, radius=1737)
    >>> ds.get_roi(-27.6, -27.0, 80.5, 81.1).shape
    (2, 2)
    """
    if not cds.inbounds(minlat, minlon) or not cds.inbounds(maxlat, maxlon):
        raise ValueError("Roi extent out of dataset bounds.")
    topind = ch.deg2pix(cds.nlat - maxlat, cds.yres)
    height = ch.deg2pix(maxlat - minlat, cds.yres)
    if cds.is_global() and (minlon < cds.wlon or maxlon > cds.elon):
        roi = wrap_roi_360(cds, minlon, maxlon, topind, height)
    else:
        leftind = ch.deg2pix(minlon - cds.wlon, cds.xres)
        width = ch.deg2pix(maxlon - minlon, cds.xres)
        w = rio.windows.Window(leftind, topind, width, height)
        roi = cds.read(1, window=w)
    return roi.astype(float)


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

    def __init__(self, cds, lat, lon, rad, wsize=1, plot=False):
        self.cds = cds
        self.lat = lat
        self.lon = ch.lon180(lon) if cds.wlon < 0 else ch.lon360(lon)

        self.rad = rad
        self.wsize = wsize
        self.extent = get_extent(cds, self.lat, self.lon, self.rad, self.wsize)
        self.roi = get_roi_latlon(cds, *self.extent)
        if plot:
            self.plot()

    def __repr__(self):
        return "CraterRoi at ({}N, {}E) with radius {} km".format(
            self.lat, self.lon, self.rad
        )

    def filter(
        self,
        vmin=float("-inf"),
        vmax=float("inf"),
        strict=False,
        nodata=np.nan,
    ):
        """Filter roi to the inclusive range [vmin, vmax].

        You can specify only vmin or vmax if only a lower bound/upper bound is
        required. Set strict to True to exclude vmin and vmax, i.e. keep data
        in the exclusive interval (vmin, vmax). The nodata value replaces
        filtered pixels. It defaults to np.nan.

        Parameters
        ----------
        vmin : int or float
            Minimum value to keep (default -inf, no lower bound).
        vmax : int or float
            Maximum value to keep (default inf, no upper bound).
        strict : bool
            Keep values strictly greater and strictly less than vmin, vmax
            (default False).
        nodata : int or float
            Number to fill in filtered values (default np.nan).

        Examples
        --------
        >>> croi.filter(0, 1)  # keep data in range (0 < data < 1)
        >>> croi.filter(0, 1, True)  # keep data in range ( <= data <= 1)
        >>> croi.filter(vmax=20)  # keep data < 20
        """
        mask = np.isfinite(self.roi)  # filter pre-existing nans and infs
        if strict:
            # Mask includes pixels >= vmin and <= vmax
            mask[mask] = mask[mask] & (
                (self.roi[mask] > vmin) & (self.roi[mask] < vmax)
            )
        else:
            mask[mask] = mask[mask] & (
                (self.roi[mask] >= vmin) & (self.roi[mask] <= vmax)
            )
        self.roi[~mask] = nodata  # set invalid pixels (~mask) with nodata

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
        if outside:
            mask = ~mask
        self.roi[np.where(mask)] = fillvalue

    def plot(self, *args, **kwargs):
        """Plot this CraterRoi. See plotting.plot_CraterRoi()"""
        plot_CraterRoi(self, *args, **kwargs)

    def save(self, fname):
        """Save roi to file given by fname"""
        np.savetxt(fname, self.roi, delimiter=",")
