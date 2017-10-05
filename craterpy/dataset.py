"""Contains the CpyDataset object which wraps gdal.Dataset."""
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import gdal
from acerim import acefunctions as af
gdal.UseExceptions()  # configure gdal to use Python exceptions


class CpyDataset(object):
    """The CpyDataset is a specialized version of the GDAL Dataset object.

    The CpyDataset only supports simple cylindrically projected datasets. It 
    can open any file format accepted by gdal.Open(). If the input file is a
    GeoTIFF, the geographical bounds and resolution will be read
    automatically. Otherwise, the attributes must be passed in the constructor.

    CpyDataset inherits all attributes and methods from gdal.Dataset.


    Attributes
    ----------
    nlat, slat, wlon, elon : int or float
        North, south, west, and east bounds of dataset [degrees].
    radius : int or float
        Radius of the planeary body [km].
    ppd : int or float
        Resolution of dataset in [pixels per degree].

    Methods
    -------
    get_roi(lat, lon, radius, wsize=1)
        Return 2D region of interest array around a crater.

    See Also
    --------
    gdal.Dataset
    
    Examples
    --------
    >>> import os.path as p
    >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
    >>> dsfile = p.join(datadir, 'moon.tif')
    >>> ds = CpyDataset(dsfile, radius=1737)
    >>> roi = ds.get_roi(-27.2, 80.9, 207)  # Humboldt crater
    """
    def __init__(self, dataset, nlat=None, slat=None, wlon=None, elon=None,
                 radius=None, ppd=None, **kwargs):
        """Initialize CpyDataset object."""
        # gdal.Dataset is not easily inherited from so we wrap it instead
        if isinstance(dataset, gdal.Dataset):
            self.gdalDataset = dataset  
        else:
            self.gdalDataset = gdal.Open(dataset)

        args = [nlat, slat, wlon, elon, radius, ppd]
        attrs = ['nlat', 'slat', 'wlon', 'elon', 'radius', 'ppd']
        geotags = self._get_info()  # Attempt to read geotiff tags
        for i, arg in enumerate(args):
            if arg is None:  # Get missing attrs from geotiff tags
                setattr(self, attrs[i], geotags[i])
            else:  # If argument is supplied, override geotiff tags
                setattr(self, attrs[i], arg)

    def __getattr__(self, name):
        """Wraps self.gdalDataset."""
        if name not in self.__dict__:  # Redirect if not in CpyDataset
            try:
                func = getattr(self.__dict__['gdalDataset'], name)
                if callable(func):  # Call method
                    def gdalDataset_wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                    return gdalDataset_wrapper
                else:  # Not callable so must be attribute
                    return func
            except AttributeError:
                raise AttributeError('Object has no attribute {}'.format(name))

    def __repr__(self):
        """Representation of CpyDataset with attribute info"""
        attrs = (self.nlat, self.slat, self.wlon, self.elon,
                 self.radius, self.ppd)
        rep = 'CpyDataset with extent ({}N, {}N)'.format(*attrs[:2])
        rep += '({}E, {}E), radius {} km, '.format(*attrs[2:5])
        rep += 'and resolution {} ppd'.format(attrs[5])
        return rep

    def calc_mpp(self, lat=0):
        """Return the ground resolution in meters/pixel at the given latitude.
        
        Due to stretching towards the poles in the simple-cylindrical
        projection, pixels have higher meter resolution near the equator.
        This simple function calculates the latitude-dependent mpp resolution.

        Parameters
        ----------
        lat: int or float
            Latitude (Default is 0, the equator).

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CpyDataset(dsfile, radius=1737)
        >>> '{:.0f}'.format(ds.calc_mpp())
        '7579.1'
        >>> '{:.0f}'.format(ds.calc_mpp(50))
        '4871.7'
        """
        circ = 2*np.pi*1000*self.radius*np.cos(lat)  # circumference at lat [m]
        npix = 360*self.ppd  # number of pixels in one circumference [pix]
        return circ/npix  # number of meters in one pixel at lat [m]/[pix]

    def _get_info(self):
        """Get geotiff args from gdal.Datast.GetGeoTransform() method.

        Returns
        -------
        nlat : int or float
            North latitude [degrees].
        slat : int or float
            South latitude [degrees].
        wlon : int or float
            West longitude [degrees].
        elon : int or float
            East longitude [degrees].
        ppd : float
            Resolution [pixels/degree].

        Examples
        --------
        >>> import os
        >>> f = os.path.dirname(os.path.abspath('__file__'))+'/tests/moon.tif'
        >>> ds = CpyDataset(f)
        >>> ds.get_info()
        (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        """
        xsize, ysize = self.RasterXSize, self.RasterYSize
        geotrans = self.GetGeoTransform()
        try:  # Try to get info assuming WKT format
            radius = 0.001*float(self.GetProjection().split(',')[3])
        except IndexError:
            raise ImportError('Dataset radius not defined')
            radius = None
        wlon, dpp, nlat = (geotrans[0], geotrans[1], geotrans[3])
        elon = wlon + xsize*dpp
        slat = nlat - ysize*dpp
        ppd = 1/dpp
        return nlat, slat, wlon, elon, radius, ppd

    def is_global(self):
        """
        Check if dataset has 360 degrees of longitude.

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CpyDataset(dsfile, radius=1737)
        >>> ds.is_global()
        True
        """
        return np.isclose(self.elon, self.wlon + 360)

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
        def wrap_lon(ads, minlon, maxlon, topind, height):
            """
            Return ROI that extends past the right edge of a global dataset by
            wrapping and concatenating the right and left sides of the ROI.
            """
            if minlon < ads.wlon:
                leftind = af.get_ind(minlon, lonarr - 360)
                leftwidth = af.deg2pix(ads.wlon - minlon, ads.ppd)
                rightind = af.get_ind(ads.wlon, lonarr)
                rightwidth = af.deg2pix(maxlon - ads.wlon, ads.ppd)
            elif maxlon > ads.elon:
                leftind = af.get_ind(minlon, lonarr)
                leftwidth = af.deg2pix(ads.elon - minlon, ads.ppd)
                rightind = af.get_ind(ads.elon, lonarr + 360)
                rightwidth = af.deg2pix(maxlon - ads.elon, ads.ppd)
            left_roi = ads.ReadAsArray(leftind, topind, leftwidth, height)
            right_roi = ads.ReadAsArray(rightind, topind, rightwidth, height)
            return np.concatenate((left_roi, right_roi), axis=1)

        # If lon out of bounds, switch lon convention [(0,360) <-> (-180,180)]
        if lon > self.elon:
            lon -= 360
        if lon < self.wlon:
            lon += 360
        # Get window extent in degrees
        dwsize = af.m2deg(wsize*rad, self.calc_mpp(lat), self.ppd)
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

    def plot_roi(self, roi, *args, **kwargs):
        """
        Implements plot_roi function in functions.py.
        """
        af.plot_roi(self, roi, *args, **kwargs)


class CraterSeries(pd.Series):
    """Underlying series object of the CraterDataFrame. Necessary when
    subclassing pandas.DataFrame.
    """
    @property
    def _constructor(self):
        return CraterSeries


class CraterDataFrame(pd.DataFrame):
    """
    Extends DataFrame from the Pandas module. DataFrames are "two-dimensional
    size-mutable, potentially heterogeneous tabular data structures". They are
    a convenient way to contain and manipulate tabular data in Python.

    The CraterDataFrame differs from the stock DataFrame in that it detects the
    columns of latitude, longitude, and radius (or diameter),
    storing a reference to each column in latcol, loncol and radcol,
    respectively. This ensures that crater location data can be extracted from
    the correct data columns. It also allows the option to initialize from
    a csv filename or by passing a pandas.DataFrame object with Lat, Lon, and
    size columns.

    Sicne CraterDataFrame inherits from pandas.DataFrame, all DataFrame methods
    are available for use in the CraterDataFrame. See help(pandas.DataFrame)
    for methods and DataFrame usage.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Str will assume a filename ending in '.csv' with data to be read. It is
        recommended to pass a pandas.DataFrame object if complicated import
        options are required.
    **kwargs :
        All optional arguments from pandas.DataFrame can be appended as keyword
        arguments.

    Examples
    --------
    >>> cdict = {'Lat' : [10, -20., 80.0],
                 'Lon' : [14, -40.1, 317.2],
                 'Diam' : [2, 12., 23.7]}
    >>> cdf = CraterDataFrame(cdict)
    >>> cdf['Diam'][0]
    2.0
    >>> index = ['Crater A', 'Crater B', 'Crater C']
    >>> cdf2 = CraterDataFrame(cdict, index=index)
    >>> cdf2.loc['Crater A']['Lat']
    10.0
    >>> cdf2.latcol
    'Lat'
    """
    def __init__(self, data=None, index_col=None, latcol=None, loncol=None,
                 radcol=None, **kwargs):
        """
        Initialize a CraterDataFrame object with data (a str filename, a
        pandas.DataFrame object, or any of the acceptable inputs to
        pandas.DataFrame). See help(CraterDataFrame) for correct usage.
        """
        if isinstance(data, str):
            data = pd.read_csv(data, index_col=index_col)
        super(CraterDataFrame, self).__init__(data, **kwargs)

        # If no lat-, lon-, or rad- col provided, try to find them in columns
        colnames = ['latitude', 'longitude', 'radius']
        colabbrevs = ['lat', 'lon', 'rad']
        attrs = [latcol, loncol, radcol]

        for i in range(len(colnames)):
            if not attrs[i]:  # If not defined in constructor
                findcol = [(colnames[i] == col.strip().lower()) or
                           (colabbrevs[i] == col.strip().lower()) for
                           col in self.columns]
                if any(findcol):
                    attrs[i] = self.columns[np.where(findcol)[0][0]]
                    continue
                elif colnames[i] == 'radius':
                    finddiam = [('diameter' == col.strip().lower()) or
                                ('diam' == col.strip().lower()) for
                                col in self.columns]
                    if any(finddiam):
                        diamcol = self.columns[np.where(finddiam)[0][0]]
                        self['radius'] = (0.5)*self[diamcol]
                        attrs[i] = 'radius'
                        continue
                raise ImportError('Unable to infer {} column from header. \
                                  Specify {}col in \
                                  constructor.'.format(colnames[i],
                                                       colabbrevs[i]))
        self.latcol = attrs[0]
        self.loncol = attrs[1]
        self.radcol = attrs[2]

    @property
    def _constructor(self):
        return CraterDataFrame

    _constructor_sliced = CraterSeries


if __name__ == "__main__":
    import doctest
    doctest.testmod()
