"""Contains the CraterpyDataset object which wraps gdal.Dataset."""
from __future__ import division, print_function, absolute_import
from craterpy.exceptions import DataImportError
import numpy as np
import gdal
import craterpy.helper as ch
gdal.UseExceptions()  # configure gdal to use Python exceptions


class CraterpyDataset(object):
    """The CraterpyDataset is a specialized version of the GDAL Dataset object.

    The CraterpyDataset only supports simple cylindrically projected datasets.
    It opens files supported by gdal.Open(). If the input file is a GeoTIFF,
    the geographical bounds and resolution will be read automatically.
    Otherwise, all attributes must be passed in the constructor.

    CraterpyDataset inherits all attributes and methods from gdal.Dataset.


    Attributes
    ----------
    nlat, slat, wlon, elon : int or float
        North, south, west, and east bounds of dataset [degrees].
    radius : int or float
        Radius of the planeary body [km].
    ppd : int or float
        Resolution of dataset in [pixels per degree].
    nodata : int or float
        Numerical value of invalid (no data) pixels.

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
    >>> ds = CraterpyDataset(dsfile, radius=1737)
    >>> roi = ds.get_roi(-27.2, 80.9, 207)  # Humboldt crater
    """
    def __init__(self, dataset, nlat=None, slat=None, wlon=None, elon=None,
                 radius=None, ppd=None, nodata=None):
        """Initialize CraterpyDataset object."""
        # gdal.Dataset is not easily inherited from so we wrap it instead
        if isinstance(dataset, gdal.Dataset):
            self.gdalDataset = dataset
        else:
            self.gdalDataset = gdal.Open(dataset)


        args = [nlat, slat, wlon, elon, radius, ppd]
        attrs = ['nlat', 'slat', 'wlon', 'elon', 'radius', 'ppd']
        geotags = self._get_geotiff_info()  # Attempt to read geotiff tags
        for i, arg in enumerate(args):
            if arg is None:  # Get missing attrs from geotiff tags
                setattr(self, attrs[i], geotags[i])
            else:  # If argument is supplied, override geotiff tags
                setattr(self, attrs[i], arg)
        if not self.ppd:
            self.ppd = self.RasterXSize/(self.elon-self.wlon)
        self.latarr = np.linspace(self.nlat, self.slat, self.RasterYSize)
        self.lonarr = np.linspace(self.wlon, self.elon, self.RasterXSize)

    def __getattr__(self, name):
        """Wraps self.gdalDataset."""
        if name not in self.__dict__:  # Redirect if not in CraterpyDataset
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
        """Representation of CraterpyDataset with attribute info"""
        attrs = (self.nlat, self.slat, self.wlon, self.elon,
                 self.radius, self.ppd)
        rep = 'CraterpyDataset with extent ({}N, {}N), '.format(*attrs[:2])
        rep += '({}E, {}E), radius {} km, '.format(*attrs[2:5])
        rep += 'and resolution {} ppd'.format(attrs[5])
        return rep

    def _get_geotiff_info(self):
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
        >>> ds = CraterpyDataset(f)
        >>> ds.get_info()
        (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        """
        xsize, ysize = self.RasterXSize, self.RasterYSize
        geotrans = self.GetGeoTransform()
        try:  # Try to get info assuming WKT format
            radius = 0.001*float(self.GetProjection().split(',')[3])
        except Exception:
            #print('Dataset radius not defined')
            radius = None
        wlon, dpp, nlat = (geotrans[0], geotrans[1], geotrans[3])
        elon = wlon + xsize*dpp
        slat = nlat - ysize*dpp
        ppd = 1/dpp if dpp else xsize/(elon-wlon)
        return nlat, slat, wlon, elon, radius, ppd

    def calc_mpp(self, lat=0):
        """Return the ground resolution in meters/pixel at the given latitude.

        Due to stretching towards the poles in the simple-cylindrical
        projection, mpp resolution is latitude-dependent.

        Parameters
        ----------
        lat: int or float
            Latitude (Default is 0, the equator).

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CraterpyDataset(dsfile, radius=1737)
        >>> '{:.0f}'.format(ds.calc_mpp())
        '7579.1'
        >>> '{:.0f}'.format(ds.calc_mpp(50))
        '4871.7'
        """
        if abs(lat) > 90:
            raise ValueError("Latitude out of bounds")
        # calculate circumference at lat in [m], divide by num pixels
        circ = 1000*2*np.pi*self.radius*np.cos(np.radians(lat))
        npix = 360*self.ppd  # num pixels in one circumference [pix]
        return circ/npix  # num meters in one pixel at lat [m]/[pix]

    def inbounds(self, lat, lon):
        """Return True if (lat, lon) point in Dataset bounds.

        Parameters
        ----------
        lat : int or float
            Latitude [degrees].
        lon : int or float
            Longitude [degrees].

        Examples
        --------
        >>> ds = CraterpyDataset(dsfile, nlat=20, slat=0, wlon=10, elon=20)
        >>> ds.inbounds(10, 15)
        True
        >>> ds.inbounds(10, 200)
        False
        >>> ds = CraterpyDataset(dsfile, nlat=20, slat=0, wlon=-180, elon=180)
        >>> ds.inbounds(10, 15)
        True
        >>> ds.inbounds(10, 200)
        True
        """
        if self.is_global():
            return self.slat <= lat <= self.nlat
        else:
            return ((self.slat <= lat <= self.nlat) and
                    (self.wlon <= lon <= self.elon))

    def is_global(self):
        """
        Check if dataset has 360 degrees of longitude.

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'examples')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CraterpyDataset(dsfile, radius=1737)
        >>> ds.is_global()
        True
        """
        return np.isclose(self.elon, self.wlon + 360)

    def get_roi(self, minlon, maxlon, minlat, maxlat):
        """Return numpy array of data specified by its geographical bounds

        Parameters
        ----------
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
        # TODO
        """
        if (not self.inbounds(minlat, minlon) or not
                self.inbounds(maxlat, maxlon)):
            raise DataImportError("Roi extent out of dataset bounds.")
        topind = ch.deg2pix(self.nlat-maxlat, self.ppd)
        height = ch.deg2pix(maxlat-minlat, self.ppd)
        if self.is_global() and (minlon < self.wlon or maxlon > self.elon):
            roi = self._wrap_roi_360(minlon, maxlon, topind, height)
        else:
            leftind = ch.deg2pix(minlon-self.wlon, self.ppd)
            width = ch.deg2pix(maxlon-minlon, self.ppd)
            roi = self.ReadAsArray(leftind, topind, width, height)
        return roi.astype(float)

    def _wrap_roi_360(self, minlon, maxlon, topind, height):
            """Return roi that is split by the 360 degree edge of a global dataset.

            Read the left and right sub-arrays and then concatenate them into
            the full roi.

            Parameters
            ----------
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
            if minlon < self.wlon:
                leftind = ch.deg2pix(minlon-(self.wlon-360), self.ppd)
                leftwidth = ch.deg2pix(self.wlon - minlon, self.ppd)
                rightind = 0
                rightwidth = ch.deg2pix(maxlon - self.wlon, self.ppd)
            elif maxlon > self.elon:
                leftind = ch.deg2pix(self.elon - minlon, self.ppd)
                leftwidth = ch.deg2pix(self.elon - minlon, self.ppd)
                rightind = 0
                rightwidth = ch.deg2pix(maxlon - self.elon, self.ppd)
            left_roi = self.ReadAsArray(leftind, topind, leftwidth, height)
            right_roi = self.ReadAsArray(rightind, topind, rightwidth, height)
            return np.concatenate((left_roi, right_roi), axis=1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
