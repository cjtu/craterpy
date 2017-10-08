"""Contains the CraterpyDataset object which wraps gdal.Dataset."""
from __future__ import division, print_function, absolute_import
import numpy as np
import gdal
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
        geotags = self._get_info()  # Attempt to read geotiff tags
        for i, arg in enumerate(args):
            if arg is None:  # Get missing attrs from geotiff tags
                setattr(self, attrs[i], geotags[i])
            else:  # If argument is supplied, override geotiff tags
                setattr(self, attrs[i], arg)

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
        rep = 'CraterpyDataset with extent ({}N, {}N)'.format(*attrs[:2])
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
        >>> ds = CraterpyDataset(dsfile, radius=1737)
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
        >>> ds = CraterpyDataset(f)
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
        >>> ds = CraterpyDataset(dsfile, radius=1737)
        >>> ds.is_global()
        True
        """
        return np.isclose(self.elon, self.wlon + 360)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
