"""Contains the CraterpyDataset object which wraps rasterio."""
import warnings
import numpy as np
import rasterio as rio
import craterpy.helper as ch
import craterpy.roi as croi


class CraterpyDataset:
    """The CraterpyDataset reads in images with the rasterio dataset.

    If the input file is georeferenced, the geographical bounds and transform
    will be read automatically. Otherwise, all attributes must be passed in the
    constructor.

    Inherits all attributes and methods of rasterio.DatasetReader.


    Attributes
    ----------
    nlat, slat, wlon, elon : int or float
        North, south, west, and east bounds of dataset [degrees].
    radius : int or float
        Radius of the planetary body [km].
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
    https://rasterio.readthedocs.io/en/latest/topics/reading.html

    Examples
    --------
    >>> import os.path as p
    >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
    >>> dsfile = p.join(datadir, 'moon.tif')
    >>> ds = CraterpyDataset(dsfile, radius=1737)
    >>> ds.get_roi(-27.6, -27.0, 80.5, 81.1).shape
    (2, 2)
    """

    def __init__(
        self,
        dataset,
        nlat=None,
        slat=None,
        wlon=None,
        elon=None,
        radius=None,
        xres=None,
        yres=None,
    ):
        """Initialize CraterpyDataset object."""
        with warnings.catch_warnings(record=True) as w:
            self._rioDataset = rio.open(dataset)
        if w and isinstance(w[0].message, rio.errors.NotGeoreferencedWarning):
            self.transform = ()
        args = [nlat, slat, wlon, elon, radius, xres, yres]
        attrs = ["nlat", "slat", "wlon", "elon", "radius", "xres", "yres"]
        if not self.transform and any(a is None for a in args[:4]):
            msg = (
                "No geotransform detected. Please specify nlat, slat,"
                + "wlon, elon, and planet radius for CraterpyDataset."
            )
            raise ImportError(msg)
        geotags = [None] * len(args)
        if self.transform:
            geotags = self._get_geotiff_info()  # Attempt to read geotiff tags
        for arg, attr, geotag in zip(args, attrs, geotags):
            if arg is None:  # Get missing attrs from geotiff tags
                setattr(self, attr, geotag)
            else:  # If argument is supplied, override geotiff tags
                setattr(self, attr, arg)
        if getattr(self, "xres") is None:
            self.xres = self.width / (self.elon - self.wlon)
        if getattr(self, "yres") is None:
            self.yres = self.xres
        self.latarr = np.linspace(self.nlat, self.slat, self.height)
        self.lonarr = np.linspace(self.wlon, self.elon, self.width)

    def __getattr__(self, name):
        """Wraps the self._rioDataset DatasetReader object."""
        if name in self.__dict__:
            return getattr(self, name)
        return getattr(self._rioDataset, name)

    def __repr__(self):
        """Representation of CraterpyDataset with attribute info"""
        attrs = (
            self.nlat,
            self.slat,
            self.wlon,
            self.elon,
            self.radius,
            self.xres,
            self.yres,
        )
        rep = "CraterpyDataset with extent ({}N, {}N), ".format(*attrs[:2])
        rep += "({}E, {}E), radius {} km, ".format(*attrs[2:5])
        rep += "xres {} ppd, and yres {} ppd".format(*attrs[5:7])
        return rep

    def _get_geotiff_info(self):
        """Get geotiff info from self._rioDataset, if it exists.

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
        xres : float
            Longitude resolution [pixels/degree].
        yres : float
            Latitude resolution [pixels/degree].
        radius : float
            Radius of body [km]

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CraterpyDataset(dsfile)
        >>> ds._get_geotiff_info()
        (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0, 4.0)
        """
        nlat = slat = wlon = elon = xres = yres = radius = None
        # print(hasattr(self, 'transform'))
        width, height = self.width, self.height
        transform = self.transform
        wlon, nlat = transform * (0, 0)
        elon, slat = transform * (width, height)
        xres = 1 / transform[0]
        yres = -1 / transform[4]

        if hasattr(self.crs, "wkt"):
            radius = 0.001 * ch.get_spheroid_rad_from_wkt(self.crs.to_wkt())
        return nlat, slat, wlon, elon, radius, xres, yres

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
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CraterpyDataset(dsfile, radius=1737)
        >>> '{:.1f}'.format(ds.calc_mpp())
        '7579.1'
        >>> '{:.1f}'.format(ds.calc_mpp(50))
        '4871.7'
        """
        if abs(lat) > 90:
            raise ValueError("Latitude out of bounds")
        # calculate circumference at lat in [m], divide by num pixels
        circ = 1000 * 2 * np.pi * self.radius * np.cos(np.radians(lat))
        npix = 360 * self.xres  # num pixels in one circumference [pix]
        return circ / npix  # num meters in one pixel at lat [m]/[pix]

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
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
        >>> dsfile = p.join(datadir, 'moon.tif')
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
        return (self.slat <= lat <= self.nlat) and (
            self.wlon <= lon <= self.elon
        )

    def is_global(self):
        """
        Check if dataset has 360 degrees of longitude.

        Examples
        --------
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
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
        >>> import os.path as p
        >>> datadir = p.join(p.dirname(p.abspath('__file__')), 'craterpy', 'data')
        >>> dsfile = p.join(datadir, 'moon.tif')
        >>> ds = CraterpyDataset(dsfile, radius=1737)
        >>> ds.get_roi(-27.6, -27.0, 80.5, 81.1).shape
        (2, 2)
        """
        return croi.get_roi_latlon(self, minlon, maxlon, minlat, maxlat)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
