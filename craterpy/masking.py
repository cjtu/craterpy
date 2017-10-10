import numpy as np
from craterpy.functions import m2pix, deg2pix


def circle_mask(shape, radius, center=None):
    """Return boolean array of True inside circle of radius at center.

    Parameters
    ----------
    shape : tuple of int
        Shape of output boolean mask array of the form (ysize, xsize).
    radius : int or float
        Radius of circle [pixels].
    center : tuple of int
        Two element tuple of (yindex, xindex) of circle center (defaults to
        center of the mask).

    Returns
    -------
    mask : numpy 2D array

    Examples
    --------
    >>> circle_mask((3,3), 1)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> circle_mask((3,3), 1, center=(1, 2))
    array([[False, False,  True],
           [False,  True,  True],
           [False, False,  True]], dtype=bool)
    """
    if center is None:  # Center circle on center of roi
        center = np.array(shape)/2 - 0.5
    cy, cx = center
    height, width = shape
    x = np.arange(width) - cx
    y = np.arange(height).reshape(-1, 1) - cy
    return x*x + y*y <= radius*radius


def ellipse_mask(shape, ysize, xsize, center=None):
    """Return numpy 2D boolean array of True inside  ellipse.

    Parameters
    ----------
    shape : tuple of int
        Shape of output boolean mask array of the form (ysize, xsize).
    ysize : int or float
        Vertical semi-axis [pixels].
    xsize : int or float
        Horizontal semi-axis [pixels].
    center : tuple of int
        Two element tuple of (yindex, xindex) of ellipse center (defautls to
        center of the mask).

    Returns
    -------
    mask : numpy 2D array

    Examples
    --------
    >>> ellipse_mask((4,5), 1.5, 3)
    array([[False, False,  True, False, False],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [False, False,  True, False, False]], dtype=bool)
    >>> ellipse_mask((4,5), 1.5, 3, center=(1.5, 3))
    array([[False, False, False,  True, False],
           [False,  True,  True,  True,  True],
           [False,  True,  True,  True,  True],
           [False, False, False,  True, False]], dtype=bool)
    """
    if center is None:  # Center ellipse on center of roi
        center = np.array(shape)/2 - 0.5
    cy, cx = center
    height, width = shape
    y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
    return (x*x)/(xsize*xsize) + (y*y)/(ysize*ysize) <= 1


def ring_mask(shape, rmin, rmax, center=None):
    """Return bool array of True in a circular ring from rmin to rmax.

    Parameters
    ----------
    shape : tuple of int
        Shape of output boolean mask array of the form (ysize, xsize).
    rmin : int or float
        Inner ring radius [pixels].
    rmax : int or float
        Outer ring radius [pixels].
    center : tuple of int
        Two element tuple of (yindex, xindex) of ellipse center (defautls to
        center of the mask).

    Returns
    -------
    mask : numpy 2D array

    Examples
    --------
    >>> ring_mask((5,5), 1, 2)
    array([[False, False,  True, False, False],
           [False,  True, False,  True, False],
           [ True, False, False, False,  True],
           [False,  True, False,  True, False],
           [False, False,  True, False, False]], dtype=bool)
    >>> ring_mask((5,5), 0.5, 1.5)
    array([[False, False, False, False, False],
           [False,  True,  True,  True, False],
           [False,  True, False,  True, False],
           [False,  True,  True,  True, False],
           [False, False, False, False, False]], dtype=bool)
    """
    inner = circle_mask(shape, rmin, center)
    outer = circle_mask(shape, rmax, center)
    return outer*~inner


def crater_floor_mask(cds, roi, lat, lon, rad):
    """Mask the floor of the crater lat, lon, with radius rad.

    Parameters
    ----------
    shape : tuple of int
        Shape of output boolean mask array of the form (ysize, xsize).
    rmin : int or float
        Inner ring radius [pixels].
    rmax : int or float
        Outer ring radius [pixels].
    center : tuple of int
        Two element tuple of (yindex, xindex) of ellipse center (defautls to
        center of the mask).

    Returns
    -------
    mask : numpy 2D array

    Examples
    --------
    """
    pixwidth = m2pix(rad, cds.calc_mpp(lat))
    pixheight = m2pix(rad, cds.calc_mpp())
    return ellipse_mask(roi, pixwidth, pixheight)


def crater_ring_mask(cds, roi, lat, lon, rmin, rmax):
    """
    Mask a ring around a crater with inner radius rmin and outer radius rmax
    crater radii.
    """
    rmax_pixheight = m2pix(rmax, cds.calc_mpp())
    rmax_pixwidth = m2pix(rmax, cds.calc_mpp(lat))
    rmin_pixheight = m2pix(rmin, cds.calc_mpp())
    rmin_pixwidth = m2pix(rmin, cds.calc_mpp(lat))
    outer = ellipse_mask(roi, rmax_pixwidth, rmax_pixheight)
    inner = ellipse_mask(roi, rmin_pixwidth, rmin_pixheight)
    return outer * ~inner


def polygon_mask(cds, roi, extent, poly_verts):
    """
    Mask the region inside a polygon given by poly_verts.

    Parameters
    ==========
    cds
    roi
    extent: (float, float, float, float)
        Extent tuple of (minlon, maxlon, minlat, maxlat).
    poly_verts: list of tuple
        List of (lon, lat) polygon vertices.

    Example
    =======
    cds = CraterpyDatset(datafile)
    roi, extent = cds.get_roi(-27, 80.9, 94.5, wsize=2, get_extent=True)
    mask = polygon_mask(cds, roi, extent, poly_verts)
    masked = mask_where(roi, ~mask)
    plot_roi(cds, masked, vmin=0, vmax=1)
    """
    from matplotlib.path import Path
    minlon, maxlon, minlat, maxlat = extent
    # Create grid
    nlat, nlon = roi.shape
    x, y = np.meshgrid(np.arange(nlon), np.arange(nlat))
    x, y = x.flatten(), y.flatten()
    gridpoints = np.vstack((x, y)).T

    poly_pix = [(deg2pix(lon-minlon, cds.ppd),
                deg2pix(lat-minlat, cds.ppd)) for lon, lat in poly_verts]
    path = Path(poly_pix)
    mask = path.contains_points(gridpoints).reshape((nlat, nlon))
    return mask
