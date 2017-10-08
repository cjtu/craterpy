import numpy as np
from craterpy.functions import m2pix, deg2pix


def circle_mask(roi, radius, center=(None, None)):
    """
    Return boolean array of True inside circle of radius at center.

    >>> roi = np.ones((3,3))
    >>> masked = circle_mask(roi, 1)
    >>> masked[1,1]
    True
    >>> masked[0,0]
    False
    """
    if not center[0]:  # Center circle on center of roi
        center = np.array(roi.shape)/2 - 0.5
    cx, cy = center
    width, height = roi.shape
    x = np.arange(width) - cx
    y = np.arange(height).reshape(-1, 1) - cy
    return x*x + y*y <= radius*radius


def ellipse_mask(roi, a, b, center=(None, None)):
    """
    Return boolean array of True inside ellipse with horizontal major axis a
    and vertical minor axis b centered at center.

    >>> roi = np.ones((9,9))
    >>> masked = ellipse_mask(roi, 3, 2)
    >>> masked[4,1]
    True
    >>> masked[4,0]
    False
    """
    if not center[0]:  # Center ellipse on center of roi
        center = np.array(roi.shape)/2 - 0.5
    cx, cy = center
    width, height = roi.shape
    y, x = np.ogrid[-cx:width-cx, -cy:height-cy]
    return (x*x)/(a*a) + (y*y)/(b*b) <= 1


def ring_mask(roi, rmin, rmax, center=(None, None)):
    """
    Return boolean array of True in a ring from rmin to rmax radius around
    center. Returned array is same shape as roi.
    """
    inner = circle_mask(roi, rmin, center)
    outer = circle_mask(roi, rmax, center)
    return outer*~inner


def crater_floor_mask(cds, roi, lat, lon, rad):
    """
    Mask the floor of the crater lat, lon, with radius rad.
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
    cds = AceDatset(datafile)
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
