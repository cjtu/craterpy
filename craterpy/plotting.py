# -*- coding: utf-8 -*-
"""
This file contains several functions used in the ACERIM workflow.

For usage, see sample/tutorial.rst.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from acerim import acestats as acs


def plot_CraterRoi(roi, figsize=((8, 8)), extent=None, title='CraterRoi', 
                   vmin=None, vmax=None, cmap='gray', **kwargs):
    """Plot 2D CraterRoi.

    The plot offers limited arguments for basic customization. It is further
    customizable by supplying valid matplotlib.imshow() keyword-arguments. See 
    matplotlib.imshow for full documentation.

    Parameters
    ----------
    roi : CraterRoi object
        2D CraterRoi to plot.
    figsize : tuple
        Length and width of plot in inches (default 8in x 8in).
    title : str
        Plot title (default 'CraterRoi').
    vmin : int or float
        Minimum pixel data value for plotting.
    vmax : int or float
        Maximum pixel data value for plotting.
    cmap : str
        Color map to plot (default 'gray'). See matplotlib.cm for full list.

    Other parameters
    ----------------
    **kwargs : object
        Keyword arguments to pass to imshow. See matplotlib.pyplot.imshow
    """
    plt.figure(title, figsize=figsize)
    plt.imshow(roi, extent=croi.extent, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    plt.title(title)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.show()


def plot_ejecta_stats():
    """Plot ejecta statistics.
    """
    pass  # TODO: implement this


def plot_ejecta_profile_stats():
    """Plot ejecta profile statistics.
    """
    pass  # TODO: implement this


# ROI MANIPULATION
def filter_roi(roi, vmin=None, vmax=None, strict=False):
    """
    Filter values outside (vmin, vmax) by setting them to np.nan. If strict,
    keep only values strictly less than vmax and strictly greater than vmin.

    E.g. strict=False keeps vmin <= roi <= vmax
         strict=True keeps vmin < roi < vmax
    """
    if vmin is None:
        vmin = -np.inf
    if vmax is None:
        vmax = np.inf
    nanmask = ~np.isnan(roi)  # build nanmask with pre-existing nans, if any
    if not strict:
        nanmask[nanmask] &= roi[nanmask] > vmax  # Add values > vmax to nanmask
        nanmask[nanmask] &= roi[nanmask] < vmin  # Add values < vmin to nanmask
    else:  # if strict, exclude values equal to vmin, vmax
        nanmask[nanmask] &= roi[nanmask] >= vmax
        nanmask[nanmask] &= roi[nanmask] <= vmin
    roi[nanmask] = np.nan
    return roi


def mask_where(ndarray, condition):
    """
    Return copy of ndarray with nan entries where condition is True.

    >>> arr = np.array([1.,2.,3.,4.,5.])
    >>> masked = mask_where(arr, arr > 3)
    >>> np.isnan(masked[3:]).all()
    True
    """
    mask = np.array(np.ones(ndarray.shape))  # Same shape array of ones
    mask[np.where(condition)] = np.nan
    return ndarray * mask


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
    if radius > 0:
        return x*x + y*y <= radius*radius
    else:
        return np.zeros(roi.shape, dtype=bool)


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


def crater_floor_mask(aceds, roi, lat, lon, rad):
    """
    Mask the floor of the crater lat, lon, with radius rad.
    """
    pixwidth = m2pix(rad, aceds.calc_mpp(lat))
    pixheight = m2pix(rad, aceds.calc_mpp())
    return ellipse_mask(roi, pixwidth, pixheight)


def crater_ring_mask(aceds, roi, lat, lon, rmin, rmax):
    """
    Mask a ring around a crater with inner radius rmin and outer radius rmax
    crater radii.
    """
    rmax_pixheight = m2pix(rmax, aceds.calc_mpp())
    rmax_pixwidth = m2pix(rmax, aceds.calc_mpp(lat))
    rmin_pixheight = m2pix(rmin, aceds.calc_mpp())
    rmin_pixwidth = m2pix(rmin, aceds.calc_mpp(lat))
    outer = ellipse_mask(roi, rmax_pixwidth, rmax_pixheight)
    inner = ellipse_mask(roi, rmin_pixwidth, rmin_pixheight)
    return outer * ~inner


def polygon_mask(aceds, roi, extent, poly_verts):
    """
    Mask the region inside a polygon given by poly_verts.
    
    Parameters
    ==========
    aceds
    roi
    extent: (float, float, float, float)
        Extent tuple of (minlon, maxlon, minlat, maxlat).
    poly_verts: list of tuple
        List of (lon, lat) polygon vertices.
        
    Example
    =======
    ads = AceDatset(datafile)
    roi, extent = ads.get_roi(-27, 80.9, 94.5, wsize=2, get_extent=True)
    mask = polygon_mask(ads, roi, extent, poly_verts)
    masked = mask_where(roi, ~mask)
    plot_roi(ads, masked, vmin=0, vmax=1)
    """
    from matplotlib.path import Path
    minlon, maxlon, minlat, maxlat = extent
    # Create grid
    nlat, nlon = roi.shape
    x, y = np.meshgrid(np.arange(nlon), np.arange(nlat))
    x, y = x.flatten(), y.flatten()
    gridpoints = np.vstack((x,y)).T
    
    poly_pix = [(deg2pix(lon-minlon, aceds.ppd), 
                     deg2pix(lat-minlat, aceds.ppd)) for lon,lat in poly_verts]
    path = Path(poly_pix)
    mask = path.contains_points(gridpoints).reshape((nlat,nlon))
    return mask

# GEOSPATIAL FUNCTIONS
def inbounds(lat, lon, mode='std'):
    """True if lat and lon within global coordinates.
    Standard: mode='std' for lat in (-90, 90) and lon in (-180, 180).
    Positive: mode='pos' for lat in (0, 180) and lon in (0, 360)

    >>> lat = -10
    >>> lon = -10
    >>> inbounds(lat, lon)
    True
    >>> inbounds(lat, lon, 'pos')
    False
    """
    if mode == 'std':
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    elif mode == 'pos':
        return (0 <= lat <= 180) and (0 <= lon <= 360)


def m2deg(dist, mpp, ppd):
    """Return dist converted from meters to degrees."""
    return dist/(mpp*ppd)


def m2pix(dist, mpp):
    """Return dist converted from meters to pixels"""
    return int(dist/mpp)


def deg2pix(dist, ppd):
    """Return dist converted from degrees to pixels."""
    return int(dist*ppd)


def get_ind(value, array):
    """Return closest index (rounded down) of a value from sorted array."""
    ind = np.abs(array-value).argmin()
    return int(ind)


def deg2rad(theta):
    """
    Convert degrees to radians.

    >>> deg2rad(180)
    3.141592653589793
    """
    return theta * (np.pi / 180)


def greatcircdist(lat1, lon1, lat2, lon2, radius):
    """
    Return great circle distance between two points on a spherical body.
    Uses Haversine formula for great circle distances.

    >>> greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
    2887.259950607111
    """
    if abs(lat1) >= 90 or abs(lat2) >= 90 or not all(map(inbounds,
                                                         (lat1, lon1),
                                                         (lat2, lon2))):
        raise ValueError("Latitude or longitude out of bounds.")
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    # Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    theta = 2 * np.arcsin(np.sqrt(a))
    dist = radius*theta
    return dist


# STATISTICS
def histogram(roi, bins, hmin=None, hmax=None, skew=False, verbose=False,
              *args, **kwargs):
    """
    Return histogram, bins of histogram computed on ROI. See np.histogram for
    full usage and optional parameters. Set verbose=True to print a summary of
    the statistics.
    """
    roi_notnan = roi[~np.isnan(roi)]
    roi_valid = roi_notnan[hmin <= roi_notnan <= hmax]
    hist, bins = np.histogram(roi, bins=bins, hmin=hmin, hmax=hmax, *args,
                              **kwargs)
    ret = [hist, bins]
    output = 'Histogram with {} pixels total, \
              {} pixels in [hmin, hmax] inclusive, \
              {} nan pixels excluded, \
              {} bins'.format(len(roi), len(roi_valid), len(roi_notnan),
                              len(bins))
    if skew:
        skewness = sp.stats.skew(hist)
        ret.append(skewness)
        output += ', {} skewness'.format(skewness)
    if verbose:
        print(output)
    return ret


# Ejecta profile stats
def fit_exp(x, y, PLOT_EXP=False):
    """
    Return an exponential that has been fit to data using scipy.curvefit().
    If plot is True, plot the data and the fit.
    """
    def exp_eval(x, a, b, c):
        """
        Return exponential of x with a,b,c parameters.
        """
        return a * np.exp(-b * x) + c

    def plot_exp(x, y, exp):
        pass  # TODO: implement this

    try:
        p_opt, cov = opt.curve_fit(exp_eval, x, y)
    except:
        RuntimeError
        return None
    if PLOT_EXP:
        plot_exp(x, y, exp_eval(x, *p_opt))
    return p_opt


def fit_gauss(data, PLOT_GAUSS=False):
    """
    Return parameters of a Gaussian that has been fit to data using least
    squares fitting. If plot=True, plot the histogram and fit of the data.
    """
    def gauss(x, p):
        """
        Return Gaussian with mean = p[0], standard deviation = p[1].
        """
        return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

    def errorfn(p, x, y):
        """Compute distance from gaussian to y."""
        gauss(x, p) - y

    def plot_gauss(bins, n, gauss):
        pass  # TODO: implement this

    data = data[data > 0]
    n, bins = np.histogram(data, bins='fd', density=True)
    p0 = [0, 1]  # initial parameter guess
    p, success = opt.leastsq(errorfn, p0[:], args=(bins[:-1], n))
    if PLOT_GAUSS:
        plot_gauss(bins, n, gauss(bins, p))
    return p


def fit_pow(xdata, ydata, p0, plot=False, cid=''):
    """
    Return a power law curve fit of y using a least squares fit.
    """
    def residuals(p, x, y):
        return ydata - pow_eval(p, x)

    def pow_eval(p, x):
        return p[0] + p[1] * (x**p[2])

    pfinal, success = opt.leastsq(residuals, p0, args=(xdata, ydata))
    xarr = np.arange(xdata[0], xdata[-1], 0.1)
    return xarr, pow_eval(pfinal, xarr)


def fit_pow_linear(xdata, ydata, p0, plot=False, cid=''):
    """
    Return a power law curve fit of y by converting to linear data using
    logarithms and then performing a linear least squares fit.
    """
    def fit_line(p, x):
        return p[0] + p[1] * x

    def residuals(p, x, y):
        return ydata - fit_line(p, x)

    def pow_eval(x, amp, index):
        return amp * (x**index)

    logx = np.log(xdata)
    logy = np.log(ydata)
    pfinal, success = opt.leastsq(residuals, p0, args=(logx, logy))

    return pow_eval(xdata, pfinal[0], pfinal[1])

#
# if __name__ == "__main__":
#    import doctest
#    doctest.testmod()
