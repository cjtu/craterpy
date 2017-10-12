import inspect
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
from craterpy import quickstats as qs
from craterpy import masking, plotting


# quickstats helpers
def _list_quickstats():
    """Return names of all functions in quickstats."""
    quickstats = inspect.getmembers(qs, inspect.isfunction)
    return [q[0] for q in quickstats]


def _get_quickstats_functions(statlist=None):
    """Return specified functions from quickstats (default: all).

    Parameters
    ----------
    statlist : str or list of str
        Must be names of functions defined in quickstats.py.

    Returns
    -------
    stat_funcs : list of list
        List of 2 element pairs of function names and functions.
        E.g. array( ['func name 1', <func1>], ['func name 2', <func2>], ...)

    Examples
    --------
    >>> _getquickstats_functions(['mean', 'median'])
    >>> [['mean', <function>], ['median', <function>]]
    """
    qs_list = _list_quickstats()
    if not statlist:
        statlist = qs_list
    elif isinstance(statlist, str):  # or isinstance(statlist, basestring):
        statlist = [statlist]
    invalid_stats = [stat for stat in statlist if stat not in qs_list]
    if invalid_stats:  # TODO: define custom Exception
        raise ValueError('The following stats are not defined ' +
                         'in quickstats.py: {}'.format(invalid_stats))
    return [[stat, qs.__dict__[stat]] for stat in statlist]


# Main craterpy stat functions
def ejecta_profile_stats(cdf, cds, ejrad=2, rspacing=0.25, stats=None,
                         plot_roi=False, vmin=None, vmax=None, strict=False,
                         plot_stats=False, savepath=None, timer=False, dt=60):
    """Compute stat profiles across ejecta blankets by computing stats in a
    series of concentric rings beginning at the crater rim and extending to
    ejrad crater radii from the crater centre. Each ring is rspacing thick and
    the crater floor is excluded. Output is a dictionary of cdf.index values to
    pandas.DataFrame of stats containing the computed stats as columns and
    ring radius as rows.

    Parameters
    ----------

    Returns
    -------
    stat_df : dict of (index: pandas.DataFrame)
        Dictionary of stats with cdf index as keys and pandas.DataFrame of
        statistics as values (see example).

    Example
    -------
    >>> compute_ejecta_profile_stats(my_cdf, my_ads, 2, 0.25, stats=['mean',
                                    'median','maximum'],savepath='/tmp/')

    /tmp/index1_ejecta_stats.csv
        radius,mean,median,maximum
        1,     55,  45,    70
        1.25,  50,  40,    65
        1.5,   35,  35,    45
        1.75,  30,  33,    42

    """
    if timer:
        from timeit import default_timer as timer
        Tstart, Tnow = timer(), timer()
        count = 1  # count craters
    if savepath:
        import os.path as p
        savename = p.join(savepath, '{}_ejpstats.csv')
    if stats is None:
        stats = _list_quickstats()
    stat_dict = {}
    for ind in cdf.index:  # Main CraterDataFrame loop
        lat = cdf.loc[ind, cdf.latcol]
        lon = cdf.loc[ind, cdf.loncol]
        rad = cdf.loc[ind, cdf.radcol]
        try:
            roi, extent = cds.get_roi(lat, lon, rad, ejrad, get_extent=True)
            ring_array = np.arange(1, ejrad+rspacing, rspacing)  # inner radius
            ring_darray = ring_array*rad  # convert inner radius to km
            stat_df = pd.DataFrame(index=ring_array[:-1], columns=stats)
            for i in range(len(ring_array)-1):  # Loop through radii
                rad_inner = ring_darray[i]
                rad_outer = ring_darray[i+1]
                mask = masking.crater_ring_mask(cds, roi, lat, lon, rad_inner,
                                                rad_outer)
                roi_masked = mask_where(roi, ~mask)
                filtered_roi = filter_roi(roi_masked, vmin, vmax, strict)
                roi_notnan = filtered_roi[~np.isnan(filtered_roi)]
                for stat, function in _get_quickstats_functions(stats):
                    stat_df.loc[ring_array[i], stat] = function(roi_notnan)
                if plot_roi:
                    cds.plot_roi(filtered_roi, extent=extent)
            stat_dict[ind] = stat_df
            if plot_stats:
                plotting.plot_ejecta_stats(stat_df)
            if savepath:
                stat_df.to_csv(savename.format(ind))
        except ImportError as e:  # Catches and prints out of bounds exceptions
            print(e, ". Skipping....")
        if timer:  # Print a status update approx every dt seconds
            if (timer() - Tnow > dt) or (count == len(cdf)):
                update = 'Finished crater {} out of {}'.format(count, len(cdf))
                elapsed = timer()-Tstart
                update += '\n  Time elapsed: \
                          {}h:{}m:{}s'.format(int(elapsed//3600),
                                              int((elapsed % 3600)//60),
                                              round((elapsed % 3600) % 60, 2))
                print(update)
                Tnow = timer()
            count += 1
    return stat_dict


def ejecta_stats(cdf, cds, ejrad=2, stats=None, plot=False, vmin=None,
                 vmax=None, strict=False):
    """Compute the specified stats from acestats.py on a circular ejecta ROI
    extending ejsize crater radii from the crater center. Return results as
    CraterDataFrame with stats appended as extra columns.

    Parameters
    ----------
    cdf : CraterDataFrame object
        Contains the crater locations and sizes to locate ROIs in aceDS. Also
        form the basis of the returned CraterDataFrame
    cds : CraterpyDataset object
        CraterpyDataset of image data used to compute stats.
    ejrad : int, float
        The radius of ejecta blanket measured in crater radii from the
        crater center to the ejecta extent.
    stats : Array-like of str
        Indicates the stat names to compute. Stat functions must be defined in
        acestats.py. Default: all stats in acestats.py.
    plot : bool
        Plot the ejecta ROI.
    vmin : int, float
        The minimum valid image pixel value. Filter all values lower.
    vmax : int, float
        The maximum valid image pixel value. Filter all values higher.
    strict : bool
        How strict the (vmin, vmax) range is. If true, exclude values <= vmin
        and values >= vmax, if they are specified.

    Returns
    -------
    CraterDataFrame
        Same length as cdf with stats included as new columns.
    """
    # If stats and index not provided, assume use all stats and all rows in cdf
    if stats is None:
        stats = _list_quickstats()
    # Initialize return CraterDataframe with stats as individual columns
    ret_cdf = cdf
    for stat in stats:
        ret_cdf.loc[:, stat] = ret_cdf.index
    # Main computation loop
    for i in cdf.index:
        # Get lat, lon, rad and compute roi for current crater
        lat = cdf.loc[i, cdf.latcol]
        lon = cdf.loc[i, cdf.loncol]
        rad = cdf.loc[i, cdf.radcol]
        roi, extent = cds.get_roi(lat, lon, rad, ejrad, get_extent=True)
        mask = masking.crater_ring_mask(cds, roi, lat, lon, rad, rad*ejrad)
        roi_masked = mask_where(roi, ~mask)  # ~mask keeps the ring interior
        filtered_roi = filter_roi(roi_masked, vmin, vmax, strict)
        roi_notnan = filtered_roi[~np.isnan(filtered_roi)]  # Collapses to 1D
        for stat, function in _get_quickstats_functions(stats):
            ret_cdf.loc[i, stat] = function(roi_notnan)
        if plot:  # plot filtered and masked roi
            cds.plot_roi(filtered_roi, extent=extent)
    return ret_cdf


# Other statistics
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
