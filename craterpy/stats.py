"""Compute stats on CraterRoi objects."""
import inspect
import os.path as p
from timeit import default_timer
import numpy as np
import pandas as pd
from craterpy import quickstats as qs
from craterpy import helper as ch
from craterpy import masking
from craterpy.roi import CraterRoi

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
    if invalid_stats:
        raise ValueError(
            "The following stats are not defined "
            + "in quickstats.py: {}".format(invalid_stats)
        )
    return [[stat, qs.__dict__[stat]] for stat in statlist]


# Main craterpy stat functions
def ejecta_profile_stats(
    df,
    cds,
    ejrad=2,
    rspacing=0.25,
    stats=None,
    plot_roi=False,
    vmin=None,
    vmax=None,
    strict=False,
    savepath=None,
    timer=False,
    dt=60,
):
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
    >>> compute_ejecta_profile_stats(my_df, my_cds, 2, 0.25, stats=['mean',
                                    'median','maximum'],savepath='/tmp/')

    /tmp/index1_ejecta_stats.csv
        radius,mean,median,maximum
        1,     55,  45,    70
        1.25,  50,  40,    65
        1.5,   35,  35,    45
        1.75,  30,  33,    42

    """
    if timer:
        Tstart, Tnow = default_timer(), default_timer()
        count = 1  # count craters
    if savepath:
        savename = p.join(savepath, "{}_ejpstats.csv")
    if stats is None:
        stats = _list_quickstats()
    stat_dict = {}
    latcol, loncol, radcol = ch.get_crater_cols(df)
    for i in df.index:  # Main DataFrame loop
        lat, lon, rad = df.loc[i, latcol], df.loc[i, loncol], df.loc[i, radcol]
        try:
            croi = CraterRoi(cds, lat, lon, rad, ejrad)
            ring_array = np.arange(
                1, ejrad + rspacing, rspacing
            )  # inner radius
            ring_darray = ring_array * rad  # convert inner radius to km
            stat_df = pd.DataFrame(index=ring_array[:-1], columns=stats)
            for j in range(len(ring_array) - 1):  # Loop through radii
                rad_inner = ring_darray[j]
                rad_outer = ring_darray[j + 1]
                mask = masking.crater_ring_mask(
                    cds, croi, lat, lon, rad_inner, rad_outer
                )
                ejecta_roi = croi.mask(~mask).filter(vmin, vmax, strict)
                ejecta_roi = ejecta_roi[~np.isnan(ejecta_roi)]
                for stat, function in _get_quickstats_functions(stats):
                    stat_df.loc[ring_array[j], stat] = function(ejecta_roi)
                if plot_roi:
                    ejecta_roi.plot()
            stat_dict[i] = stat_df
            if savepath:
                stat_df.to_csv(savename.format(i))
        except ImportError as e:  # Catches and prints out of bounds exceptions
            print(e, ". Skipping....")
        if timer:  # Print a status update approx every dt seconds
            if (default_timer() - Tnow > dt) or (count == len(df)):
                update = "Finished crater {} out of {}".format(count, len(df))
                elapsed = default_timer() - Tstart
                update += "\n  Time elapsed: \
                          {}h:{}m:{}s".format(
                    int(elapsed // 3600),
                    int((elapsed % 3600) // 60),
                    round((elapsed % 3600) % 60, 2),
                )
                print(update)
                Tnow = default_timer()
            count += 1
    return stat_dict


def compute_stats(
    df,
    cds,
    stats=None,
    plot=False,
    save=False,
    prefix="",
    vmin=float("-inf"),
    vmax=float("inf"),
    strict=False,
    maskfunc=None,
    mask_out=False,
    ejrad=None,
    buffer=1,
    verbosity=1,
):
    """Computes stats on craters in df with image data from cds."""
    # If stats and index not provided, assume use all stats and all rows in cdf
    if stats is None:
        stats = _list_quickstats()
    # Initialize return Dataframe with stats as individual columns
    ret_df = df.copy()
    for stat in stats:
        ret_df.loc[:, stat] = ret_df.index
    latcol, loncol, radcol = ch.get_crater_cols(df)
    # Main computation loop
    for i in df.index:
        # Get lat, lon, rad and compute roi for current crater
        lat, lon, rad = df.loc[i, latcol], df.loc[i, loncol], df.loc[i, radcol]
        try:
            # print(lat, lon, rad)
            croi = CraterRoi(cds, lat, lon, rad, ejrad)
            croi.filter(vmin, vmax, strict)
            if maskfunc:
                if maskfunc == "crater":
                    mask = masking.crater_floor_mask(croi, buffer)
                elif maskfunc == "ejecta":
                    mask = masking.crater_ring_mask(
                        croi, rad * buffer, rad * ejrad
                    )
                croi.mask(mask, mask_out)
            data_arr = croi.roi[~np.isnan(croi.roi)]  # Collapses to 1D
            for stat, function in _get_quickstats_functions(stats):
                ret_df.loc[i, stat] = function(data_arr)
            if plot:  # plot filtered and masked roi
                croi.plot()
            if save:  # save roi to csv: prefix_lat_lon_rad.csv
                fname = "{}croi_{:.3f}_{:.3f}_{:.3f}.csv".format(
                    prefix, lat, lon, rad
                )
                croi.save(fname)

        except ValueError as e:
            for stat, _ in _get_quickstats_functions(stats):
                ret_df.loc[i, stat] = np.nan
            if verbosity:
                print(
                    "Exception '{}' skipping crater at ({}, {}) \
                       with radius {}".format(
                        e, lat, lon, rad
                    )
                )
            continue
    return ret_df


def crater_stats(
    df,
    cds,
    stats=None,
    plot=False,
    vmin=float("-inf"),
    vmax=float("inf"),
    strict=False,
    verbosity=1,
):
    """Computes stats on all craters in df using image data from cds

    Crater latitude, longitude, and radius are read in from df. All stats are
    computed assuming a circular crater centered on (lat, lon). Stats must be
    present in quickstats.py.

    Parameters
    ----------
    df : pandas.DataFrame object
        Contains the crater locations and sizes to locate ROIs in aceDS. Also
        form the basis of the returned DataFrame
    cds : CraterpyDataset object
        CraterpyDataset of image data used to compute stats.
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
    DataFrame
        Copy of df with stats appended as new columns.
    """
    return compute_stats(
        df,
        cds,
        stats,
        plot,
        vmin,
        vmax,
        strict,
        maskfunc="crater",
        mask_out=True,
        verbosity=verbosity,
    )


def ejecta_stats(
    df,
    cds,
    ejrad=2,
    buffer=1,
    stats=None,
    plot=None,
    save=None,
    prefix=None,
    vmin=float("-inf"),
    vmax=float("inf"),
    strict=False,
    verbosity=1,
):
    """Compute the specified stats from acestats.py on a circular ejecta ROI
    extending ejsize crater radii from the crater center. Return results as
    DataFrame with stats appended as extra columns.

    Parameters
    ----------
    df : pandas.DataFrame object
        Contains the crater locations and sizes to locate ROIs in aceDS. Also
        form the basis of the returned DataFrame
    cds : CraterpyDataset object
        CraterpyDataset of image data used to compute stats.
    ejrad : int, float
        The radius of ejecta blanket measured in crater radii from the
        crater center to the ejecta extent.
    buffer : int, float
        Extra buffer around crater rim to mask (multiplicative factor of
        crater radius). Defaults to 1 (no extra buffer).
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
    DataFrame
        Copy of df with stats appended as new columns.
    """
    return compute_stats(
        df,
        cds,
        stats,
        plot,
        save,
        prefix,
        vmin,
        vmax,
        strict,
        maskfunc="ejecta",
        mask_out=True,
        ejrad=ejrad,
        buffer=buffer,
        verbosity=verbosity,
    )

    # If stats and index not provided, assume use all stats and all rows in cdf
    # if stats is None:
    #     stats = _list_quickstats()
    # # Initialize return Dataframe with stats as individual columns
    # ret_df = df.copy()
    # for stat in stats:
    #     ret_df.loc[:, stat] = ret_df.index
    # latcol, loncol, radcol = ch.get_crater_cols(df)
    # # Main computation loop
    # for i in df.index:
    #     # Get lat, lon, rad and compute roi for current crater
    #     lat, lon, rad = df.loc[i, latcol], df.loc[i, loncol], df.loc[i, radcol]
    #     croi = CraterRoi(cds, lat, lon, rad, ejrad)
    #     mask = masking.crater_ring_mask(croi, rad, rad*ejrad)
    #     croi.filter(vmin, vmax, strict)
    #     croi.mask(~mask)
    #     data_arr = croi.roi[~np.isnan(croi.roi)]  # Collapses to 1D
    #     for stat, function in _get_quickstats_functions(stats):
    #         ret_df.loc[i, stat] = function(data_arr)
    #     if plot:  # plot filtered and masked roi
    #         croi.plot()
    # return ret_df


# Other statistics
def histogram(
    roi, bins, hmin=None, hmax=None, skew=False, verbose=False, **kwargs
):
    """
    Return histogram, bins of histogram computed on ROI. See np.histogram for
    full usage and optional parameters. Set verbose=True to print a summary of
    the statistics.
    """
    roi_notnan = roi[~np.isnan(roi)]
    roi_valid = roi_notnan[hmin <= roi_notnan <= hmax]
    hist, bins = np.histogram(roi, bins, range=(hmin, hmax), **kwargs)
    ret = [hist, bins]
    output = "Histogram with {} pixels total, \
              {} pixels in [hmin, hmax] inclusive, \
              {} nan pixels excluded, \
              {} bins".format(
        len(roi), len(roi_valid), len(roi_notnan), len(bins)
    )
    if skew:
        skewness = pd.DataFrame(roi.flatten).skew
        ret.append(skewness)
        output += ", {} skewness".format(skewness)
    if verbose:
        print(output)
    return ret
