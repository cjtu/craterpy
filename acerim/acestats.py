# -*- coding: utf-8 -*-
"""
This file contains statistical functions that can be used to quickly
generate statistics on ROIs (e.g. by using the ejecta_stats or
ejecta_profile_stats functions in acefunctions.py).

Additional desired statistical functions can be added to this file by
following the naming convention used here:

    def statname(roi_array):
        '''What this stat function does'''
        return statistics(roi_array)

Each function should take a single numpy array as an arugument and return a
single value.

The private functions are:

    _listStats(): return names of all non-protected functions in this file

    _getFunctions(stats): return array of pairs of function names and functions
                            as specified by stats

Non-statistical functions in this file must be private (begin with "_").
"""

from __future__ import division, print_function, absolute_import
import inspect
import numpy as np


# Statistical Functions
def maximum(roi):
    """Return maximum pixel value in roi"""
    return np.max(roi)


def mean(roi):
    """Return the mean of roi"""
    return np.mean(roi)


def median(roi):
    """Return the median (50/50) of roi"""
    return np.median(roi)


def minimum(roi):
    """Return minimum pixel value in roi"""
    return np.min(roi)


def pct95(roi):
    """Return the 95th percentile (95/5) value of roi"""
    return np.percentile(roi, 95)


def q1(roi):
    """Return the first quartile value (25/75) of roi"""
    return np.percentile(roi, 25)


def q3(roi):
    """Return the third quartile (75/25) value of roi"""
    return np.percentile(roi, 75)

# num pix
# area
# std


# Private Functions (must begin with "_")
def _listStats():
    """
    Return list of the names of all non-private functions from acestats.
    Private functions are excluded by their leading '_'.
    """
    from acerim import acestats
    all_func = np.array(inspect.getmembers(acestats, inspect.isfunction))
    stat_func = all_func[np.where([a[0][0] != '_' for a in all_func])]
    return stat_func[:, 0]


def _getFunctions(stats):
    """
    Return functions from this module according to stats. If stats is
    undefined, return all functions from this module, excluding private
    functions.

    Returns
    -------
    List of lists containing 2 element pairs of function names and functions.
        E.g. array( ['func name 1', <func1>], ['func name 2', <func2>])
    """
    from acerim import acestats
    if isinstance(stats, str):
        stats = [stats]
    invalid_stats = [stat for stat in stats if stat not in _listStats()]
    if invalid_stats:
        raise ValueError('The following stats are not defined in acestats.py: '
                         + str(invalid_stats))
    all_func = inspect.getmembers(acestats, inspect.isfunction)
    stat_func = []
    for i, func in enumerate(all_func):
        if func[0] in stats:
            stat_func.append(all_func[i])
    return stat_func
