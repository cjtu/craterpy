"""This file contains simple statistical functions that can be used to quickly
characterize CraterRois (e.g. by using the ejecta_stats or ejecta_profile_stats
functions in stats.py).

All functions must take a numpy 2D array and return a numerical value that can
be input into a table, DataFrame, or written to a csv.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats


def size(roi):
    """Return the number of elements in roi"""
    return roi.size


def mean(roi):
    """Return the mean of roi"""
    return np.mean(roi)


def median(roi):
    """Return the median (50/50) of roi"""
    return np.median(roi)


def mode(roi):
    """Return the mode of roi"""
    return scipy.stats.mode(roi, None)[0][0]


def std(roi):
    """Return the standard deviation of roi"""
    return np.std(roi, ddof=1)


def maximum(roi):
    """Return maximum pixel value in roi"""
    return np.max(roi)


def minimum(roi):
    """Return minimum pixel value in roi"""
    return np.min(roi)


def q1(roi):
    """Return the first quartile value (25/75) of roi"""
    return np.percentile(roi, 25)


def q3(roi):
    """Return the third quartile (75/25) value of roi"""
    return np.percentile(roi, 75)


def pct95(roi):
    """Return the 95th percentile (95/5) value of roi"""
    return np.percentile(roi, 95)

# TODO: area
