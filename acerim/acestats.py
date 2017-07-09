# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:59:59 2017

@author: Christian
"""
import numpy as np
import scipy as sp
import inspect

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





# Protected Functions
def _listStats():
    """
    Get a list of the names of all non protcted functions from acestats. 
    Protected functions start with '_'.
    
    Returns
    -------
    List of string of names of functions in this module.
    """
    from acerim import acestats
    all_func = np.array(inspect.getmembers(acestats, inspect.isfunction))
    stat_func = all_func[np.where([a[0][0]  != '_' for a in all_func])]
    return stat_func[:,0]

def _getFunctions(stats):
    """
    Get functions from this module according to stats. If stats_list is
    undefined, gets all functions from this module, excluding protected 
    functions.
    
    Returns
    -------
    Array containing 2 element lists of function names and corresponding functions. 
        E.g. array( ['function name', <function>], ['func2 name'], <func2>)    
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