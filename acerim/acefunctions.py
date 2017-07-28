# -*- coding: utf-8 -*-
"""
This file contains several functions used in the ACERIM workflow. They are 
sorted into different sections:

    ACERIM FUNCTIONS - top-level functions
    PLOTTING - plot figures
    ROI MANIPULATION - manipulate 2D arrays of image data
    GEOSPATIAL CALCULATIONS - manipulate geospatial data
    STATISTICS - compute statistics and fitting functions

For usage, see sample/tutorial.rst.
"""
from acerim import acestats as acs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


######################### ACERIM FUNCTIONS ####################################
def ejecta_profile_stats(cdf, ads, ejrad=2, rspacing=0.25, stats=None, 
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
        Dictionary of stats using the cdf index as keys and the pandas.DataFrame
        of statistics as values (see example)
    
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
        count = 1 # count craters
    if savepath:
        import os.path as p
        savename = p.join(savepath,'{}_ejpstats.csv')
    if stats is None:
        stats = acs._listStats()
    stat_dict = {}
    for ind in cdf.index: # Main CraterDataFrame loop
        lat = cdf.loc[ind, cdf.latcol]
        lon = cdf.loc[ind, cdf.loncol]
        rad = cdf.loc[ind, cdf.radcol]
        try:
            roi, extent = ads.get_roi(lat, lon, rad, ejrad, get_extent=True)
            ring_array = np.arange(1, ejrad+rspacing, rspacing) # inner ring radius [crater radii]
            ring_darray = ring_array*rad # inner ring radius [km]
            stat_df = pd.DataFrame(index=ring_array[:-1], columns=stats)
            for i in range(len(ring_array)-1): # Ejecta ring radii loop  
                rad_inner = ring_darray[i] # inner radius of this ejecta ring 
                rad_outer = ring_darray[i+1] # outer radius of this ejecta ring
                mask = crater_ring_mask(ads, roi, lat, lon, rad_inner, rad_outer)
                roi_masked = mask_where(roi, ~mask)
                filtered_roi = filter_roi(roi_masked, vmin, vmax, strict)
                roi_notnan = filtered_roi[~np.isnan(filtered_roi)]
                for stat, function in acs._getFunctions(stats):
                    stat_df.loc[ring_array[i], stat] = function(roi_notnan)                
                if plot_roi:
                    ads.plot_roi(filtered_roi, extent=extent) 
            stat_dict[ind] = stat_df
            if plot_stats:
                plot_ejecta_stats(stat_df)
            if savepath:
                stat_df.to_csv(savename.format(ind))
        except ImportError as e: # Catches and prints out of bounds exceptions
            print(e, ". Skipping....")
        if timer: # Print a status update approx every dt seconds
            if (timer() - Tnow > dt) or (count == len(cdf)):
                update = 'Finished crater {} out of {}'.format(count, len(cdf))
                elapsed = timer()-Tstart
                update += '\n  Time elapsed: {}h:{}m:{}s'.format(int(elapsed//3600),
                              int((elapsed%3600)//60), round((elapsed%3600)%60,2))
                print(update)
                Tnow = timer()
            count +=1
    return stat_dict
    

def ejecta_stats(cdf, ads, ejrad=2, stats=None, plot=False, vmin=None, 
                 vmax=None, strict=False):
    """Compute the specified stats from acestats.py on a circular ejecta ROI
    extending ejsize crater radii from the crater center. Return results as a 
    CraterDataFrame the same length as cdf with stats appended as extra columns.    
    
    Parameters
    ---------- 
    cdf : CraterDataFrame object
        Contains the crater locations and sizes to locate ROIs in aceDS. Also
        form the basis of the returned CraterDataFrame
    ads : AceDataset object
        AceDataset of image data used to compute stats.
    ejrad : int, float
        The radius of ejecta blanket measured in crater radii from the 
        crater center to the ejecta extent.
    stats : Array-like of str
        Indicates the stat names to compute. Stat functions must be defined in 
        acestats.py. Default: all stats in acestats.py. 
    plot : bool
        Plot the ejecta ROI.
    vmin : int, float
        The minimum valid image data value. Set this to filter all values lower.
    vmax : int, float
        The maximum valid image data value. Set this to filter all values higher.
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
        stats = acs._listStats()
    # Initialize return CraterDataframe with stats as individual columns
    ret_cdf = cdf
    for stat in stats:
        ret_cdf[stat] = ret_cdf.index
    # Main computation loop
    for i in cdf.index:
        # Get lat, lon, rad and compute roi for current crater
        lat = cdf.loc[i, cdf.latcol]
        lon = cdf.loc[i, cdf.loncol]
        rad = cdf.loc[i, cdf.radcol]
        roi, extent = ads.get_roi(lat, lon, rad, ejrad, get_extent=True)
        mask = crater_ring_mask(ads, roi, lat, lon, rad, rad*ejrad)
        roi_masked = mask_where(roi, ~mask) # ensure it is ~mask to keep the ring interior
        filtered_roi = filter_roi(roi_masked, vmin, vmax, strict)
        roi_notnan = filtered_roi[~np.isnan(filtered_roi)] # Collapse to 1D non-nan array
        for stat, function in acs._getFunctions(stats):
            ret_cdf.loc[i, stat] = function(roi_notnan)
        if plot: # plot filtered and masked roi
            ads.plot_roi(filtered_roi, extent=extent)
    return ret_cdf
  
   
################################ PLOTTING #####################################
def plot_roi(ads, roi, figsize=((8,8)), extent=None, title='ROI', vmin=None,
             vmax=None, cmap='gray', **kwargs):
    """
    Plot roi 2D array. 
    
    If extent, cname and cdiam are supplied, the axes will display the 
    lats and lons specified and title will inclue cname and cdiam.
    
    Parameters
    ----------
    ads : AceDataset object
        The parent dataset of roi.
    roi : 2D array
        The roi from ads to plot.
    figsize : tuple
        The (length,width) of plot in inches.
    extent : array-like
        The [minlon, maxlon, minlat, maxlat] extents of the roi in degrees.
    title : str
        Title of the roi.
    vmin : int, float
        Minimum pixel data value for plotting.
    vmax : int, float
        Maximum pixel data value for plotting.
    cmap : str
        Color map name for plotting. Must be a valid colorbar in
        matplotlib.cm.cmap_d. See help(matplotlib.cm) for full list.
        
    Other parameters
    ----------------
    **kwargs : object
        Additional keyword arguments to be passed to the imshow function. See 
        help(matplotlib.pyplot.imshow) for more info.
    """
    plt.figure("ROI", figsize=figsize)
    plt.imshow(roi, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.show()


########################### ROI MANIPULATION ##################################
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
    nanmask = ~np.isnan(roi) # build nanmask with pre-existing nans, if any
    nanmask[nanmask] &= roi[nanmask] > vmax # Add values outside of (vmin,vmax) to nanmask
    nanmask[nanmask] &= roi[nanmask] < vmin
    if strict: # if strict, also exclude values equal to vmin, vmax
        nanmask[nanmask] &= roi[nanmask] == vmax
        nanmask[nanmask] &= roi[nanmask] == vmin
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
    mask = np.array(np.ones(ndarray.shape)) # Same shape array of ones
    mask[np.where(condition)] = np.nan 
    return ndarray * mask


def circle_mask(roi, radius, center=(None,None)):
    """
    Return boolean array of True inside circle of radius at center.
    
    >>> roi = np.ones((3,3))
    >>> masked = circle_mask(roi, 1)
    >>> masked[1,1]
    True
    >>> masked[0,0]
    False
    """
    if not center[0]: # Center circle on center of roi
        center = np.array(roi.shape)/2 - 0.5
    cx, cy = center
    width, height = roi.shape
    x = np.arange(width) - cx
    y = np.arange(height).reshape(-1,1) - cy
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
    if not center[0]: # Center ellipse on center of roi
        center = np.array(roi.shape)/2 - 0.5
    cx, cy = center    
    width, height = roi.shape
    y, x = np.ogrid[-cx:width-cx, -cy:height-cy]
    return (x*x)/(a*a) + (y*y)/(b*b) <= 1


def ring_mask(roi, rmin, rmax, center=(None,None)):
    """
    Return boolean array of True in a ring from rmin to rmax radius around 
    center. Returned array is same shape as roi.
    """
    inner = circle_mask(roi, rmin, center)
    outer = circle_mask(roi, rmax, center)
    return outer*~inner


def crater_floor_mask(aceds, roi, lat, lon, rad):
    """
    """
    pixwidth = m2pix(rad, aceds.calc_mpp(lat))
    pixheight = m2pix(rad, aceds.calc_mpp())
    return ellipse_mask(roi, pixwidth, pixheight)


def crater_ring_mask(aceds, roi, lat, lon, rmin, rmax):
    """
    """ 
    rmax_pixheight = m2pix(rmax, aceds.calc_mpp())
    rmax_pixwidth = m2pix(rmax, aceds.calc_mpp(lat))
    rmin_pixheight = m2pix(rmin, aceds.calc_mpp())
    rmin_pixwidth = m2pix(rmin, aceds.calc_mpp(lat))
    outer = ellipse_mask(roi, rmax_pixwidth, rmax_pixheight) 
    inner = ellipse_mask(roi, rmin_pixwidth, rmin_pixheight)
    return outer * ~inner


######################### GEOSPATIAL CALCULATIONS #############################
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
    if not all(map(inbounds,(lat1,lon1),(lat2,lon2))) or abs(lat1)==90 or abs(lat2)==90:
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


############################### STATISTICS ####################################
#import numpy as np
#import scipy.optimize as opt
#import helper_functions as hf

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
      
    try:
        p_opt, cov = opt.curve_fit(expEval, x, y)
    except:
        RuntimeError
        return None
    if PLOT_EXP:
        hf.plot_exp(x, y, exp_eval(x, *p_opt))
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
   
    data = data[data > 0]       
    n, bins = np.histogram(data, bins='fd', density=True) 
    p0 = [0, 1] #initial parameter guess
    errfn = lambda p, x, y: gauss(x, p) - y
    p, success = opt.leastsq(errfn, p0[:],args=(bins[:-1], n))
    if PLOT_GAUSS:
        hf.plot_gauss(bins, n, gauss(bins, p))
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
    
    
def fit_pow_linear(xdata,ydata,p0,plot=False,cid=''):
    """
    Return a power law curve fit of y by converting to linear data using 
    logarithms and then performing a linear least squares fit.
    """
    def fit_line(p, x):
        return p[0] + p[1] * x        
        
    def residuals(p, x, y):
        return ydata - fitLine(p,x)
    
    def pow_eval(x,amp,index):
        return amp * (x**index)
        
    logx = np.log(xdata)
    logy = np.log(ydata)
    pfinal, success = opt.leastsq(residuals, p0, args=(logx, logy))
    
    return pow_eval(xdata, pfinal[0], pfinal[1])
    
#%%
#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()
    
