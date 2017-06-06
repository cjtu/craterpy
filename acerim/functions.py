# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:15:23 2017

@author: Christian
"""
import numpy as np
#import acerim.acestats as acestats


######################## ACERIM FUNCTIONS ##############################
def computeStats(cdf, ads, stats=None, craters=None):
    """Return a CraterDataFrame object with chosen statistics from stats on 
    craters in cdf using data in ads.
    
    Parameters
    ---------- 
    ads : AceDataset object
        Contains the image data used to compute stats.
    cdf : CraterDataFrame object
        Contains the crater locations and sizes to locate regions of interest
        from aceDS.
    stats : Array-like of str
        Indicates the stat names to compute. Stat functions are given in 
        acestats.py. Use print(acestats.list_all to list valid stats.
    craters : array-like
        Indicies of craters in craterDF to compute stats on. If None, 
        computes stats on all craters in craterDF.

    Returns
    -------
    CraterDataFrame
        Includes craters which stats were computed on, with stats included
        as new columns.
    """
    # Get stat functions to compute from acestats
    if not stats:
        stats = acestats._listStats()
    elif not all(stats in acestats._listStats):
        raise ValueError('One or more of stats not in acestats.py')
    stat_functions = acestats._getFunctions(stats)
    if not craters:
        craters = cdf.index
    elif not all(cdf.isin(craters)):
        raise ValueError('One or more of craters is not in cdf')
    retdf = cdf.loc[craters] # Initialize return Dataframe
    for stat in stats:
        retdf[stat] = retdf.index # Add stat columns to retdf
    for cid in craters:
        # Get lat, lon, rad and compute roi for current crater
        lat = cdf.loc[cid, cdf.latcol]
        lon = cdf.loc[cid, cdf.loncol]
        rad = cdf.loc[cid, cdf.radcol]
        roi = ads.getROI(lat, lon, rad)
        if roi is None:
            raise RuntimeError('Unable to open roi at ({}N, {}E)'.format(lat,lon))
#        roi = roi[(roi > ads.dmin) & (roi < ads.dmax)]
        for stat, function in stat_functions:
            retdf.loc[cid, stat] = function(roi)
    return retdf

######################### ROI manipulation ################3
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
    if not center[0]: # Center circle on center of roi
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
    degwidth = m2deg(rad, aceds.calc_mpp(lat), aceds.ppd)
    degheight = m2deg(rad, aceds.calc_mpp(), aceds.ppd)
    return ellipse_mask(roi, degwidth, degheight)


def crater_ring_mask(roi, aceds, lat, lon, rmin, rmax):
    """
    """ 
    rmax_degheight = m2deg(rmax, aceds.calc_mpp(), aceds.ppd)
    rmax_degwidth = m2deg(rmax, aceds.calc_mpp(lat), aceds.ppd)
    rmin_degheight = m2deg(rmin, aceds.calc_mpp(), aceds.ppd)
    rmin_degwidth = m2deg(rmin, aceds.calc_mpp(lat), aceds.ppd)
    outer = ellipse_mask(roi, rmax_degwidth, rmax_degheight) 
    inner = ellipse_mask(roi, rmin_degwidth, rmin_degheight)
    return outer*~inner

    
########################### Geo CALCULATIONS ###############################
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
    
def m2deg(distance, mpp, ppd):
    """Return distance converted from meters to degrees."""
    return distance/(mpp*ppd)


def deg2pix(dist,ppd):
    """Return distance converted from degrees to pixels."""
    return int(dist*ppd)  


def getInd(value, array):
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


#%% STATISTICAL 
#import numpy as np
#import scipy.optimize as opt
#import helper_functions as hf

def fitExp(x,y,PLOT_EXP=False):
    """
    Return an exponential that has been fit to data using scipy.curvefit(). 
    If plot is True, plot the data and the fit. 
    """     
    def expEval(x,a,b,c):
        """
        Return exponential of x with a,b,c parameters.
        """
        return a * np.exp(-b*x) + c      
      
    try:
        p_opt, cov = opt.curve_fit(expEval,x,y)
    except:
        RuntimeError
        return None
    if PLOT_EXP:
        hf.plot_exp(x,y,expEval(x,*p_opt))
    return p_opt


def fitGauss(data,PLOT_GAUSS=False):
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
    n,bins = np.histogram(data,bins='fd',density=True) 
    p0 = [0,1] #initial parameter guess
    errfn = lambda p,x,y: gauss(x,p) - y
    p, success = opt.leastsq(errfn, p0[:],args=(bins[:-1],n))
    if PLOT_GAUSS:
        hf.plot_gauss(bins,n,gauss(bins,p))
    return p
    
    
def fitPow(xdata,ydata,p0,plot=False,cid=''):
    """
    Return a power law curve fit of y using a least squares fit.
    """
    def residuals(p,x,y):
        return ydata - powEval(p,x)
    
    def powEval(p,x):
        return p[0] + p[1]*(x**p[2])

    pfinal,success = opt.leastsq(residuals,p0,args=(xdata,ydata))
    xarr = np.arange(xdata[0],xdata[-1],0.1)
    return xarr,powEval(pfinal,xarr)
    
    
def fitPowLin(xdata,ydata,p0,plot=False,cid=''):
    """
    Return a power law curve fit of y by converting to linear data using 
    logarithms and then performing a linear least squares fit.
    """
    def fitLine(p,x):
        return p[0] + p[1]*x        
        
    def residuals(p,x,y):
        return ydata - fitLine(p,x)
    
    def powEval(x,amp,index):
        return amp*(x**index)
        
    logx = np.log(xdata)
    logy = np.log(ydata)
    pfinal,success = opt.leastsq(residuals,p0,args=(logx,logy))
    
    return powEval(xdata,pfinal[0],pfinal[1])
    
#%%
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    