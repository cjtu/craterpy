# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:15:23 2017

@author: Christian
"""
import numpy as np
import acerim.acestats as acestats


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


########################### IMAGE CALCULATIONS ###############################
def m2deg(distance,mpp,ppd):
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
    >>> 3.141592653589793
    """
    return theta * (np.pi / 180)

def cropROI(roi, mask):
    """Crop the roi with the provided shape"""
    return roi * mask

def circleMask(roi, r, cx=None, cy = None, encl=False):
    """
    Return boolean array the same size as roi with soecified circle of radius r (in pixels), centered at 
    cx, cy. If no cx, cy are provided, the circle is drawn at the centre. 
    """
    pass

def getCmask(xind, yind, radius, array):
    """Retun circular mask array of True iff array index is in circle."""
    nx, ny = array.shape
    y, x = np.ogrid[-xind:nx-xind, -yind:ny-yind]
    cmask = x*x + y*y <= radius*radius
    return cmask

def greatcircdist(lat1, lon1, lat2, lon2, radius):
    """
    Return great circle distance between two points on a spherical body.
    Uses Haversine formula for great circle distances.
    
    >>> greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
    2887.259950607111
    """
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

#def computeStats(radius, roi, stats):
#    """Return dict of computed statistics on current roi."""
#    roi = roi[(roi > 0) & (roi < 1)]
#    if roi.any(): # if roi is not empty, compute stats
#        median = np.median(roi)
#        pct95 = np.percentile(roi, 95)
#        # Append results 
#        if 'radii' in stats:
#            stats['radii'].append(radius)
#            stats['median'].append(median)
#            stats['pct95'].append(pct95)
#        else: # Initialize if this is the first set of stats computed
#            stats['radii'] = [radius]
#            stats['median'] = [median]
#            stats['pct95'] = [pct95]
#    return stats

    
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
    