from __future__ import division
"""
Created on Thu Jun 23 13:36:34 2016

@author: christian
"""
import numpy as np
import scipy.optimize as opt
import helper_functions as hf

# GLOBALS #
RMOON = 1737.400 #km
#%%
def computeStats(radius, roi, stats):
    """Return dict of computed statistics on current roi."""
    roi = roi[(roi > 0) & (roi < 1)]
    if roi.any(): # if roi is not empty, compute stats
        median = np.median(roi)
        pct95 = np.percentile(roi, 95)
        # Append results 
        if 'radii' in stats:
            stats['radii'].append(radius)
            stats['median'].append(median)
            stats['pct95'].append(pct95)
        else: # Initialize if this is the first set of stats computed
            stats['radii'] = [radius]
            stats['median'] = [median]
            stats['pct95'] = [pct95]
    return stats

def getSfd(cdict, cIDs, norm=1, mode='fullmoon'):
    """Return size-frequency distribution of cIDs"""
    clons = np.array([cdict[cid].lon for cid in cIDs])
    diams = np.array([cdict[cid].diam for cid in cIDs])
    clons[clons > 180] -= 360
    if mode == 'nearside':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-180 <= clons[i] < -90) or (90 <= clons[i] < 180)]
        norm /= 2
    elif mode == 'farside':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-90 <= clons[i] < 90)]
        norm /= 2
    elif mode == 'leading':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-180 <= clons[i] < 0)]
        norm /= 2
    elif mode == 'trailing':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (0 <= clons[i] < 180)]
        norm /= 2             
    area = norm*4*np.pi*(RMOON**2) # norm*Amoon (area of region of moon)
    bins = np.zeros(len(diams))
    bins = np.sort(diams)
    hist, bins = np.histogram(diams, bins=bins)  
    sfd_diams = bins[:-1]
    norm_hist = (hist/area)
    sizefreq = np.cumsum(norm_hist[::-1])[::-1]
    return sfd_diams, sizefreq
    
    
def fitExp(x,y,PLOT_EXP=False):
    """
    Return an exponential that has been fit to data using scipy.curvefit(). 
    If plot is True, plot the data and the fit. 
    """                
    try:
        p_opt, cov = opt.curve_fit(expEval,x,y)
    except:
        RuntimeError
        return None
    if PLOT_EXP:
        hf.plot_exp(x,y,expEval(x,*p_opt))
    return p_opt

    
def expEval(x,a,b,c):
    """Return exponential of x with a,b,c parameters"""
    return a * np.exp(-b*x) + c  
    

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
    
    
    
    