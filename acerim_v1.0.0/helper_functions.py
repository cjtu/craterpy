from __future__ import division
"""
Created on Mon Jun 20 12:16:52 2016

@author: christian
"""
import matplotlib.pyplot as plt
import numpy as np
import statistical_functions as sf
import ace_settings as s

# GLOBALS #
s.init()
#%% Acerim Functions
def getAcerim(c, metric):
    """
    Return the radius of anomalous data (acerim) around the crater.
    """
    radii = np.array(c.stats['fit_'+metric+'_xarr'])
    fitExp = c.stats['fit_'+metric+'_shift']
    if fitExp[0] <= 0:
        return 0.
    efold = (fitExp[0])/(np.e)
    efold_ind = (np.abs(fitExp - efold)).argmin()
    return radii[efold_ind]


def getAcedomain(c, thresh, SHOW=False):
    """
    Return the distance (in crater radii) to where the gradient of the median
    falls below a threshold. This gives the point at which the median reaches
    the background value.
    """
    dx = 0.01 
    p = c.stats['pfit_median'] # parameters of exponential fit
    aceDomain = None
    xmax = 30
    x = np.arange(1, xmax, dx) # in crater radii
    exp_fit = sf.expEval(x,*p)
    deriv_exp = np.gradient(exp_fit, dx)
    aceDom_ind = np.abs(deriv_exp-thresh).argmin()
    if aceDom_ind == (len(x)-1): # If aceDom is as long as x, fails to get a domain
        aceDomain = 0
    else:
        aceDomain = x[aceDom_ind]

    if SHOW: # plot exp and deriv of exp vs. x
        f,ax=plt.subplots(2, sharex=True)
        ax[0].plot(aceDomain,exp_fit[aceDom_ind],'o')
        ax[0].plot(x, exp_fit)
        ax[0].set_title('Exponential fit of Median')
        ax[1].plot(aceDomain,deriv_exp[aceDom_ind],'o')
        ax[1].plot(x, deriv_exp)
        ax[1].set_title('Derivative of exponential')
        ax[1].set_xlabel('Crater Radii')
    return aceDomain

def getAcerange(c, rim_ind):
    """
    Return the range through which the values drop off with distance from the
    crater. Acerange is the median value at the rim - bg value for the region 
    (assumed by the value the exponential fit converges to). Rim is taken to be
    the rim_ind-th shell past the crater rim (e.g. rim_ind=0 for the shell just beyond 
    the rim, rim_ind=1 for the next, etc.)
    """
    rim_value = c.stats['fit_median'][rim_ind] # median value of the first shell past rim
    bg_value = c.stats['pfit_median'][2] # additive constant in exp fit
    return rim_value - bg_value    

def getSlope(c,x1=1,x2=2):
    """
    Return slope between 1 and 2 crater radii and the height.
    """
    radii = np.array(c.stats['radii'])
    medians = c.stats['median']
    x2ind = int(np.abs(radii-x2).argmin())
    x1ind = int(np.abs(radii-x1).argmin())
    m = (medians[x2ind] - medians[x1ind])/(x2-x1)
    b = medians[0]
    return m,b

def getMetrics(c, AceDS, PLOT_ROI=False): 
    """ Compute and return the metrics for the specified crater.

    Arguments:
    ----------
    c: Crater object
        Current crater with attributes c.lat, c.lon, c.rad.
    AceDS: AceDataset object
        Contains a gdal.Dataset with extent, resolution and plotting attributes.
    nr: int
        The number of crater radii used to specify a ROI (region of interest)
        around crater. The ROI is a square that extends out to nr from the 
        center of the crater (i.e. square of side length 2*nr)
    dr: float
        The step size between consecutive shells in the acerim calculation
    n: int
        The number of shells to include in the moving window average.

    Return:
    -------
    Acerim: float
        The computed run-out radius of the crater (meters).

    Raises:
    -------
    Exception
        Target crater is not found in data or is too close to the edge.
    """        
    try:
        roi = AceDS.getROI(c, s.RMAX, PLOT_ROI)
    except ImportError as e:
        #print(str(e)) # Print import error (out of bound info)
        return False

    # Compute crater statistics with a moving & expanding shell average 
    shell_radii = np.arange(1, s.DR+s.RMAX, s.DR)  # Shell array in crater radii
    shell_pixradii = m2pix(shell_radii*c.rad, AceDS.mpp) # Shell array (pixels)
    c.stats = movingShell(c, shell_radii, shell_pixradii, roi, s.WSIZE, AceDS, s.PLOT_ROINF, s.PLOT_SHELLS) 
    if not c.stats:
        return False
    else:
        return True

def getFits(c, cid, metrics):
    """
    """
    success = False
    radii = np.array(c.stats['radii'])
    for m in metrics:
        metric = np.array(c.stats[m])
        nonglassy_ind = np.where((radii < 1.55) | (radii > 1.85))
        r = radii[nonglassy_ind] # exclude glassy region in fit stats
        ngmetric = metric[nonglassy_ind]
        p = sf.fitExp(r,ngmetric)
        c.stats['pfit_'+m] = p
        if p is None: # No exponential fit    
            xarr = radii
            fit = metric # No exponential so use original datapoints 
            yshift = metric[-1]
            fit_shift = metric - yshift # Subtract to shift plots to 0                  
        else:         
            xarr = np.arange(radii[0], radii[-1], 0.01)
            fit = sf.expEval(xarr, p[0],p[1],p[2])
            yshift = p[2]
            fit_shift = fit - yshift 
            if m == 'median':
                success = True 

        c.stats[m+'_shift'] = metric - yshift
        c.stats['fit_'+m+'_xarr'] = xarr
        c.stats['fit_'+m] = fit
        c.stats['fit_'+m+'_shift'] = fit_shift
    return success
        




#%% Dataset Helpers
def getCmask(xind, yind, radius, array):
    """Retun circular mask array of True iff array index is in circle."""
    nx, ny = array.shape
    y, x = np.ogrid[-xind:nx-xind, -yind:ny-yind]
    cmask = x*x + y*y < radius*radius
    return cmask
    
    
def movingShell(c, shell_radii, shell_pixradii, roi, wsize, AceDS, PLOT_ROINF=False, PLOT_SHELLS=False):
    """
    """
    cx, cy =  roi.shape[0]//2, roi.shape[1]//2 # crater center x,y indices
    stats = {}
    for i in range(len(shell_radii)-wsize):
        midrad = np.round((shell_radii[i+(wsize)]+shell_radii[i])/2, 3)
        maxrad = shell_pixradii[i+wsize]
        minrad = shell_pixradii[i]
       
        shell_mask = getCmask(cx, cy, maxrad, roi) *\
                        ~getCmask(cx, cy, minrad, roi)
        shell_window = roi*shell_mask
        stats = sf.computeStats(midrad, shell_window, stats)
        
        if PLOT_SHELLS:
            rmax = shell_radii[-1]
            extent = (-rmax, rmax, -rmax, rmax)
            plot_shells(shell_window, c, midrad, extent, AceDS)   
            
    if PLOT_ROINF: # Exclude crater floors
        roi_nofloor = ~getCmask(cx, cy, shell_pixradii[0], roi)*roi
        rmax = shell_radii[-1]
        plot_shells(roi_nofloor, c, c.rad, c.extent, AceDS)
    return stats    
    
def readIDs(fname,skiprows=0):
    with open(fname) as f:
        for i in range(skiprows):
            f.readline()
        txt = f.readlines()
    if len(txt) > 0:
        return txt[0].split(',') #Split IDs into list (comma sep), remove final comma
    else:
        return []

def saveIDs(fname, IDs, i):
    with open(fname,'w') as f:
        f.write('last index, i={}; Number of IDs={}\n'.format(i,len(IDs)))
        if len(IDs) > 0:        
            for cid in IDs[:-1]:                
                f.write("{},".format(cid)) # Print ids in 
            f.write("{}".format(IDs[-1])) # Omit final comma
    print('Progress saved to {}'.format(fname),flush=True)  
    return
  
#%% Plotting Functions
def plot_exp(x,y,expeval):
    plt.figure("Exp")
    plt.plot(x,y,'r+',label='Original Data')
    plt.plot(x,expeval,'b-',label='Exp curve_fit')

def plot_gauss(bins, n, gauss):
    fig,ax = plt.subplots(num="Gauss")
    #plt.xlim((np.min(data),np.max(data)))
    #plt.hist(data,bins='auto',density=True)
    #n=n/max(n)
    widths = bins[1:]-bins[:-1]
    ax.bar(bins[:-1],n,widths)    
    plt.plot(bins,gauss)
    plt.show()
    
def plot_roi(roi, c, AceDS):
    plt.figure("ROI",figsize=(8,8))
    plt.imshow(roi, cmap=AceDS.cmap, extent=c.extent, vmin=AceDS.pltmin, vmax=AceDS.pltmax)
    plt.title('Crater {0}, Radius {1}km'.format(c.name, c.diam))
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.show()

def plot_shells(shell, c, midrad, extent, AceDS): 
    plt.figure("Shells")
    plt.imshow(shell, cmap=AceDS.cmap, extent=extent, vmin=AceDS.pltmin, vmax=AceDS.pltmax)
    plt.title('Crater {0}, Shell of Radius {1:.4f}'.format(c.name, midrad))
    plt.xlabel('Distance (crater radii)')
    plt.ylabel('Distance (crater radii)')
    plt.show()    
    

#%% Conversion Functions
def deg2pix(dist,ppd):
    """Return distance converted from degrees to pixels."""
    return int(dist*ppd)  
    
    
def m2deg(distance,mpp,ppd):
    """Return distance converted from meters to degrees."""
    return distance/(mpp*ppd)


def m2pix(distance,mpp):
    """Return distance converted from meters to pixels."""
    temp = [int(round(d/mpp)) for d in distance]
    return np.array(temp)


def pix2m(distance,mpp):
    """Return distance converted from pixels to meters."""
    return distance*mpp
 
def getInd(value, array):
    """Return closest index (rounded down) of a value from sorted array."""
    ind = np.abs(array-value).argmin() 
    return int(ind)    
    
def getXYind(lat,lon,latarr,lonarr):
    """Return raster x,y index from latarr,lonarr."""
    xind = np.abs(lonarr-lon).argmin()
    yind = np.abs(latarr-lat).argmin()  
    return (int(xind),int(yind))

def projfwd(lat,lon):
    """Return x,y map coordinates from lat,lon in the simple cylindrical projection """
    lat0 = 0 #standard parallel 
    x = lon*np.cos(lat0)
    y = lat
    return (x,y)
    
    
def projinv(x,y):
    """Return lat,lon map coordinates from x,y in the simple cylindrical projection"""
    lat0 = 0 #standard parallel   
    lat = y
    lon = x/np.cos(lat0)
    return(lat,lon) 