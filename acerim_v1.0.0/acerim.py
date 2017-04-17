"""
*DEPRECATED*
Created on Mon Mar 28 00:12:13 2016

Automated Crater Ejecta Region of Interest Mapper (ACERIM)
Package for automatically determining the run-out distance of ejecta for
all specified craters in a given dataset.

Find fresh craters using the the following algorithm:
    1) Take a region of RMAX crater radii around a crater
    2) Eliminate the crater floor (circle at c.lat, c.lon with radius c.rad)
    3) Compute the median data value in concentric shells of increaing radius
        from c.rad to RMAX with DR thickness.
    4) Fit a negative exponential to approximate this median(r) function. For 
        fresh craters, there is a steep drop off, while old craters are flat.
    5) Measure the ACERANGE (the data median at the rim - the value at inf)
        and the ACEDOMAIN (where the data falls close enought to its value at inf)
    6) If the ACERANGE is above the ACERNG_THLD and ACEDOMAIN is above the 
        ACEDOM_THLD then keep the crater as an ACECRATER
    7) Verify the ACECRATERS by visual inspection and eliminate clear anomalies
        (on another crater's ejecta, superimposed crater, other anomaly)
    8) Pass through final list and sort them into fresh, maybe, and not-fresh

TODO: FIX the 180 degree overlap (stitch tofether image array from before the 
boundary and after)
TODO: Implement Suresh's crater overlap idea. Exclude all craters which are 
fully superimposed on another crater's ejecta AND exclude ejecta of small craters
which lie on top of other crater regions (ensures that omat curve is only from 
crater of interest, and superimposed bright craters won't mess with exp curve)


@author: Christian
"""
import numpy as np
import matplotlib.pyplot as plt
import statistical_functions as sf
import helper_functions as hf
import plotting_functions as pf
import matplotlib.colorbar as cbar
from ace_classes import CraterDict, AceDataset
from timeit import default_timer as timer
Tstart = timer() # Startup Timer
#==============================================================================

# FUNCTIONS #
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
        roi = hf.getROI(c, RMAX, AceDS, PLOT_ROI)
    except ImportError as e:
        #print(str(e)) # Print import error (out of bound info)
        return False

    # Compute crater statistics with a moving & expanding shell average 
    shell_radii = np.arange(1, DR+RMAX, DR)  # Shell array in crater radii
    shell_pixradii = hf.m2pix(shell_radii*c.rad, AceDS.mpp) # Shell array (pixels)
    c.stats = hf.movingShell(c, shell_radii, shell_pixradii, roi, WSIZE, AceDS, PLOT_ROINF, PLOT_SHELLS) 
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
        
#==============================================================================
#%% MAIN  #
def main(cIDs, cdict, AceDS):
    Tstart = timer()
    cIDs = [cid for cid in cIDs if (-50 < cdict[cid].lat < 50)] # cut down OMAT oob craters
    
    #Initialize main vars  
    NID = len(cIDs)
    aceIDs = []
    oobIDs = []
    noexpIDs = []
    fit_success = False    
    print('\nBegin Main {} craters\n'.format(NID))
    
    for i, cid in enumerate(cIDs):  # iterate over copy to allow deletion
        c = cdict[cid]
        metric_success = getMetrics(c, AceDS, PLOT_ROI)         
        if not metric_success:  # If this crater was skipped, remove from cIDs list                       
            #print('No stats for ', c.name, '. Skipping..')
            oobIDs.append(cid)
            continue   
        if FMETRICS:
            fit_success = getFits(c, cid, METRICS)
        if FDMETRICS:
            getFits(c, cid, DMETRICS)           
        # DO FILTERING
        if not fit_success:
            noexpIDs.append(cid)
        else:
            aceDomain = getAcedomain(c, EXPDERIV_THLD)
            aceRange = getAcerange(c, 0)         
            if (aceDomain > ACEDOM_THLD) and (aceRange > ACERNG_THLD):
                #print('{} is an aceCrater!'.format(c.name))
                aceIDs.append(cid)             
            c.stats['ACEDOM'] = aceDomain
            c.stats['ACERNG'] = aceRange
        
        # Print updates and save progress. Scale update interval by size of crater
        # since larger craters take longer to complete.
        if c.diam > 30:
            DU = 50 # Number of craters between updates and saves
        elif c.diam > 20:
            DU = 100
        elif c.diam > 10:
            DU = 500
        elif c.diam > 1:
            DU = 1000
        else:
            DU = 5000
        if i%DU == DU-1:         
            print('Finished crater {} out of {} ({} ace, diam={}km)'.format(i+1,NID,len(aceIDs),c.diam))
            # Display time
            Telapsed= timer()-Tstart
            Hrs = int(Telapsed//3600)
            Mins = int((Telapsed%3600)//60)
            Secs = round((Telapsed%3600)%60,2)           
            print('{} hrs, {} mins and {} secs have passed'.format(Hrs,Mins,Secs))
            # Save acecraters 
            fileout = 'ace_progress.txt'
            hf.saveIDs(fileout, aceIDs, i)
                    
    print('FINISHED MAIN')
    print('{}% aceCraters, {}% NoExp, {}% OutOfBounds'.format(
                  100*len(aceIDs)//i,100*len(noexpIDs)//i,100*len(oobIDs)//i))
    Telapsed= timer()-Tstart
    Hrs = int(Telapsed//3600)
    Mins = int((Telapsed%3600)//60)
    Secs = round((Telapsed%3600)%60,2)           
    print('Total Run Time: {} hrs, {} mins and {} secs'.format(Hrs,Mins,Secs))
    return aceIDs

    
    #==============================================================================

#%% 
def compare(cIDs, cdict, ACEdatasets, folder='figs/'):
    """
    Side-by-side plot of craters specified by cIDs in the datasets in ACEdatasets.
    """
    for i,cid in enumerate(cIDs):
        c = cdict[cid]
        fig, axarr = plt.subplots(nrows=len(ACEdatasets), sharex=False, sharey=True)
        for i,ACEds in enumerate(ACEdatasets):
            try:
                roi = hf.getROI(c, RMAX, ACEds)
            except ImportError as e:
                print(str(e))
                roi = None
            else:
                 im = axarr[i].imshow(roi, cmap=ACEds.cmap, extent=c.extent, 
                                        vmin=ACEds.pltmin, vmax=ACEds.pltmax)
                 axarr[i].set_title(ACEds.name)
                 axarr[i].set_xlabel('Longitude (degrees)')
                 axarr[i].set_ylabel('Latitude (degrees)')
                 cax,kw = cbar.make_axes(axarr[i])
                 plt.colorbar(im, cax=cax, **kw)
                
            plt.setp([a.get_xticklabels() for a in axarr], visible=True)
            fig.suptitle('"{}": Diameter = {}km. Loc: {}\nOMAT freshness: "{}". RA age = {}Mya'.format(c.name, c.diam, c.loc, c.omat, c.age))
            plt.rcParams.update({'axes.titlesize':'large','xtick.labelsize':'x-small', 'ytick.labelsize':'x-small'})
            fig.set_figheight(17.4)
            fig.set_figwidth(10)
            #figname = '{}_{}_comp'.format(cid,'-'.join([ACEds.name for ACEds in ACEdatasets]))
            figname = cid+'.png'            
            plt.savefig(PATH+folder+figname)
            #plt.savefig(PATH+'foo.png',bbox_inches='tight',dpi=300) #Highres raster
            #plt.savefig(PATH+'bar.pdf',bbox_inches='tight') #Vectorized but blurs image 

            
#%%            
def verify(cdict, cIDs, AceDS, fpath):
    """
    """
    vtimer = timer()
    cIDs = np.array(cIDs)
    def save2txt(i):    
        vfIDs = cIDs[np.where(vf)] # compute ids from bool arrays
        vmfIDs = cIDs[np.where(vmf)]
        vnfIDs = cIDs[np.where(vnf)]
        anomIDs = cIDs[np.where(anom)]
        hf.saveIDs(fpath+'vfIDs.txt', vfIDs, i)
        hf.saveIDs(fpath+'vmfIDs.txt', vmfIDs, i)
        hf.saveIDs(fpath+'vnfIDs.txt', vnfIDs, i)
        hf.saveIDs(fpath+'vanomIDs.txt', anomIDs, i)
        return vfIDs, vmfIDs, vnfIDs, anomIDs
    
    N = len(cIDs)
    vf = [False]*N # Verified Fresh
    vmf = [False]*N # Verified Maybe Fresh
    vnf = [False]*N # Verified Not Fresh
    anom = [False]*N # Anomalous craters 
    i = 0
    savetimer = timer()
    while i < N:
        cid = cIDs[i]
        print('Crater {}/{}'.format(i,N))
        try:
            hf.getROI(cdict[cid], RMAX, AceDS, PLOT=True)
        except ImportError as e:
            print(str(e))
            print('Crater {0} out of bounds. Skipping...'.format(cdict[cid].name))
                  
        inpt = ''
        while not inpt:
            inpt = input('Fresh/Maybe/NotFresh: type (1/2/3). Anomalous: type (0).\n "b" go back 1 crater. "q" to save & quit\n')
            if inpt == '1':
                vf[i]= True
            elif inpt == '2':
                vmf[i] = True
            elif inpt == '3':
                vnf[i] = True
            elif inpt == '0':
                anom[i] = True
            elif inpt == 'b' and i > 0:
                vf[i-1],vmf[i-1],vnf[i-1],anom[i-1] = (False,False,False,False)
                i -= 2
            elif inpt == 'q':
                return save2txt(i) # save and quit
            else:
                print('INVALID INPUT')
                inpt = ''
        t_min = (timer()-savetimer)
        if t_min > 30: # save every 2 mins then reset timer
            save2txt(i)
            savetimer = timer()
        i += 1 # increment index
    print('Finished {} craters in {:.2} mins'.format(i,(timer()-vtimer)/60))
    return save2txt(i)
        

#==============================================================================
#%% USER DEFINED PARAMS
# ACERIM PARAMS #
RMAX = 12 # Max radial extent to consider (in crater radii from center)
DR = 0.2 #0.25  # Shell thickness in (crater radii)
WSIZE = 1 # Size of moving window in number of shells
EXPDERIV_THLD = -0.001 # Threshold for when derivative of exp reaches 0, see getAcedomain
ACEDOM_THLD = 2.7 # Acerim domain low threshold
ACERNG_THLD = 0.05 # Ace range low threshold

# FLAGS #
PLOT_ROI = False  # Set to True to plot ROI of each crater
PLOT_ROINF = False # Plot roi with no floors (excludes crater interior)
PLOT_SHELLS = False # Plot each ring/shell roi of each crater
BG_GAUSS = False  # Set to True to fit a Gaussian to the background data
FMETRICS = True
FDMETRICS = False  # ['fit_'+d for d in DMETRICS]

#%%
# Exclude following craters from future calcs (out of bounds, no exponential fit, etc)
EXCLUDE = False
if EXCLUDE:
    OOB_KEYS = ['C-20.54N-178.41E', 'C-8.03N179.49E', 'C-10.18N179.07E', 'C43.65N-178.42E', 'C2.13N178.44E', 'C7.17N177.83E', 'C9.53N178.55E', 'C4.64N-179.89E', 'C25.10N177.66E', 'C42.59N-177.06E', 'C5.11N-179.97E', 'C17.28N178.20E', 'C-36.72N177.43E', 'C14.03N178.06E', 'C-13.62N177.55E', 'C13.95N179.81E', 'C11.01N-177.74E', 'C37.19N177.84E', 'C12.50N176.98E', 'C0.42N178.83E', 'C-4.13N-179.87E', 'C42.23N-178.81E', 'C28.67N176.32E', 'C-46.86N-17.63E', 'C29.82N-178.96E', 'C-13.16N177.10E', 'C46.93N-151.14E', 'C19.53N179.61E', 'C24.77N-178.59E', 'C-2.16N-179.90E', 'C20.12N-176.12E', 'C36.06N-177.01E', 'C23.63N179.29E', 'C46.13N-155.66E', 'C-8.22N-178.59E', 'C-46.44N103.57E', 'C-46.03N170.78E', 'C8.76N-178.82E', 'C-10.52N-178.41E', 'C-45.95N-22.21E', 'C-45.70N-87.37E', 'C46.38N-170.36E', 'C26.97N179.43E', 'C41.06N177.22E', 'C17.83N177.90E', 'C-16.55N-177.57E', 'C3.83N178.84E', 'C12.68N-176.45E', 'C-45.99N53.98E', 'C-20.23N178.03E', 'C-46.37N158.90E', 'C45.97N-158.68E', 'C-46.07N0.94E', 'C-21.36N-177.59E', 'C-46.48N12.50E', 'C45.84N-79.33E', 'C-43.23N-175.47E', 'C-45.92N10.06E', 'C-46.12N85.57E', 'C13.32N179.62E', 'C46.53N128.92E', 'C33.74N177.69E', 'C-46.83N31.42E', 'C39.96N-178.25E', 'C7.00N179.11E', 'C-46.38N24.73E', 'C-46.23N63.50E', 'C-45.78N14.35E', 'C10.44N176.70E', 'C-46.71N52.17E', 'C43.23N175.71E', 'C-45.16N113.05E', 'C-45.01N-7.93E', 'C-19.86N179.28E', 'C-46.17N87.34E', 'C45.02N-94.66E', 'C-44.77N122.53E', 'C45.14N108.47E', 'C-44.92N-65.45E', 'C-1.29N178.75E', 'C-46.53N-17.99E', 'C16.75N176.66E', 'C-6.57N-177.43E', 'C-44.55N160.84E', 'C45.56N124.70E', 'C46.04N-153.72E', 'C-17.02N178.95E', 'C-11.50N-175.32E', 'C45.22N-169.38E', 'C44.78N-142.10E', 'C-46.27N36.59E', 'C-45.63N111.37E', 'C22.18N-179.34E', 'C3.13N-174.33E', 'C45.71N-161.29E', 'C-44.13N18.29E', 'C-3.91N-176.04E', 'C45.64N-150.58E', 'C-12.46N178.89E', 'C32.97N175.39E', 'C-22.41N176.72E', 'C-7.65N175.24E', 'C45.75N-40.22E', 'C-9.91N-174.07E', 'C31.49N175.49E', 'C-7.77N-176.29E', 'C15.64N176.74E', 'C-13.77N-174.76E', 'C-46.66N160.38E', 'C46.57N140.40E', 'C0.88N-177.34E', 'C45.07N28.21E', 'C-46.09N56.90E', 'C26.57N-177.07E', 'C-44.80N-159.04E', 'C44.82N-155.50E', 'C-18.72N176.44E', 'C-44.94N-105.06E', 'C-12.34N-175.89E', 'C46.83N-120.34E', 'C44.43N69.69E', 'C-46.47N71.73E', 'C45.00N-138.71E', 'C44.87N-159.83E', 'C-2.25N-175.68E', 'C-43.20N61.57E', 'C16.28N173.80E', 'C43.65N93.01E', 'C44.95N-129.67E', 'C-43.49N51.73E', 'C42.52N-137.99E', 'C-46.43N-5.04E', 'C-46.27N122.82E', 'C-46.43N-96.43E', 'C-44.21N-39.62E', 'C37.39N175.03E', 'C45.75N117.84E', 'C28.50N-173.99E', 'C33.58N-179.55E', 'C45.75N88.68E', 'C45.01N51.57E', 'C43.85N141.32E', 'C-10.06N-172.23E', 'C37.25N-177.55E', 'C-45.53N156.06E', 'C-43.71N4.29E', 'C44.33N127.55E', 'C6.89N176.70E', 'C25.36N180.00E', 'C28.01N-177.10E', 'C-16.56N-178.77E', 'C46.12N-145.06E', 'C-43.38N-3.88E', 'C42.66N-135.30E', 'C-44.64N44.40E', 'C41.40N-85.46E', 'C45.16N-173.38E', 'C-0.52N-175.03E', 'C41.34N-163.73E', 'C-19.94N175.19E', 'C41.70N-124.91E', 'C-42.77N84.41E', 'C-19.68N-176.78E', 'C44.17N162.69E', 'C-44.75N-82.03E', 'C41.29N-107.83E', 'C-4.15N-178.91E', 'C-46.58N-10.74E', 'C-45.41N54.05E', 'C22.54N174.73E', 'C-46.64N129.59E', 'C19.74N172.90E', 'C-46.92N-147.98E', 'C-43.56N-7.49E', 'C44.24N145.29E', 'C41.98N-154.63E', 'C-46.55N-116.72E', 'C42.42N60.81E', 'C-46.40N67.80E', 'C41.97N-128.87E', 'C3.79N-175.63E', 'C41.04N-78.08E', 'C-44.57N-170.80E', 'C-43.46N127.60E', 'C40.79N129.50E', 'C-42.31N63.41E', 'C-44.76N165.42E', 'C-42.57N-139.56E', 'C-23.67N-179.38E', 'C-42.11N2.49E', 'C46.66N66.07E', 'C-39.48N-17.82E', 'C6.65N174.26E', 'C44.68N179.46E', 'C41.54N170.15E', 'C-13.76N175.57E', 'C-41.07N-1.52E', 'C20.86N174.82E', 'C-3.34N177.46E', 'C45.15N72.98E', 'C46.54N-83.13E', 'C-46.98N150.51E', 'C-40.82N130.68E', 'C-43.06N105.70E', 'C46.82N39.21E', 'C-41.71N164.94E', 'C-42.45N8.75E', 'C-44.86N-139.89E', 'C-4.55N-171.42E', 'C46.13N69.00E', 'C-41.45N-7.87E', 'C44.27N16.23E', 'C-41.23N-33.52E', 'C-41.54N50.98E', 'C-13.50N-179.87E', 'C-38.67N162.43E', 'C21.90N-175.29E', 'C-25.77N-175.08E', 'C-42.29N-177.72E', 'C39.63N-97.28E', 'C-45.41N63.27E', 'C-42.01N95.98E', 'C-40.57N-77.45E', 'C40.28N153.25E', 'C38.95N-114.63E', 'C12.37N173.10E', 'C38.04N118.69E', 'C-24.91N177.34E', 'C-45.83N-20.76E', 'C42.33N-103.44E', 'C46.78N-103.73E', 'C37.41N-118.90E', 'C0.56N171.80E', 'C-43.76N76.09E', 'C-42.75N41.84E', 'C-22.93N-170.27E', 'C-45.66N127.43E', 'C42.11N154.84E', 'C-30.17N174.01E', 'C-39.28N-9.44E', 'C43.39N136.98E', 'C-42.61N169.22E', 'C7.55N-167.28E', 'C-44.98N16.81E', 'C25.00N170.89E', 'C42.19N-141.93E', 'C-40.42N43.37E', 'C-40.75N-168.65E', 'C43.66N-121.83E', 'C-43.30N-11.22E']
    NOEXP_KEYS = ['C26.87N56.96E', 'C-23.53N39.08E', 'C-22.18N14.88E', 'C-19.05N58.24E', 'C21.97N34.12E', 'C26.87N56.96E', 'C-23.53N39.08E', 'C-22.18N14.88E', 'C-19.05N58.24E', 'C21.97N34.12E', 'C28.60N-74.58E', 'C-44.49N-40.70E', 'C41.00N-45.54E', 'C-16.23N12.06E', 'C5.97N-2.35E', 'C25.34N-73.76E', 'C-43.20N19.76E', 'C24.67N2.27E', 'C-39.25N-24.48E', 'C-36.25N22.35E', 'C-24.89N33.55E', 'C-12.69N-46.64E', 'C35.47N-86.51E', 'C-45.20N-24.45E', 'C-26.95N38.22E', 'C-13.29N-30.08E', 'C-26.72N30.54E', 'C-20.52N1.97E', 'C3.92N40.51E', 'C26.96N-55.16E', 'C-37.88N62.93E', 'C-42.52N29.00E', 'C-27.40N8.99E', 'C-15.89N159.65E', 'C-41.62N-28.93E', 'C-8.42N-77.56E', 'C-11.91N81.18E', 'C23.58N4.51E', 'C34.14N52.11E', 'C-46.70N-38.77E', 'C-46.69N-15.13E', 'C-16.18N109.06E', 'C-23.27N-55.19E', 'C-43.64N-24.38E', 'C-26.29N-81.47E', 'C-6.13N18.09E', 'C-1.95N35.86E', 'C-34.88N-25.31E', 'C-11.57N-11.57E', 'C-9.90N2.00E', 'C-37.95N44.44E', 'C-31.90N5.32E', 'C-26.32N-47.41E', 'C-13.83N36.72E', 'C-3.71N39.30E', 'C-45.98N3.11E', 'C8.84N-90.24E', 'C10.41N78.36E', 'C-44.22N49.07E', 'C-42.82N-6.32E', 'C-25.60N68.12E', 'C-2.87N-83.85E', 'C-44.27N3.11E', 'C-30.87N26.57E', 'C-6.20N20.70E', 'C-10.25N-71.64E', 'C-10.13N169.20E', 'C4.55N17.28E', 'C5.56N54.74E', 'C-25.84N-67.36E', 'C-45.28N8.17E', 'C-13.86N66.47E', 'C-4.54N29.78E', 'C35.35N77.71E', 'C-27.88N11.58E', 'C-15.90N25.31E', 'C-3.01N-30.64E', 'C-26.76N106.70E', 'C-35.75N-35.06E', 'C-3.87N71.49E', 'C-42.90N22.32E', 'C-29.40N-51.84E', 'C-27.18N24.07E', 'C-17.09N26.17E', 'C1.73N5.18E', 'C5.13N-71.47E', 'C5.29N-73.85E', 'C-28.85N-43.36E', 'C-1.39N-69.23E', 'C9.07N73.12E', 'C18.66N29.27E', 'C-18.25N41.10E', 'C-41.94N12.53E', 'C-42.66N48.48E', 'C-38.51N58.38E', 'C-32.40N50.16E', 'C-29.48N-66.30E', 'C-45.85N95.69E', 'C-44.08N-46.68E', 'C-42.58N-15.42E', 'C-24.02N-52.83E', 'C20.84N45.03E', 'C31.88N48.50E', 'C-17.64N4.44E', 'C7.14N-36.12E', 'C-22.47N80.90E', 'C4.87N72.27E', 'C-44.67N-2.16E', 'C-28.88N42.39E', 'C-1.32N81.37E', 'C37.46N-60.70E', 'C-37.69N52.99E', 'C-35.70N-17.24E', 'C-26.58N42.41E', 'C17.79N-75.61E', 'C46.58N-51.72E', 'C-34.32N-27.73E', 'C-40.79N38.27E', 'C-20.91N80.94E', 'C-19.08N-79.99E', 'C2.16N-11.59E', 'C-26.74N46.59E', 'C-35.73N28.04E', 'C-31.44N7.68E', 'C-1.81N127.41E', 'C2.36N-36.78E', 'C-14.86N-48.39E', 'C6.19N45.93E', 'C-17.58N62.24E', 'C-36.97N-33.17E', 'C5.89N69.81E', 'C-6.35N-64.88E', 'C-27.51N-3.97E', 'C-35.70N-61.16E', 'C-2.03N46.94E', 'C22.86N-75.40E', 'C3.40N100.64E', 'C-30.94N13.41E', 'C-27.83N-2.00E', 'C-30.85N10.88E', 'C-24.28N-47.75E', 'C-25.74N52.21E', 'C-23.73N92.95E', 'C-22.88N-1.57E', 'C-32.35N65.54E', 'C-30.94N43.91E', 'C0.96N18.78E', 'C-25.99N-28.67E', 'C-15.71N-56.17E', 'C-29.16N25.40E', 'C-22.11N81.78E', 'C27.26N50.66E', 'C-22.41N-80.85E', 'C-7.07N69.88E', 'C-5.76N36.39E', 'C5.57N72.33E', 'C-27.72N59.97E', 'C-12.36N12.26E', 'C5.51N40.21E', 'C-33.54N59.03E', 'C-32.43N0.75E', 'C-26.14N25.44E', 'C-15.97N-89.91E', 'C-1.10N9.38E', 'C43.43N66.14E', 'C-17.00N-57.49E', 'C-16.04N154.97E', 'C-3.58N34.84E', 'C-16.74N-3.88E', 'C-7.72N-71.01E', 'C-36.47N0.62E', 'C7.76N18.47E', 'C-15.90N81.41E', 'C-37.67N0.92E', 'C-32.82N-87.31E', 'C-12.43N105.08E', 'C-37.98N-7.06E', 'C-33.83N-74.24E', 'C0.73N9.79E', 'C-3.00N-1.41E', 'C-5.14N88.97E', 'C46.87N33.06E', 'C-32.58N-14.92E', 'C-30.51N35.90E', 'C-35.71N-32.47E', 'C21.03N-71.11E', 'C22.47N48.53E', 'C-38.34N-26.41E', 'C-34.36N22.66E', 'C-38.05N50.05E', 'C5.90N64.34E', 'C-3.68N36.80E', 'C-33.57N-21.86E', 'C-44.43N62.46E', 'C-27.73N9.45E', 'C-36.76N30.58E', 'C-24.94N16.83E', 'C-32.16N-1.96E', 'C-20.44N5.68E', 'C0.96N-18.76E', 'C45.90N64.24E', 'C-18.57N62.07E', 'C-46.99N29.42E', 'C-40.91N-20.52E', 'C42.90N33.87E', 'C-20.85N37.38E', 'C1.37N-141.91E', 'C24.13N37.05E', 'C19.63N11.68E', 'C-30.50N-45.55E', 'C-32.80N-43.13E', 'C46.52N-15.38E', 'C-30.44N-10.86E', 'C1.41N136.11E', 'C-44.73N-16.17E', 'C9.35N62.43E', 'C-46.10N-6.06E', 'C2.92N69.52E', 'C-35.58N-80.64E', 'C-34.94N-72.54E', 'C-37.52N-19.32E', 'C4.63N57.48E', 'C-33.35N-1.27E', 'C17.43N41.02E', 'C-46.50N4.83E', 'C-12.36N21.50E', 'C-41.15N51.18E', 'C-7.68N37.18E', 'C-17.23N40.13E', 'C-9.66N5.16E', 'C-46.09N43.58E', 'C-12.57N98.40E', 'C22.98N69.09E', 'C10.35N-71.63E', 'C-32.01N42.13E', 'C-7.40N-29.56E', 'C9.56N38.63E', 'C22.34N91.99E', 'C22.57N56.99E', 'C-30.70N162.41E', 'C-43.84N-15.25E', 'C7.71N91.41E', 'C-43.37N-36.83E', 'C-13.42N-59.69E', 'C-31.37N26.28E', 'C-27.05N-52.65E', 'C-45.06N32.22E', 'C-45.12N10.24E', 'C-27.93N-3.53E', 'C22.59N35.56E', 'C-34.20N27.89E', 'C-43.27N-41.08E', 'C43.39N67.85E', 'C-7.70N133.15E', 'C24.36N72.72E', 'C-32.97N-62.71E', 'C-30.30N22.02E', 'C-21.78N-25.69E', 'C21.46N40.23E', 'C-25.56N25.51E', 'C46.12N-68.18E', 'C-36.60N9.63E', 'C-16.04N126.87E', 'C-23.72N18.68E', 'C-8.26N-8.22E', 'C-3.62N31.27E', 'C24.51N35.81E', 'C-24.38N24.35E', 'C-4.41N-86.97E', 'C-14.72N126.47E', 'C-30.85N15.80E', 'C-27.35N20.05E', 'C-40.20N-37.32E', 'C-18.70N-163.29E', 'C0.02N123.81E', 'C33.15N76.87E', 'C15.35N-30.91E', 'C-28.37N166.70E', 'C-3.15N89.07E', 'C9.00N-1.18E', 'C-18.17N-6.56E', 'C-16.71N77.48E', 'C9.58N162.51E', 'C-17.05N7.64E', 'C-13.98N66.72E', 'C-6.23N123.27E', 'C-3.21N-5.19E', 'C10.01N-30.23E', 'C-28.68N-58.56E', 'C-40.68N25.26E', 'C14.34N20.25E', 'C-21.16N-61.50E', 'C-8.02N25.98E', 'C16.13N86.94E', 'C-6.64N-9.81E', 'C-6.50N-2.41E', 'C-14.77N53.37E', 'C-14.61N70.57E', 'C3.06N-72.33E', 'C-35.62N21.76E', 'C-0.97N96.04E', 'C4.57N177.46E', 'C-35.14N-11.46E', 'C-26.14N33.75E', 'C-4.62N144.83E', 'C9.02N75.83E', 'C-27.54N47.14E', 'C-28.46N15.69E', 'C7.25N49.90E', 'C1.19N4.21E', 'C-43.21N25.26E', 'C8.30N122.12E', 'C-20.22N17.98E', 'C-17.80N-97.68E', 'C-31.20N-10.89E', 'C-40.98N-15.40E', 'C16.37N-72.49E', 'C36.12N-86.29E', 'C32.14N38.33E', 'C41.04N46.51E', 'C-0.34N16.75E', 'C28.07N-6.43E', 'C-15.33N-64.65E', 'C1.51N98.30E', 'C-21.12N-62.05E', 'C-19.06N-62.73E', 'C-12.02N97.09E', 'C-37.32N93.83E', 'C-45.01N-62.44E', 'C-8.98N65.74E', 'C2.76N86.96E', 'C-35.90N-57.06E', 'C-20.24N31.02E', 'C6.21N12.53E', 'C5.56N130.47E', 'C28.88N55.06E', 'C-40.91N64.07E', 'C-21.84N14.70E', 'C1.47N68.97E', 'C-40.76N26.00E', 'C-12.67N61.92E', 'C-6.21N163.91E', 'C-13.82N112.21E', 'C-10.62N-116.22E', 'C-12.95N6.38E', 'C-31.20N-0.85E', 'C2.75N18.86E', 'C-7.64N147.91E', 'C-32.08N49.90E', 'C-31.58N157.12E', 'C-23.22N38.96E', 'C-2.66N145.59E', 'C-32.61N130.07E', 'C-13.91N124.09E', 'C-1.70N139.23E', 'C-41.86N-77.56E', 'C-27.98N-11.17E', 'C-25.77N-57.53E', 'C9.64N168.27E', 'C-42.34N-13.66E', 'C-28.97N22.78E', 'C-13.43N88.20E', 'C-3.96N117.61E', 'C-45.97N-30.05E', 'C8.09N83.14E', 'C-38.66N-57.84E', 'C-31.80N-37.63E', 'C-35.11N18.06E', 'C23.51N73.53E', 'C-22.42N44.84E', 'C-13.87N173.46E', 'C-17.20N-71.78E', 'C1.17N-88.30E', 'C-32.55N-64.70E', 'C-20.21N22.27E', 'C-34.69N-25.70E', 'C-26.59N114.75E', 'C-3.51N2.10E', 'C-32.73N9.55E', 'C11.00N50.00E', 'C-37.23N-0.42E', 'C-22.70N51.08E', 'C-9.52N-15.98E', 'C-0.59N174.06E', 'C-21.62N-32.61E', 'C4.16N13.73E', 'C-20.47N39.36E', 'C-5.65N60.14E', 'C-15.14N12.20E', 'C3.99N1.12E', 'C-35.05N39.75E', 'C-25.79N8.89E', 'C-23.27N137.36E', 'C-21.72N-76.86E', 'C-19.44N-62.03E', 'C-25.70N6.52E', 'C-17.61N-57.98E', 'C-7.90N-90.04E', 'C0.80N-69.90E', 'C-37.55N-32.31E', 'C-13.56N-60.15E', 'C-36.08N52.36E', 'C-35.11N-55.34E', 'C20.83N-165.15E', 'C-33.08N42.47E', 'C-46.22N79.25E', 'C-12.23N-7.75E', 'C27.91N-1.27E', 'C-24.04N16.89E', 'C-40.13N62.81E', 'C-3.67N-64.54E', 'C23.17N70.96E', 'C3.91N6.16E', 'C-26.29N159.20E', 'C-16.43N10.75E', 'C9.89N174.23E', 'C-43.11N69.83E', 'C-37.79N-37.88E', 'C-35.64N131.36E', 'C41.12N61.20E', 'C-30.57N43.91E', 'C-5.41N52.90E', 'C-21.08N-27.53E', 'C-28.93N-45.02E', 'C-28.75N20.72E', 'C-38.56N24.44E', 'C-6.05N107.43E', 'C12.26N175.92E', 'C18.98N-77.14E', 'C-15.80N105.57E', 'C18.37N46.55E', 'C-10.81N11.24E', 'C-3.26N133.31E', 'C40.98N58.51E', 'C5.36N80.19E', 'C20.01N106.39E', 'C-30.60N145.63E', 'C-10.79N172.61E', 'C-43.78N9.61E', 'C13.36N74.67E', 'C45.83N20.09E', 'C7.75N-98.02E', 'C-35.38N48.41E', 'C-21.25N1.34E', 'C-34.04N43.46E', 'C-0.75N-69.54E', 'C-9.22N35.42E', 'C-6.04N155.11E', 'C-13.46N-146.42E', 'C-19.76N-45.99E', 'C40.85N-86.71E', 'C-29.71N7.39E', 'C-13.11N85.89E', 'C15.69N-165.86E', 'C-27.90N97.54E', 'C-5.22N-58.42E', 'C-43.99N-13.92E', 'C-26.12N-68.64E', 'C-12.02N116.09E', 'C14.62N-168.44E', 'C-43.86N-63.90E', 'C-18.33N-86.22E', 'C-13.77N23.12E', 'C32.47N-31.98E', 'C-42.47N23.60E', 'C-2.82N64.62E', 'C-2.26N167.83E', 'C-40.72N37.03E', 'C-20.51N150.54E', 'C-19.85N37.37E', 'C-9.14N-62.64E', 'C-45.61N-88.53E', 'C7.66N14.45E', 'C-33.96N118.33E', 'C2.15N66.98E', 'C3.82N-73.51E', 'C-9.46N1.89E', 'C-7.59N3.20E', 'C41.76N7.80E', 'C-32.69N49.39E', 'C-1.90N47.65E', 'C45.19N62.85E', 'C-28.56N-52.04E', 'C20.12N67.89E', 'C-32.31N101.83E', 'C-32.23N-14.94E', 'C1.72N149.54E', 'C4.35N141.54E', 'C-33.19N-19.38E', 'C-38.97N-75.11E', 'C-3.05N20.32E', 'C-25.43N43.91E', 'C-43.74N-52.25E', 'C3.75N-172.68E', 'C-15.94N144.32E', 'C45.11N64.22E', 'C17.06N33.24E', 'C-38.60N69.90E', 'C-6.98N104.52E', 'C-14.69N16.80E', 'C-37.54N84.44E', 'C-31.13N-25.07E', 'C2.85N-68.23E', 'C-36.95N13.28E', 'C-2.73N167.61E', 'C46.34N-88.07E', 'C-36.12N20.53E', 'C7.06N104.96E', 'C33.16N31.02E', 'C-21.78N-98.28E', 'C-19.74N43.39E', 'C8.34N118.74E', 'C-3.99N41.79E', 'C1.46N71.12E', 'C13.32N42.23E', 'C-21.46N9.40E', 'C-44.88N47.49E', 'C-34.50N24.74E', 'C-15.84N66.56E', 'C-46.37N152.27E', 'C6.47N-28.00E', 'C20.90N-36.94E', 'C-8.50N13.17E', 'C-45.23N71.24E', 'C-41.91N6.61E', 'C3.25N18.93E', 'C3.94N15.09E', 'C8.31N119.65E', 'C-41.15N-12.62E', 'C-39.41N-6.20E', 'C-37.99N-24.43E', 'C-3.61N90.42E', 'C-34.73N34.03E', 'C-12.08N15.19E', 'C5.46N44.52E', 'C13.18N52.79E', 'C-24.38N40.69E', 'C23.90N-74.00E', 'C-16.40N76.07E', 'C-16.31N27.78E', 'C-7.53N37.70E', 'C-21.16N6.77E', 'C-12.53N142.67E', 'C-27.43N50.53E', 'C42.44N68.91E', 'C-5.22N-1.09E', 'C7.36N150.02E', 'C-30.63N-15.77E', 'C-28.75N-165.03E', 'C-40.18N44.37E', 'C-26.05N80.65E', 'C-5.39N-71.33E', 'C40.39N-75.47E', 'C-28.36N-22.76E', 'C-29.06N62.94E', 'C-9.76N18.60E', 'C-33.91N-10.17E', 'C4.59N99.34E', 'C12.16N65.72E', 'C-0.89N13.68E', 'C-37.05N-77.48E', 'C-14.59N-15.31E', 'C-27.11N-92.67E', 'C11.80N65.84E', 'C15.73N14.75E', 'C-21.10N18.47E', 'C3.43N101.97E', 'C-18.43N109.86E', 'C-36.71N18.38E', 'C-19.45N97.96E', 'C-15.81N-52.28E', 'C-6.57N-5.61E', 'C19.73N107.26E', 'C29.46N60.16E', 'C-35.46N-39.05E', 'C2.32N131.85E', 'C-23.42N145.40E', 'C16.92N-77.91E', 'C-45.95N-61.10E', 'C-0.88N20.76E', 'C-9.03N39.91E', 'C-14.61N-23.57E', 'C-33.78N55.99E', 'C-27.62N171.36E', 'C30.80N48.57E', 'C-17.53N-65.34E', 'C-11.11N-63.77E', 'C12.21N-26.00E', 'C-13.36N-33.97E', 'C-4.55N-59.40E', 'C-32.73N8.15E', 'C-24.74N41.57E', 'C-22.22N48.83E', 'C-2.57N69.38E', 'C46.33N158.89E', 'C-21.56N22.91E', 'C-28.19N153.83E', 'C-46.74N-15.75E', 'C11.93N168.12E', 'C23.28N-75.30E', 'C-4.33N0.21E', 'C-4.28N34.08E', 'C-27.02N46.97E', 'C-1.27N-71.15E', 'C-9.28N-53.59E', 'C1.40N116.86E', 'C5.46N143.55E', 'C7.07N130.57E', 'C2.05N7.33E', 'C-46.19N42.31E', 'C-8.30N111.04E', 'C-45.42N28.53E', 'C20.68N-76.15E', 'C-31.46N19.28E', 'C-4.90N22.22E', 'C-6.05N66.83E', 'C-31.71N7.36E', 'C-26.44N30.36E', 'C35.68N44.26E', 'C-32.91N-67.46E', 'C-20.60N113.11E', 'C-30.12N-31.71E', 'C-18.71N26.80E', 'C-38.12N-35.01E', 'C1.42N119.69E', 'C-37.94N-33.45E', 'C-19.08N-125.15E', 'C-24.85N-4.49E', 'C11.32N142.67E', 'C-20.48N67.58E', 'C8.27N168.63E', 'C13.18N-20.38E', 'C-20.74N-33.19E', 'C16.52N46.22E', 'C-4.37N140.23E', 'C3.49N116.50E', 'C21.47N42.06E', 'C-20.51N-71.12E', 'C2.63N111.61E', 'C-22.53N-50.02E', 'C-10.37N-5.10E', 'C-26.99N-2.05E', 'C-32.15N85.06E', 'C-2.93N-70.69E', 'C11.51N67.29E', 'C5.10N166.11E', 'C-2.88N6.99E', 'C-22.02N-167.37E', 'C-40.40N28.87E', 'C-25.87N-6.93E', 'C-1.00N16.44E', 'C41.01N169.30E', 'C-3.28N-90.62E', 'C29.06N88.50E', 'C-14.51N16.37E', 'C-2.20N102.92E', 'C29.22N43.54E', 'C-43.87N143.30E', 'C8.84N46.31E', 'C21.73N17.92E', 'C-3.30N121.40E', 'C1.26N147.73E', 'C-45.83N45.04E', 'C-17.71N-5.11E', 'C-14.79N74.83E', 'C10.67N26.68E', 'C-44.86N-0.68E', 'C-22.07N-161.72E', 'C-34.29N95.63E', 'C-19.22N-47.06E', 'C-16.16N15.98E', 'C4.24N59.29E', 'C-14.64N47.80E', 'C-12.41N9.88E', 'C-33.35N35.57E', 'C-26.98N-4.26E', 'C-22.34N1.63E', 'C8.16N-175.85E', 'C-18.92N23.37E', 'C-21.20N38.35E', 'C3.79N99.62E', 'C-31.28N43.66E', 'C24.36N75.74E', 'C30.35N-159.25E', 'C-31.43N3.20E', 'C-22.36N-8.59E', 'C-21.43N174.33E', 'C-3.63N-97.37E', 'C-1.29N79.76E', 'C40.57N72.65E', 'C38.64N-38.79E', 'C-42.45N19.48E', 'C-23.82N158.21E', 'C-20.08N-4.24E', 'C20.91N40.81E', 'C-9.03N-90.46E', 'C-43.59N-104.47E', 'C-18.19N-5.83E', 'C25.65N-141.85E', 'C11.43N-67.72E', 'C-43.86N5.75E', 'C-41.76N59.77E', 'C-0.87N-57.47E', 'C-17.54N24.91E', 'C-30.84N134.82E', 'C-14.02N153.42E', 'C-20.41N-33.95E', 'C-19.71N-152.39E', 'C10.56N137.09E', 'C-35.42N-66.21E', 'C10.48N-62.83E', 'C30.58N47.29E', 'C-46.66N-56.80E', 'C-26.68N1.50E', 'C-35.46N116.39E', 'C-33.35N13.91E', 'C-14.06N145.50E', 'C-15.96N-59.29E', 'C2.73N73.57E', 'C16.17N74.30E', 'C-1.98N127.91E', 'C15.24N-28.30E', 'C36.82N-81.85E', 'C1.26N62.58E', 'C-2.52N161.03E', 'C-35.76N-15.42E', 'C-19.92N29.59E', 'C1.68N144.49E', 'C13.07N65.20E', 'C15.39N157.16E', 'C-23.04N27.98E', 'C4.44N-162.10E', 'C22.95N39.52E', 'C-18.89N27.49E', 'C-29.83N-11.38E', 'C-11.78N166.56E', 'C13.40N175.15E', 'C-33.30N-33.69E', 'C-37.06N55.87E', 'C-7.15N93.64E', 'C-8.39N113.80E', 'C3.05N107.44E', 'C29.63N81.93E', 'C-31.24N29.84E', 'C37.37N59.81E', 'C22.96N74.85E', 'C-32.02N2.84E', 'C-32.10N10.57E', 'C-31.13N-19.24E', 'C-7.17N165.23E', 'C-30.25N150.21E', 'C-13.06N-169.17E', 'C-6.64N163.04E', 'C-7.41N8.21E', 'C-15.21N-153.44E', 'C14.72N48.39E', 'C4.34N61.86E', 'C1.88N-9.87E', 'C-40.91N-13.21E', 'C-22.68N146.11E', 'C43.56N-170.52E', 'C-6.56N145.39E', 'C-32.23N73.92E', 'C-24.16N159.86E', 'C-28.96N38.43E', 'C-11.26N-146.70E', 'C-14.40N-84.21E', 'C41.81N-76.41E', 'C40.51N4.78E', 'C-9.91N143.19E', 'C7.70N140.09E', 'C24.89N-164.68E', 'C-41.62N112.24E', 'C-5.77N122.21E', 'C-27.91N8.98E', 'C-20.23N-19.84E', 'C5.44N49.05E', 'C25.65N-159.53E', 'C25.70N-89.60E', 'C-11.16N10.00E', 'C-10.23N84.18E', 'C8.79N65.27E', 'C-33.17N-41.06E', 'C-18.31N-164.93E', 'C22.13N64.33E', 'C-25.06N-5.05E', 'C-0.59N37.19E', 'C-34.27N27.51E', 'C-23.92N-78.46E', 'C-39.13N67.91E', 'C-21.49N-70.87E', 'C-6.90N82.17E', 'C8.70N17.60E', 'C-46.52N-107.66E', 'C-28.53N-62.90E', 'C-42.70N4.92E', 'C-31.34N134.16E', 'C-40.20N77.68E', 'C-8.33N64.96E', 'C-32.12N-65.33E', 'C-14.05N-72.74E', 'C29.32N79.71E', 'C-29.72N9.14E', 'C7.58N2.01E', 'C27.42N47.03E', 'C-42.56N69.42E', 'C8.12N121.78E', 'C-4.59N120.88E', 'C7.31N65.77E', 'C2.77N17.29E', 'C27.28N-115.51E', 'C-37.42N114.48E', 'C1.89N173.20E', 'C46.99N127.52E', 'C-39.19N7.84E', 'C-17.56N61.79E', 'C-5.85N4.89E', 'C-8.40N124.96E', 'C-2.22N158.08E', 'C-18.35N-17.69E', 'C-21.07N103.30E', 'C9.57N60.66E', 'C0.47N15.63E', 'C-24.43N36.44E', 'C-2.60N76.51E', 'C14.91N41.46E', 'C25.94N-110.96E', 'C-13.85N99.40E', 'C5.24N-109.56E', 'C1.12N-75.04E', 'C-44.07N-74.59E', 'C-5.68N157.89E', 'C41.42N74.35E', 'C17.70N102.67E', 'C-21.83N19.60E', 'C-35.75N68.29E', 'C-0.89N-163.88E', 'C22.87N-87.65E', 'C-36.28N76.82E', 'C-2.23N156.70E', 'C31.07N158.84E', 'C0.10N169.66E', 'C13.51N-172.18E', 'C37.42N110.42E', 'C26.71N81.29E', 'C28.96N-109.27E', 'C-37.92N61.70E', 'C35.72N-88.21E', 'C27.62N-34.30E', 'C-45.09N66.75E', 'C-30.25N163.90E', 'C12.05N150.40E', 'C17.21N26.34E', 'C21.67N-46.57E', 'C40.36N41.77E', 'C-14.76N-150.80E', 'C-2.41N15.79E', 'C-26.19N-163.54E', 'C-17.62N1.17E', 'C17.75N33.81E', 'C21.14N161.46E', 'C13.66N-136.50E', 'C-43.57N26.72E', 'C29.79N40.29E', 'C36.54N63.40E', 'C-12.20N37.94E', 'C19.43N-96.97E', 'C22.83N167.75E', 'C-20.97N-71.12E', 'C-23.88N101.08E', 'C-0.04N174.55E', 'C39.72N149.88E', 'C-11.38N90.71E', 'C18.83N-165.79E', 'C37.11N126.34E', 'C-28.32N-17.29E', 'C-7.93N-159.94E', 'C-6.60N171.73E', 'C6.71N-2.45E', 'C-27.96N22.33E', 'C16.37N32.96E', 'C-19.00N92.55E', 'C7.97N154.69E', 'C-2.11N80.96E', 'C0.26N152.35E', 'C-35.36N26.07E', 'C-30.01N121.83E', 'C-28.64N25.01E', 'C31.38N76.98E', 'C-37.51N77.84E', 'C-30.47N171.66E', 'C0.24N120.70E', 'C2.86N71.90E', 'C-1.47N-29.21E', 'C-39.01N22.75E', 'C11.96N-93.56E', 'C-19.03N12.91E', 'C-9.39N22.10E', 'C34.83N78.39E', 'C34.26N79.01E', 'C-0.81N15.42E', 'C-6.44N96.77E', 'C-20.48N143.91E', 'C41.71N70.79E', 'C8.49N94.48E', 'C-20.70N16.15E', 'C-18.63N-174.41E', 'C-10.30N-161.83E', 'C40.78N96.81E', 'C-28.80N-162.34E', 'C-9.82N66.41E', 'C0.73N123.46E', 'C3.51N-1.39E', 'C37.21N-85.61E', 'C-3.15N-6.58E', 'C-22.30N6.88E', 'C-39.10N-22.18E', 'C7.93N0.69E', 'C-8.53N-66.83E', 'C3.22N-32.53E', 'C5.89N-81.61E', 'C-32.15N-10.48E', 'C-20.82N60.99E', 'C-10.16N-107.21E', 'C-26.87N50.80E', 'C-7.96N136.47E', 'C16.87N157.02E', 'C5.35N-150.04E', 'C-16.12N1.39E', 'C-0.06N6.44E', 'C42.41N80.02E', 'C-24.55N-63.81E', 'C-41.06N17.58E', 'C-36.18N-0.75E', 'C-37.30N12.72E', 'C-26.44N172.82E', 'C-39.47N4.14E', 'C-1.19N-109.18E', 'C4.62N91.15E', 'C34.63N80.28E', 'C40.04N95.31E', 'C-35.75N-40.08E', 'C43.16N113.64E', 'C-4.74N-60.47E', 'C-33.73N3.91E', 'C-14.38N65.77E', 'C0.09N68.71E', 'C36.50N-76.62E', 'C-45.70N49.99E', 'C2.50N-158.06E', 'C-45.30N-0.59E', 'C-40.23N-132.18E', 'C-20.96N-159.69E', 'C-28.50N-11.63E', 'C-18.60N-125.39E', 'C-45.56N60.04E', 'C14.56N141.08E', 'C-22.53N-64.23E', 'C-8.02N-67.10E', 'C-23.78N-136.02E', 'C43.57N76.73E', 'C-43.58N-105.48E', 'C7.48N153.04E', 'C40.69N55.81E', 'C-37.81N101.79E', 'C-26.77N174.71E', 'C-44.62N1.84E', 'C-36.20N79.87E', 'C-14.34N-155.15E', 'C-7.08N66.98E', 'C31.59N155.43E', 'C7.56N44.90E', 'C-7.90N-0.80E', 'C-11.13N118.00E', 'C-5.30N-168.86E', 'C20.57N-20.59E', 'C36.91N-88.18E', 'C9.88N83.77E', 'C-30.56N95.59E', 'C18.26N53.35E', 'C25.95N176.68E', 'C-31.43N41.80E', 'C12.37N49.49E', 'C22.21N-88.52E', 'C-41.50N-26.25E', 'C18.34N-95.17E', 'C-41.40N74.88E', 'C-20.71N100.05E', 'C-31.56N-47.78E', 'C-34.57N136.72E', 'C-0.52N-158.45E', 'C24.19N160.29E', 'C44.21N-89.82E', 'C3.43N59.51E', 'C10.52N53.57E', 'C-33.24N-149.77E', 'C-26.37N133.21E', 'C42.83N66.73E', 'C-20.81N123.26E', 'C-45.77N46.97E', 'C9.41N65.69E', 'C-24.82N-173.51E', 'C-25.72N-157.32E', 'C-33.46N52.71E', 'C-17.32N107.90E', 'C1.40N112.09E', 'C-38.95N25.17E', 'C-27.68N-103.21E', 'C-45.01N-90.66E', 'C-6.25N80.71E', 'C3.39N150.94E', 'C-4.49N-44.34E', 'C24.01N57.17E', 'C-13.46N51.59E', 'C2.94N-155.69E', 'C-18.55N17.00E', 'C-8.42N106.17E', 'C-39.36N62.77E', 'C-18.08N61.02E', 'C-34.29N26.86E', 'C-32.26N145.45E', 'C-24.04N64.31E', 'C-46.07N-16.25E', 'C-36.37N7.28E', 'C-8.84N-149.17E', 'C-0.85N75.53E', 'C-7.68N151.36E', 'C-44.85N51.67E', 'C28.92N50.53E', 'C-24.65N-161.81E', 'C-25.38N-140.59E', 'C5.95N123.01E', 'C-34.11N16.68E', 'C15.10N-91.67E', 'C-25.84N23.76E', 'C15.00N17.12E', 'C-33.36N153.02E', 'C-4.47N-17.33E', 'C3.95N-74.50E', 'C-20.07N161.42E', 'C6.39N102.18E', 'C2.31N158.73E', 'C-44.13N-21.76E', 'C19.88N-94.12E', 'C-44.70N-22.11E', 'C-10.68N99.68E', 'C-1.89N12.92E', 'C9.03N12.72E', 'C-16.00N2.21E', 'C9.84N68.35E', 'C16.02N-76.51E', 'C-34.82N-21.52E', 'C-15.89N65.79E', 'C18.75N49.89E', 'C4.45N49.83E', 'C33.96N125.03E', 'C13.83N-91.88E', 'C24.08N-128.02E', 'C-10.84N153.83E', 'C12.05N-138.03E', 'C-38.73N-67.41E', 'C15.28N-142.88E', 'C-41.08N-13.91E', 'C-39.77N-157.49E', 'C11.75N74.75E', 'C20.38N113.93E', 'C-21.58N-4.93E', 'C11.87N-83.50E', 'C-37.30N70.99E', 'C-32.04N138.63E', 'C9.76N123.99E', 'C-46.01N-9.14E', 'C-41.54N9.65E', 'C-20.42N121.54E', 'C-1.26N-125.15E', 'C29.45N33.81E', 'C-10.99N42.84E', 'C-38.37N-59.43E', 'C10.16N15.75E', 'C6.46N-83.73E', 'C-31.53N-50.56E', 'C-26.58N-158.39E', 'C-26.14N-45.83E', 'C-27.52N101.59E', 'C-39.57N176.34E', 'C13.88N115.49E', 'C-15.33N2.08E', 'C36.41N60.88E', 'C-10.93N-151.75E', 'C-9.92N114.09E', 'C-42.50N-15.99E', 'C-5.48N167.00E', 'C-33.08N7.37E', 'C10.52N20.10E', 'C-25.47N-2.29E', 'C-25.93N13.22E', 'C-20.78N30.57E', 'C-34.36N-36.30E', 'C7.99N2.18E', 'C-31.12N52.55E', 'C-41.81N-142.18E', 'C5.25N101.92E', 'C-40.02N-117.51E', 'C-12.61N90.30E', 'C26.39N-78.30E', 'C26.87N122.44E', 'C-5.42N18.97E', 'C-1.16N116.85E', 'C-22.32N15.78E', 'C-17.33N-55.18E', 'C-41.28N-38.80E', 'C-10.16N94.74E', 'C-3.84N90.65E', 'C-3.66N-65.65E', 'C29.48N83.23E', 'C-43.30N-15.38E', 'C-25.58N11.84E', 'C36.54N82.77E', 'C45.43N106.58E', 'C46.33N165.48E', 'C-32.98N135.38E', 'C-12.20N-47.67E', 'C0.05N-85.21E', 'C-39.26N131.26E', 'C-40.30N-171.01E', 'C-7.18N66.26E', 'C-32.25N-48.52E', 'C-16.92N100.99E', 'C-30.83N83.64E', 'C43.61N45.91E', 'C4.18N35.30E', 'C12.24N147.04E', 'C-12.32N38.56E', 'C21.66N1.95E', 'C-36.44N-26.32E', 'C12.25N-73.50E', 'C-12.26N-15.33E', 'C12.45N65.08E', 'C-2.29N-98.89E', 'C-9.75N-144.20E', 'C-5.45N169.37E', 'C-26.24N93.05E', 'C28.61N54.58E', 'C9.03N-79.82E', 'C29.51N-154.25E', 'C4.34N-83.85E', 'C11.14N51.41E', 'C-33.74N57.72E', 'C-18.67N91.26E', 'C25.18N-83.36E', 'C-45.72N-89.29E', 'C33.52N-139.64E', 'C-10.18N-141.13E', 'C8.36N171.17E', 'C-45.34N48.30E', 'C-43.17N34.37E', 'C-23.46N-21.96E', 'C7.02N75.33E', 'C-31.82N28.42E', 'C-37.62N27.70E', 'C-39.18N-25.14E', 'C26.32N74.48E', 'C-46.41N45.34E', 'C3.77N6.72E', 'C-6.69N172.87E', 'C-39.96N26.38E', 'C-0.98N60.00E', 'C-3.66N-156.29E', 'C-39.73N-144.25E', 'C31.54N-82.51E', 'C39.32N120.47E', 'C44.11N-88.03E', 'C7.22N-122.06E', 'C16.63N140.27E', 'C34.38N154.19E', 'C-30.01N-5.80E', 'C-3.56N160.19E', 'C-23.02N160.17E', 'C-38.04N-10.81E', 'C29.13N33.05E', 'C-37.27N-3.41E', 'C-8.23N111.54E', 'C28.35N160.30E', 'C-21.09N-29.72E', 'C-26.70N97.03E', 'C-34.76N50.74E', 'C-3.80N-114.82E', 'C7.71N1.37E', 'C-44.12N-15.93E', 'C-14.42N5.69E', 'C34.93N-141.32E', 'C-35.39N-12.23E', 'C29.19N45.31E', 'C-24.00N152.80E', 'C-1.11N78.82E', 'C27.62N-104.91E', 'C-36.24N28.77E', 'C-15.49N-7.90E', 'C23.87N168.14E', 'C-24.76N-150.24E', 'C-23.20N120.41E', 'C6.20N35.17E', 'C-12.17N65.46E', 'C0.80N162.58E', 'C-30.22N154.47E', 'C5.32N17.63E', 'C-20.55N92.50E', 'C2.72N-102.30E', 'C-45.67N28.03E', 'C-1.97N-145.94E', 'C33.80N30.79E', 'C-30.12N15.43E', 'C-31.45N146.36E', 'C-16.20N132.32E', 'C12.63N-80.07E', 'C-33.70N-93.91E', 'C-8.67N67.32E', 'C-21.59N167.13E', 'C17.76N0.01E', 'C-22.27N23.06E', 'C-25.11N-167.86E', 'C-45.48N-40.73E', 'C19.71N-95.94E', 'C-3.07N164.02E', 'C-14.23N3.60E', 'C-7.94N76.89E', 'C-4.14N-157.84E', 'C-40.96N21.47E', 'C-19.71N-60.32E', 'C35.97N102.89E', 'C10.04N50.06E', 'C34.00N66.53E', 'C-34.15N-150.84E', 'C-2.94N-69.32E', 'C-0.41N38.69E', 'C-18.15N159.44E', 'C-2.66N95.33E', 'C-22.79N10.44E', 'C0.59N68.22E', 'C-34.74N159.81E', 'C45.34N49.56E', 'C36.35N176.31E', 'C38.92N166.71E', 'C-34.78N57.24E', 'C-4.68N90.47E', 'C-0.01N78.94E', 'C14.57N54.72E', 'C-18.68N147.25E', 'C4.50N108.15E', 'C16.36N33.80E', 'C-40.89N65.41E', 'C-23.67N-32.10E', 'C-10.35N-64.00E', 'C-22.66N-145.72E', 'C6.13N-173.62E', 'C-35.62N-10.42E', 'C17.31N-99.64E', 'C-4.27N138.48E', 'C6.97N137.49E', 'C28.50N38.11E', 'C-7.13N-166.68E', 'C-12.81N125.50E', 'C43.54N-71.66E', 'C-13.52N20.92E', 'C7.49N111.65E', 'C7.86N52.94E', 'C10.46N156.68E', 'C34.18N155.31E', 'C32.38N-124.73E', 'C22.41N61.26E', 'C-22.41N0.02E', 'C-13.32N-65.02E', 'C35.44N-99.90E', 'C-3.67N-140.67E', 'C30.31N39.79E', 'C-27.07N121.99E', 'C-22.39N-63.16E', 'C20.74N-97.26E', 'C-40.37N1.62E', 'C-24.23N-24.68E', 'C29.02N-45.61E', 'C13.54N148.99E', 'C-13.97N127.72E', 'C28.72N44.22E', 'C-13.26N-0.20E', 'C-3.20N14.61E', 'C35.17N112.99E', 'C-11.97N-150.50E', 'C25.41N-96.84E', 'C-23.88N-60.96E', 'C32.37N121.55E', 'C-28.28N174.09E', 'C12.48N50.15E', 'C-9.37N94.36E', 'C-3.14N-51.72E', 'C9.05N132.53E', 'C-0.84N120.50E', 'C1.99N110.49E', 'C27.19N37.25E', 'C-22.43N100.57E', 'C-0.84N92.71E', 'C-16.45N-3.27E', 'C28.37N45.53E', 'C-26.63N16.27E', 'C-7.80N-124.18E', 'C0.02N9.80E', 'C-7.72N97.45E', 'C3.28N-101.77E', 'C-19.07N78.95E', 'C-43.23N-16.35E', 'C-33.11N8.66E', 'C-11.02N-68.09E', 'C32.89N129.14E', 'C-43.90N142.06E', 'C-28.74N17.25E', 'C-32.02N93.55E', 'C36.94N-107.85E', 'C-28.49N97.69E', 'C-7.64N2.04E', 'C34.24N-115.19E', 'C-32.15N37.78E', 'C-12.58N-67.46E', 'C-21.77N-73.14E', 'C9.67N47.90E', 'C7.84N-125.10E', 'C-41.89N-120.15E', 'C-25.43N-54.27E', 'C-21.68N-0.73E', 'C-19.07N-170.25E', 'C-4.96N119.31E', 'C5.02N103.43E', 'C-33.53N-133.99E', 'C40.43N-23.11E', 'C-39.00N58.02E', 'C-17.03N1.78E', 'C-40.26N-140.86E', 'C37.29N-174.97E', 'C43.17N69.36E', 'C6.23N-168.93E', 'C-3.81N99.34E', 'C-27.40N-16.28E', 'C-36.06N-40.24E', 'C-35.90N2.03E', 'C-18.01N158.45E', 'C24.29N40.12E', 'C-25.81N8.47E', 'C36.84N65.51E', 'C0.47N98.14E', 'C4.97N-71.88E', 'C-42.60N-40.19E', 'C-45.68N33.32E', 'C7.35N43.30E', 'C24.26N140.48E', 'C31.25N106.71E', 'C-11.13N8.48E', 'C21.37N-155.27E', 'C-41.71N-29.44E', 'C-13.92N7.25E', 'C3.68N-130.08E', 'C28.58N142.39E', 'C32.01N103.61E', 'C27.15N-155.37E', 'C42.18N59.35E', 'C38.05N-124.09E', 'C-36.90N-43.93E', 'C-11.66N-70.33E', 'C-25.86N114.11E', 'C13.55N28.64E', 'C-14.80N51.74E', 'C-42.93N33.27E', 'C-0.48N-151.75E', 'C12.29N-94.86E', 'C-8.39N17.66E', 'C-0.70N-5.88E', 'C-0.77N111.35E', 'C-4.50N77.27E', 'C11.48N52.84E', 'C19.94N-123.33E', 'C18.55N143.80E', 'C22.23N-167.57E', 'C11.67N21.74E', 'C-18.60N-145.50E', 'C-12.92N-65.53E', 'C33.01N139.49E', 'C4.82N56.85E', 'C44.76N-98.00E', 'C4.03N-1.47E', 'C-45.78N128.80E', 'C-32.29N174.30E', 'C-38.48N-156.12E', 'C39.05N-83.08E', 'C-44.49N71.38E', 'C33.83N-171.64E', 'C0.92N-15.24E', 'C23.02N174.35E', 'C29.34N-77.10E', 'C-32.66N10.13E', 'C-14.50N149.35E', 'C9.35N56.83E', 'C-16.96N-51.34E', 'C-5.46N89.67E', 'C10.71N99.83E', 'C-12.35N-174.13E', 'C40.67N174.64E', 'C-27.82N-39.19E', 'C-25.07N133.28E', 'C-15.85N107.78E', 'C0.63N41.31E', 'C28.15N-70.91E', 'C-43.16N68.78E', 'C-44.89N64.47E', 'C31.53N-153.75E', 'C-40.66N12.22E', 'C33.72N158.26E', 'C-22.59N37.25E', 'C29.23N-113.90E', 'C11.64N12.96E', 'C15.72N-83.12E', 'C-3.50N65.61E', 'C-32.96N-31.87E', 'C27.12N-73.78E', 'C31.41N47.73E', 'C-31.69N50.87E', 'C8.77N-131.99E', 'C-5.26N145.95E', 'C23.63N-166.04E', 'C41.90N-82.34E', 'C-8.91N-14.53E', 'C-8.03N64.29E', 'C1.95N13.50E', 'C3.25N161.56E', 'C-18.70N6.96E', 'C-4.75N-64.87E', 'C42.05N127.07E', 'C-29.40N43.91E', 'C-15.38N140.01E', 'C-15.95N-135.80E', 'C-22.21N-21.57E', 'C16.57N40.24E', 'C-14.02N-62.50E', 'C-32.38N12.17E', 'C-22.79N122.05E', 'C0.74N66.91E', 'C32.31N43.10E', 'C-15.99N115.15E', 'C-15.43N65.48E', 'C24.72N44.62E', 'C-22.42N-122.03E', 'C-4.30N-1.77E', 'C33.62N119.07E', 'C-5.25N-136.48E', 'C-39.10N157.38E', 'C-44.12N28.90E', 'C36.23N134.69E', 'C44.38N-108.11E', 'C20.26N-8.75E', 'C42.25N-173.86E', 'C-7.25N106.07E', 'C-22.79N-16.63E', 'C-44.34N167.31E', 'C-36.55N-38.13E', 'C-35.73N155.80E', 'C30.09N-153.47E', 'C19.27N78.99E', 'C-24.43N165.98E', 'C19.88N132.21E', 'C-14.00N147.30E', 'C-35.35N50.05E', 'C35.81N100.76E', 'C-45.58N-14.07E', 'C23.26N-29.18E', 'C-27.47N-167.92E', 'C2.88N7.27E', 'C31.44N85.77E', 'C4.28N-21.67E', 'C-15.11N5.83E', 'C15.17N28.79E', 'C19.07N98.06E', 'C-1.00N124.47E', 'C21.22N168.21E', 'C25.39N158.72E', 'C29.89N124.86E', 'C-23.42N112.81E', 'C-19.33N144.75E', 'C-9.75N11.12E', 'C-13.34N-143.64E', 'C-36.66N127.23E', 'C-26.31N151.12E', 'C-13.63N-66.72E', 'C-5.69N171.12E', 'C-36.49N-138.61E', 'C7.46N116.22E', 'C-16.33N81.44E', 'C24.43N38.64E', 'C-41.70N147.70E', 'C-7.56N87.67E', 'C25.80N-163.39E', 'C-37.72N-10.22E', 'C-16.61N88.42E', 'C35.32N62.86E', 'C-34.84N63.88E', 'C-20.91N101.77E', 'C-13.83N93.04E', 'C-15.31N149.60E', 'C38.04N138.05E', 'C45.52N-88.22E', 'C16.09N46.89E', 'C5.16N117.15E', 'C30.72N46.64E', 'C13.49N121.21E', 'C-7.58N21.99E', 'C35.57N-172.73E', 'C0.56N-159.44E', 'C-23.39N-18.90E', 'C16.26N15.93E', 'C11.54N-148.44E', 'C-44.97N21.41E', 'C20.85N-114.74E', 'C24.28N39.15E', 'C-17.67N8.46E', 'C-32.76N105.24E', 'C32.74N37.25E', 'C-0.67N-128.48E', 'C-13.25N0.80E', 'C-38.91N-126.63E', 'C-15.13N-159.89E', 'C23.02N-139.58E', 'C-37.47N17.25E', 'C11.66N-162.09E', 'C43.14N121.39E', 'C44.08N141.84E', 'C17.68N136.70E', 'C-11.04N29.76E', 'C-42.52N21.03E', 'C-23.45N2.75E', 'C-34.36N10.90E', 'C-17.95N-5.96E', 'C31.47N40.80E', 'C24.67N173.89E', 'C44.65N114.50E', 'C8.51N-173.29E', 'C6.95N100.52E', 'C-38.98N129.99E', 'C-38.37N154.62E', 'C-15.97N-145.13E', 'C6.26N170.64E', 'C0.19N-7.53E', 'C7.96N-144.20E', 'C-11.59N-112.92E', 'C13.07N97.80E', 'C-37.15N-62.75E', 'C16.97N-0.32E', 'C-38.25N31.35E', 'C8.64N100.68E', 'C0.65N119.88E', 'C24.51N-166.74E', 'C-16.51N-25.68E', 'C-17.43N2.54E', 'C-8.22N42.42E', 'C35.42N31.16E', 'C20.55N98.46E', 'C-12.44N158.10E', 'C-22.08N23.49E', 'C-17.24N136.89E', 'C-10.62N-50.60E', 'C-1.41N71.48E', 'C-37.54N-27.17E', 'C-1.50N171.52E', 'C6.19N174.86E', 'C-37.44N80.77E', 'C-10.78N-145.08E', 'C-43.32N8.04E', 'C24.58N-116.16E', 'C-7.92N7.31E', 'C36.97N-73.26E', 'C21.50N31.39E', 'C17.16N145.16E', 'C-9.26N93.54E', 'C40.30N32.98E', 'C4.42N166.33E', 'C15.85N45.45E', 'C-2.24N134.57E', 'C-5.88N57.78E', 'C-10.21N-4.52E', 'C40.52N67.89E', 'C-25.47N-5.69E', 'C-0.29N108.04E', 'C0.34N-150.62E', 'C20.12N153.11E', 'C18.03N108.23E', 'C-4.16N174.87E', 'C36.98N38.68E', 'C-0.87N-149.53E', 'C39.40N99.73E', 'C29.21N61.86E', 'C39.75N-104.76E', 'C43.88N127.48E', 'C-39.94N-12.07E', 'C6.90N58.18E', 'C-3.91N-62.84E', 'C-18.13N163.59E', 'C-7.88N66.86E', 'C2.34N139.81E', 'C-30.86N-171.76E', 'C-3.23N67.56E', 'C38.17N-78.94E', 'C-7.93N-169.51E', 'C-39.58N61.61E', 'C-7.99N83.10E', 'C7.41N-5.89E', 'C21.72N107.11E', 'C27.92N-93.31E', 'C-34.74N-9.40E', 'C21.80N-101.32E', 'C-12.73N44.99E', 'C-28.54N90.20E', 'C-9.05N84.55E', 'C-20.97N0.52E', 'C0.03N34.09E', 'C27.17N77.01E', 'C1.96N19.17E', 'C-15.92N167.57E', 'C6.26N-163.94E', 'C-33.15N149.77E', 'C-18.57N1.23E', 'C28.55N150.29E', 'C-38.40N3.48E', 'C-39.51N-53.87E', 'C-40.97N-39.84E', 'C-3.99N5.85E', 'C-11.21N157.98E', 'C2.26N14.60E', 'C20.10N151.50E', 'C1.38N20.07E', 'C-26.85N-135.50E', 'C-21.09N-16.37E', 'C-8.98N156.91E', 'C-2.26N143.56E', 'C-31.41N16.06E', 'C-3.73N-166.83E', 'C-23.94N-2.81E', 'C39.72N72.16E', 'C-25.01N30.14E', 'C10.52N-131.59E', 'C25.77N-20.99E', 'C-25.16N-169.09E', 'C29.33N-121.92E', 'C2.41N75.88E', 'C-27.87N17.63E', 'C-8.73N-73.89E', 'C12.63N-82.94E', 'C4.35N-117.48E', 'C-32.10N18.71E', 'C-22.27N154.58E', 'C31.42N42.63E', 'C-39.91N172.09E', 'C7.99N142.44E', 'C-19.25N4.77E', 'C-2.04N172.60E', 'C-38.75N21.80E', 'C-23.30N-152.95E', 'C-5.88N119.58E', 'C-1.25N69.40E', 'C-20.52N138.09E', 'C-12.43N91.56E', 'C-5.74N-156.81E', 'C23.61N117.44E', 'C-6.05N40.04E', 'C-3.82N68.39E', 'C-14.78N143.69E', 'C39.07N68.91E', 'C1.94N84.64E', 'C39.27N144.97E', 'C28.48N39.18E', 'C9.54N134.14E', 'C38.28N-146.43E', 'C-10.62N20.20E', 'C-14.15N71.28E', 'C-4.72N28.40E', 'C-18.41N-149.59E', 'C-25.83N-50.83E', 'C-33.51N3.52E', 'C17.66N31.28E', 'C-23.16N-46.84E', 'C34.59N131.61E', 'C-11.69N21.71E', 'C-17.82N40.17E', 'C-37.70N-8.79E', 'C43.18N-112.26E', 'C5.87N57.60E', 'C-38.50N94.77E', 'C27.76N-106.32E', 'C16.15N151.02E', 'C13.76N147.34E', 'C20.98N-140.55E', 'C-0.44N-67.38E', 'C-40.32N-147.84E', 'C-1.26N77.15E', 'C-31.54N65.70E', 'C28.21N143.84E', 'C-41.88N-17.85E', 'C-0.18N-160.53E', 'C-24.33N134.00E', 'C-16.04N-56.79E', 'C0.58N-73.20E', 'C-1.37N-4.36E', 'C-15.86N138.85E', 'C-20.59N-155.01E', 'C12.36N52.84E', 'C-23.48N-155.97E', 'C2.41N32.45E', 'C28.63N126.72E', 'C-40.40N39.95E', 'C-16.24N40.06E', 'C33.81N36.75E', 'C-11.12N-71.89E', 'C40.14N140.31E', 'C42.94N-161.60E', 'C29.47N51.62E', 'C-6.11N69.45E', 'C-38.65N123.07E', 'C-32.83N-2.57E', 'C-27.57N37.06E', 'C-35.93N-147.95E', 'C-18.49N64.92E', 'C-35.71N18.74E', 'C-14.89N173.21E', 'C-35.25N150.07E', 'C23.78N48.55E', 'C26.70N-132.12E', 'C-32.60N128.45E', 'C28.43N111.58E', 'C16.08N123.08E', 'C-15.55N-39.80E', 'C-7.00N13.01E', 'C38.85N64.38E', 'C32.61N161.78E', 'C5.91N116.07E', 'C18.99N135.13E', 'C-36.91N35.54E', 'C-24.52N-128.91E', 'C-30.20N-49.67E', 'C9.09N161.68E', 'C-6.96N130.67E', 'C1.47N104.89E', 'C-40.06N-49.62E', 'C-32.05N-73.91E', 'C2.91N131.18E', 'C-38.09N126.65E', 'C18.24N118.59E', 'C-17.71N132.58E', 'C-31.79N-151.01E', 'C34.63N-161.36E', 'C12.77N-158.67E', 'C10.21N64.01E', 'C-10.41N12.38E', 'C-36.75N-163.37E', 'C-25.82N-161.92E', 'C-4.70N57.93E', 'C-19.26N-17.25E', 'C40.07N-136.90E', 'C-11.99N84.11E', 'C-2.80N-2.06E', 'C-10.93N37.82E', 'C15.54N-29.17E', 'C-32.41N-7.80E', 'C29.06N157.73E', 'C-1.86N65.34E', 'C42.70N30.51E', 'C6.22N-137.87E', 'C-30.18N-141.98E', 'C-27.21N-128.04E', 'C-40.75N-38.67E', 'C27.53N117.42E', 'C8.10N-134.66E', 'C-28.28N48.87E', 'C40.82N-172.27E', 'C-36.92N-1.39E', 'C31.40N-79.24E', 'C-26.08N10.66E', 'C-39.89N-14.54E', 'C41.94N-130.33E', 'C-38.14N65.36E', 'C-16.55N5.80E', 'C0.73N100.77E', 'C-24.86N96.96E', 'C29.44N-104.24E', 'C-11.85N-8.15E', 'C3.08N70.30E', 'C9.46N53.52E', 'C43.46N-124.01E', 'C38.92N10.72E', 'C19.07N148.36E', 'C-18.75N-63.62E', 'C-36.42N158.62E', 'C-30.35N-59.47E', 'C-23.68N19.44E', 'C26.72N-13.10E', 'C42.02N128.21E', 'C-42.30N115.42E', 'C-21.09N-169.21E', 'C1.82N10.16E', 'C28.71N51.47E', 'C33.65N-169.91E', 'C-11.59N-159.32E', 'C15.64N-95.31E', 'C41.14N54.86E', 'C-9.84N131.06E', 'C-28.22N44.74E', 'C3.29N77.87E', 'C-37.43N-170.42E', 'C-0.61N101.71E', 'C-33.97N-158.99E', 'C-27.43N53.68E', 'C-3.98N87.35E', 'C-28.12N-143.19E', 'C-6.71N87.81E', 'C-22.52N102.87E', 'C1.20N80.67E', 'C15.91N71.83E', 'C-36.75N21.94E', 'C-0.97N66.81E', 'C16.10N161.80E', 'C23.79N126.50E', 'C-44.02N53.90E', 'C-32.95N-158.83E', 'C44.12N-46.04E', 'C28.10N37.13E', 'C4.66N125.28E', 'C-9.98N-58.70E', 'C7.69N12.28E', 'C0.22N-128.70E', 'C35.78N-76.73E', 'C14.60N51.70E', 'C-9.37N-148.06E', 'C8.02N115.05E', 'C-38.20N-0.35E', 'C30.40N58.48E', 'C-24.63N-59.83E', 'C20.15N-143.53E', 'C-5.76N97.12E', 'C-10.17N78.51E', 'C-1.78N90.66E', 'C40.68N-166.21E', 'C31.67N-98.80E', 'C0.36N133.13E', 'C-17.18N-52.37E', 'C7.46N-154.08E', 'C33.18N-119.26E', 'C2.18N156.00E', 'C-4.32N72.66E', 'C11.00N98.26E', 'C30.90N-172.93E', 'C33.19N134.17E', 'C35.99N81.60E', 'C3.77N107.89E', 'C-39.40N-13.74E', 'C41.07N109.13E', 'C-33.29N-137.10E', 'C-25.97N163.27E', 'C-11.93N172.92E', 'C34.47N173.49E', 'C6.50N119.75E', 'C1.12N113.08E', 'C44.01N71.09E', 'C-7.55N-129.79E', 'C10.45N-164.15E', 'C11.23N73.36E', 'C13.94N67.00E', 'C-43.11N17.48E', 'C-20.69N5.10E', 'C-7.70N13.84E', 'C23.25N-49.84E', 'C-34.98N-160.64E', 'C-37.93N-59.20E', 'C28.74N-84.24E', 'C23.20N32.88E', 'C5.57N169.54E', 'C-39.03N-153.11E', 'C11.17N14.15E', 'C-38.12N67.95E', 'C-11.67N145.33E', 'C5.55N160.41E', 'C29.70N163.93E', 'C-1.93N36.60E', 'C0.18N-4.59E', 'C-5.28N16.65E', 'C42.02N37.31E', 'C-21.66N17.39E', 'C-33.59N131.35E', 'C-33.36N11.44E', 'C-8.83N69.26E', 'C29.35N-51.52E', 'C-39.62N-117.31E', 'C-28.60N92.51E', 'C41.47N-101.35E', 'C3.34N144.73E', 'C-37.69N1.79E', 'C-4.85N-61.25E', 'C2.74N-6.98E', 'C-12.78N-154.73E', 'C12.50N-130.13E', 'C-10.22N163.26E', 'C40.74N141.79E', 'C-36.13N81.41E', 'C1.40N161.02E', 'C-19.27N9.38E', 'C42.02N105.09E', 'C-17.04N150.23E', 'C14.94N-143.88E', 'C-34.47N-168.14E', 'C-11.98N44.07E', 'C-35.64N-13.57E', 'C26.16N-153.34E', 'C-38.05N73.47E', 'C9.10N44.95E', 'C30.57N55.55E', 'C-35.04N101.51E', 'C9.99N-163.49E', 'C-37.43N153.68E', 'C-12.29N134.41E', 'C3.06N168.05E', 'C-9.05N109.16E', 'C-11.88N-172.54E', 'C-7.75N71.72E', 'C-10.61N-124.93E', 'C37.55N-148.39E', 'C39.43N9.16E', 'C-41.17N-32.79E', 'C39.81N-146.57E', 'C-27.61N151.15E', 'C-22.71N19.79E', 'C0.37N78.04E', 'C40.45N-139.73E', 'C2.39N57.69E', 'C39.75N161.24E', 'C-9.99N101.99E', 'C32.52N159.67E', 'C-1.78N132.75E', 'C-26.41N68.39E', 'C-4.23N119.16E', 'C-1.31N-165.74E', 'C32.50N-111.00E', 'C27.50N-148.08E', 'C-20.75N7.55E', 'C-26.22N-85.21E', 'C17.54N-1.12E', 'C-8.26N94.88E', 'C-16.76N123.12E', 'C9.27N80.94E', 'C33.43N-140.60E', 'C-42.60N74.67E', 'C0.19N107.94E', 'C26.78N33.78E', 'C14.45N9.07E', 'C36.93N-163.96E', 'C-26.74N-51.37E', 'C-22.43N96.43E', 'C38.73N-127.30E', 'C-0.31N-26.63E', 'C12.05N-108.81E', 'C14.99N132.84E', 'C-4.25N22.58E', 'C-36.36N-17.70E', 'C39.99N64.95E', 'C30.68N1.49E', 'C-18.14N5.61E', 'C34.68N162.60E', 'C-34.10N20.05E', 'C-24.35N-48.30E', 'C-1.85N137.76E', 'C-2.43N86.81E', 'C39.50N-160.47E', 'C-23.97N-57.01E', 'C-5.69N-2.09E', 'C37.77N160.63E', 'C-33.29N-21.18E', 'C-20.02N9.85E', 'C26.64N126.21E', 'C-4.23N15.35E', 'C-30.49N-60.33E', 'C-29.88N-140.09E', 'C40.68N45.78E', 'C2.93N124.69E', 'C12.47N-101.05E', 'C20.99N133.90E', 'C37.39N47.69E', 'C19.36N34.80E', 'C8.27N73.09E', 'C-25.43N21.55E', 'C8.00N-81.18E', 'C29.76N43.67E', 'C-28.64N-47.80E', 'C-15.58N71.00E', 'C-14.68N153.85E', 'C23.73N-47.49E', 'C-4.79N62.30E', 'C-16.05N-4.51E', 'C-15.34N131.81E', 'C-33.66N76.12E', 'C16.15N124.36E', 'C-43.38N7.33E', 'C-31.55N-86.74E', 'C32.79N-145.45E', 'C-18.51N-114.23E', 'C-19.68N-167.62E', 'C7.21N125.57E', 'C18.25N-79.80E', 'C-35.58N17.22E', 'C-17.39N-89.68E', 'C-42.47N25.87E', 'C-4.32N146.38E', 'C30.03N-109.15E', 'C43.17N-88.71E', 'C-33.51N96.39E', 'C6.69N127.21E', 'C30.73N-111.27E', 'C-13.62N131.67E', 'C1.72N-145.00E', 'C-42.38N111.11E', 'C-1.52N-0.45E', 'C-10.67N67.22E', 'C5.00N111.20E', 'C-7.95N91.26E', 'C10.76N-91.69E', 'C35.07N111.70E', 'C-23.21N70.59E', 'C22.19N138.15E', 'C38.36N124.73E', 'C-30.66N-40.76E', 'C14.04N72.54E', 'C34.13N-173.23E', 'C-35.13N-148.21E', 'C22.03N105.56E', 'C15.36N23.61E', 'C-41.31N-143.30E', 'C-7.96N33.50E', 'C-30.34N10.56E', 'C-39.23N-3.93E', 'C-10.78N147.59E', 'C-5.09N143.05E', 'C-0.14N67.56E', 'C-31.36N-149.64E', 'C-17.67N-17.24E', 'C-36.72N82.60E', 'C-25.29N9.13E', 'C31.19N164.11E', 'C-30.81N27.70E', 'C-23.52N-59.02E', 'C0.17N75.95E', 'C-28.02N64.39E', 'C-8.15N137.18E', 'C-1.58N64.10E', 'C-36.90N141.20E', 'C-9.64N97.00E', 'C32.75N-159.83E', 'C-25.78N160.23E', 'C-12.04N63.01E', 'C43.09N47.25E', 'C3.92N171.48E', 'C35.52N167.36E', 'C23.05N-115.66E', 'C-15.89N121.55E', 'C-30.42N-37.55E', 'C32.43N-143.70E', 'C-28.38N48.08E', 'C5.08N85.82E', 'C13.86N172.47E', 'C35.08N151.28E', 'C-9.42N87.38E', 'C-40.76N61.57E', 'C2.75N102.09E', 'C-31.71N129.79E', 'C-17.88N-23.89E', 'C-20.26N-171.58E', 'C24.91N36.79E', 'C37.42N103.36E', 'C-20.99N11.89E', 'C3.76N11.86E', 'C-11.75N-69.59E', 'C-29.42N-16.42E', 'C30.91N-159.21E', 'C-11.99N2.53E', 'C5.39N-159.47E', 'C-35.15N-57.76E', 'C-21.37N11.09E', 'C25.43N36.41E', 'C4.10N10.47E', 'C-34.49N31.99E', 'C-39.18N13.19E', 'C-15.32N25.49E', 'C-29.22N145.75E', 'C-13.33N70.25E', 'C-37.59N69.62E', 'C-38.02N75.48E', 'C-14.57N-70.89E', 'C28.32N52.85E', 'C-12.49N-60.51E', 'C-16.26N142.76E', 'C13.60N100.07E', 'C-14.61N-142.67E', 'C4.08N66.65E', 'C15.42N160.27E', 'C-4.26N8.72E', 'C15.04N136.84E', 'C32.77N-87.68E', 'C-3.82N109.13E', 'C-23.38N-140.97E', 'C-5.83N56.34E', 'C-27.24N-155.99E', 'C23.82N36.17E', 'C-19.46N13.85E', 'C6.19N108.14E', 'C-3.87N-59.96E', 'C41.19N149.97E', 'C-21.89N101.03E', 'C-11.53N-52.06E', 'C23.97N64.80E', 'C21.09N-66.66E', 'C31.35N57.11E', 'C7.76N113.07E', 'C-17.50N48.81E', 'C32.83N-135.14E', 'C37.22N168.68E', 'C-10.04N41.13E', 'C34.72N-119.56E', 'C-39.98N-7.11E', 'C3.39N-140.71E', 'C-14.77N96.81E', 'C-22.41N151.75E', 'C-26.31N-22.63E', 'C-13.83N-50.24E', 'C20.60N-154.15E', 'C-11.91N33.61E', 'C-10.48N138.18E', 'C39.11N-131.76E', 'C9.37N-157.37E', 'C-26.06N-61.77E', 'C-16.68N-66.96E', 'C-24.47N-163.98E', 'C3.68N-55.69E', 'C14.59N-163.83E', 'C25.49N-44.14E', 'C12.43N100.02E', 'C-31.15N94.77E', 'C-25.44N-120.39E', 'C-32.41N71.45E', 'C22.69N-143.75E', 'C-2.61N165.62E', 'C-29.25N-26.11E', 'C0.91N143.41E', 'C-28.04N-27.90E', 'C36.03N128.18E', 'C-26.30N5.76E', 'C28.29N34.75E', 'C17.42N-82.02E', 'C37.73N-156.83E', 'C11.48N-85.07E', 'C40.37N-151.74E', 'C-17.75N-143.47E', 'C-24.20N138.71E', 'C-18.75N114.20E', 'C20.88N136.92E', 'C-3.84N126.80E', 'C-36.53N99.53E', 'C10.58N-152.05E', 'C34.50N-162.84E', 'C-7.88N-15.78E', 'C34.93N60.61E', 'C-36.15N169.80E', 'C12.41N159.36E', 'C8.94N92.32E', 'C-22.19N12.70E', 'C25.65N-150.51E', 'C40.50N-112.81E', 'C-19.19N62.53E', 'C-7.97N18.90E', 'C-3.01N158.70E', 'C-25.43N134.59E', 'C-11.74N15.67E', 'C-16.85N15.14E', 'C-40.35N162.38E', 'C-21.24N94.35E', 'C12.55N47.64E', 'C33.71N-150.65E', 'C41.89N111.17E', 'C4.38N69.95E', 'C-7.65N34.92E', 'C16.81N85.21E', 'C3.43N-126.20E', 'C18.38N6.35E', 'C8.76N70.29E', 'C-12.67N-61.40E', 'C-32.89N-85.27E', 'C-10.79N70.59E', 'C-38.41N103.17E', 'C-16.32N-17.64E', 'C-36.62N-59.25E', 'C24.72N97.24E', 'C-24.05N106.01E', 'C-4.54N93.24E', 'C0.45N-133.40E', 'C-6.74N131.72E', 'C-22.85N-139.70E', 'C-15.07N-151.90E', 'C-31.35N39.88E', 'C32.65N108.23E', 'C27.91N-88.12E', 'C5.48N-1.65E', 'C-22.63N-78.42E', 'C13.27N-72.50E', 'C5.75N-165.64E', 'C5.85N109.77E', 'C35.26N-166.11E', 'C-8.25N103.91E', 'C-23.56N8.45E', 'C-4.02N105.62E', 'C-23.64N94.84E', 'C-33.84N139.86E', 'C-15.72N-170.86E', 'C-1.54N94.86E', 'C16.69N-88.25E', 'C34.17N166.81E', 'C19.51N154.05E', 'C-27.95N99.00E', 'C-4.23N101.89E', 'C-9.86N90.91E', 'C4.51N60.96E', 'C18.34N168.40E', 'C-39.06N-148.25E', 'C-17.93N-131.82E', 'C35.50N-122.72E', 'C4.20N112.70E', 'C31.37N-147.86E', 'C16.56N-72.72E', 'C-27.54N96.31E', 'C-2.45N0.73E', 'C-38.90N17.68E', 'C-5.63N-149.51E', 'C-2.49N117.07E', 'C20.38N-109.41E', 'C-1.94N17.39E', 'C13.57N-96.18E', 'C-30.31N-53.10E', 'C-39.06N-154.63E', 'C8.66N-68.47E', 'C17.79N106.17E', 'C-19.85N-31.11E', 'C-41.04N0.14E', 'C-36.10N64.12E', 'C-19.45N89.80E', 'C-38.67N119.35E', 'C-5.09N-147.15E', 'C-6.88N117.90E', 'C21.91N-99.04E', 'C-31.34N89.17E', 'C-27.91N138.68E', 'C-13.76N142.47E', 'C24.15N34.45E', 'C37.11N86.58E', 'C33.20N114.40E', 'C-6.09N85.43E', 'C-12.67N150.26E', 'C-13.52N-152.46E', 'C-35.90N112.52E', 'C-32.05N124.42E', 'C-32.35N24.77E', 'C-36.49N6.48E', 'C-20.08N-158.14E', 'C14.25N97.40E', 'C38.37N-156.21E', 'C-25.15N-70.96E', 'C-35.06N119.29E', 'C19.45N85.62E', 'C-19.96N-72.29E', 'C-9.50N132.36E', 'C-15.37N92.38E', 'C-14.52N-93.88E', 'C-24.84N-133.01E', 'C-3.99N130.78E', 'C-28.74N166.86E', 'C29.82N-150.01E', 'C35.58N114.75E', 'C-34.55N82.39E', 'C0.96N116.63E', 'C-1.92N67.99E', 'C-16.23N-75.54E', 'C31.11N56.39E', 'C-33.21N-156.15E', 'C-22.01N-4.02E', 'C-23.87N12.27E', 'C-38.43N-76.51E', 'C-26.65N138.61E', 'C-31.35N97.99E', 'C9.01N-147.68E', 'C32.38N-139.21E', 'C-5.50N141.24E', 'C-16.77N133.60E', 'C-18.94N134.57E', 'C-1.74N-169.56E', 'C33.84N48.58E', 'C-40.31N-34.02E', 'C-28.42N14.36E', 'C13.86N-80.78E', 'C-14.13N111.87E', 'C-32.48N-140.36E', 'C4.92N130.20E', 'C-6.82N134.30E', 'C-40.75N165.39E', 'C-17.42N-146.78E', 'C-37.90N145.30E', 'C-16.61N79.41E', 'C7.02N107.01E', 'C14.24N153.95E', 'C-39.38N76.57E', 'C-31.11N63.30E', 'C-26.33N117.71E', 'C-8.50N-157.37E', 'C-22.69N111.74E', 'C7.25N63.43E', 'C40.25N4.64E', 'C1.22N155.19E', 'C29.24N-116.41E', 'C32.67N-81.96E', 'C-18.37N103.82E', 'C5.50N46.54E', 'C-32.97N-63.83E', 'C-31.60N75.15E', 'C19.88N81.36E', 'C-39.52N59.06E', 'C2.75N113.51E', 'C-0.16N12.73E', 'C5.07N-0.21E', 'C-8.77N158.21E', 'C-20.38N88.16E', 'C-39.67N74.62E', 'C-5.08N-61.38E', 'C-23.90N56.62E', 'C-7.71N110.03E', 'C14.47N-11.32E', 'C-10.47N-148.99E', 'C-32.71N113.56E', 'C20.74N119.73E', 'C-35.72N65.50E', 'C33.64N91.05E', 'C23.39N-113.17E', 'C5.10N-66.93E', 'C-29.58N19.58E', 'C14.14N91.14E', 'C-38.21N-51.48E', 'C-25.32N2.44E', 'C1.45N108.22E', 'C-9.58N134.66E', 'C19.35N-146.51E', 'C3.53N-41.47E', 'C-25.86N96.35E', 'C32.19N167.45E', 'C-32.47N141.95E', 'C28.78N-163.67E', 'C17.07N-84.45E', 'C-20.75N-22.26E', 'C-11.57N-14.19E', 'C-28.81N-169.80E', 'C9.91N-83.51E', 'C10.01N-143.69E', 'C-1.47N112.95E', 'C-40.02N145.04E', 'C20.59N137.97E', 'C-39.37N0.78E', 'C-29.39N45.75E', 'C17.41N131.22E', 'C17.25N-158.37E', 'C-25.04N70.80E', 'C-17.25N-128.23E', 'C22.63N-102.46E', 'C5.68N162.38E', 'C-21.74N-132.37E', 'C-13.87N13.91E', 'C31.96N-136.28E', 'C25.44N103.79E', 'C-12.97N-103.42E', 'C22.05N135.08E', 'C21.26N45.97E', 'C-6.81N157.59E', 'C-39.58N-118.20E', 'C-31.89N68.39E', 'C2.71N135.45E', 'C7.84N-164.89E', 'C3.11N96.51E', 'C-26.96N7.87E', 'C5.68N98.54E', 'C-24.79N161.88E', 'C-6.97N-143.09E', 'C-4.52N-27.69E', 'C10.24N97.09E', 'C1.76N127.28E', 'C35.40N-105.92E', 'C36.92N-115.28E', 'C-13.76N-70.32E', 'C-3.23N114.79E', 'C5.40N-78.06E', 'C11.47N10.31E', 'C-30.48N47.95E', 'C18.33N-141.73E', 'C-28.48N-41.51E', 'C-35.80N15.15E', 'C1.79N72.65E', 'C35.30N-159.51E', 'C-23.85N157.93E', 'C32.04N-106.23E', 'C-13.06N163.21E', 'C36.21N-147.36E', 'C12.11N125.53E', 'C-20.30N-28.77E', 'C25.90N-134.67E', 'C38.54N-165.21E', 'C-0.73N-153.22E', 'C16.74N36.08E', 'C25.57N115.01E', 'C-31.76N-19.59E', 'C-20.97N-155.05E', 'C-28.22N-150.43E', 'C-26.93N154.81E', 'C-23.68N1.08E', 'C-4.88N136.37E', 'C-4.20N140.69E', 'C-6.41N84.11E', 'C19.56N158.34E', 'C-19.38N75.62E', 'C-10.06N122.86E', 'C14.78N-159.63E', 'C26.66N30.50E', 'C-38.18N4.86E', 'C-26.75N100.66E', 'C10.48N-13.77E', 'C20.99N99.02E', 'C23.37N-100.93E', 'C-34.37N107.66E', 'C19.24N-83.31E', 'C-11.42N123.90E', 'C-9.63N-133.78E', 'C-11.05N-7.25E', 'C24.18N79.05E', 'C-5.60N164.70E', 'C28.15N109.34E', 'C12.71N-86.52E', 'C-28.46N-139.06E', 'C-8.16N-56.70E', 'C-30.42N157.29E', 'C-28.03N3.29E', 'C-34.90N3.79E', 'C-8.61N41.25E', 'C-37.10N47.17E', 'C-32.49N54.14E', 'C-29.32N135.71E', 'C-37.02N26.43E', 'C-26.36N31.78E', 'C24.67N-161.73E', 'C-29.83N-39.70E', 'C-5.33N138.53E', 'C-36.48N126.66E', 'C7.30N-76.14E', 'C-10.05N45.03E', 'C21.59N-122.05E', 'C24.35N110.91E', 'C8.63N89.50E', 'C-5.06N110.06E', 'C19.38N-116.82E', 'C4.49N102.80E', 'C-6.99N122.28E', 'C-16.22N103.35E', 'C-24.62N-154.86E', 'C29.33N167.14E', 'C-3.33N-3.73E', 'C4.96N120.49E', 'C-17.25N-123.89E', 'C-24.06N-5.63E', 'C15.73N-96.74E', 'C-24.92N121.61E', 'C-1.99N155.12E', 'C-4.75N134.34E', 'C36.93N-141.76E', 'C-29.13N147.83E', 'C-28.92N70.02E', 'C-18.37N84.27E', 'C10.79N-115.01E', 'C-31.94N21.89E', 'C-15.26N46.02E', 'C-33.91N-12.74E', 'C-30.60N5.13E', 'C14.43N79.52E', 'C20.57N117.79E', 'C9.01N-100.64E', 'C-10.67N76.28E', 'C18.07N92.62E', 'C24.59N152.25E', 'C29.72N-3.99E', 'C34.42N56.66E', 'C-34.78N23.46E', 'C13.32N104.56E', 'C5.14N23.32E', 'C-14.38N-129.16E', 'C-14.76N64.56E', 'C-24.73N-65.38E', 'C-21.49N-49.34E', 'C-30.60N145.63E', 'C5.10N166.11E', 'C-43.87N143.30E', 'C-24.55N-63.81E', 'C35.97N102.89E']
    EXCLUDE_KEYS = OOB_KEYS + NOEXP_KEYS
    for KEY in EXCLUDE_KEYS:
        if KEY in cdict.cIDs:
            cdict.cIDs.remove(KEY)
            cdict.pop(KEY, None)

#%%

Tend = timer() #Startup timer
print('Startup Time = {}s \n'.format(Tend-Tstart),flush=True)

#%% MAIN STEP
if __name__ == '__main__': 
    # Crater Lists
    LUlt8 = LUdict.cIDs[0:55873] # 240m <= diam <= 8km
    LUlt10 = LUdict.cIDs[55873:59576] # 8km <= diam <= 10km
    LUlt20 = LUdict.cIDs[59576:70843] #10km <= diam < 20km
    LUlt30 = LUdict.cIDs[70843:74208] #20km <= diam < 30km 
    LUlt40 = LUdict.cIDs[74208:75752] #30km <= diam <= 40.523 (Aristarchus)
    
    GET_ACE = False
    if GET_ACE:
        targetIDs = LUlt8+LUlt10+LUlt20+LUlt30+LUlt40
        LUaceIDs = main(targetIDs, LUdict, OMATds)
    else:
        LUaceIDs = hf.readIDs(PATH+'LU_aceCraters_v1.7.txt',skiprows=1)
    
    VERIFY = False
    if VERIFY:
        start_ind = 4268 # 8km
        end_ind = -1
        print('\nVerify ace craters')
        vfIDs,vmfIDs,vnfIDs,vanomIDs = verify(LUdict,LUaceIDs[start_ind:end_ind],
                                              OMATds,PATH) 
    else:
        vfIDs = hf.readIDs(PATH+'vfIDs.txt',skiprows=1)
        vmfIDs = hf.readIDs(PATH+'vmfIDs.txt',skiprows=1)
        vnfIDs = hf.readIDs(PATH+'vnfIDs.txt',skiprows=1)
        vanomIDs = hf.readIDs(PATH+'vanomIDs.txt',skiprows=1)
   #%% 
    PLOT_SFD = False
    if PLOT_SFD:    
        # All craters
        pf.plot_sfd(LUdict, LUdict.cIDs, 1, 'fullmoon')
        d,sfd = sf.getSfd(LUdict, LUdict.cIDs, 1, 'fullmoon')
        pf.plotSfd(d,sfd)
        
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'fullmoon'),fig='ace',label='LUace')        
        pf.plotSfd(*sf.getSfd(SARAdict, SARAdict.cIDs, 1, 'fullmoon',ds='OMAT'),fig='ace',label='SARA') 
        
        # AceCraters
        aceIDs = np.concatenate((vfIDs,vmfIDs)) 
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'fullmoon'),fig='ace',label='LUace') 
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'leading'),fig='ace',label='lead',mark='<') 
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'trailing'),fig='ace',label='trail',mark='>') 
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'nearside'),fig='ace',label='near',mark='v') 
        pf.plotSfd(*sf.getSfd(LUdict, aceIDs, 1, 'farside'),fig='ace',label='far',mark='^') 

    
        # Longitude distributions
        pf.plot_lfd(LUdict, LUlt20+LUlt30+LUlt40)
        pf.plot_lfd(LUdict, aceIDs)
    
        # Compare and save plots        
        compare(aceIDs,LUdict,(OMATds,RAds),folder='v1.8figs/')
        
    #simpleIDs = [c for c in cIDs if cdict[c].diam <= 15.]
    #transIDs = [c for c in cIDs if (cdict[c].diam > 15.) & (cdict[c].diam <= 25.)]
    #complexIDs = [c for c in cIDs if cdict[c].diam > 25.]     
    #gt10IDs = [c for c in cdict.cIDs if (cdict[c].diam > 10.) & (cdict[c].diam <= 85.5)] # 10km to Tycho   
    
    #pf.plot_sfd(cdict,lt10ids+gt10ids, 1-np.cos(50*np.pi/180), 'leading')
    #pf.plot_sfd(cdict,lt10ids+gt10ids, 1-np.cos(50*np.pi/180), 'trailing')
#    pf.plot_sfd(cdict,ctudict.cIDs, 1-np.cos(50*np.pi/180),'leading')
#    pf.plot_sfd(cdict,ctudict.cIDs, 1-np.cos(50*np.pi/180),'trailing')


    #len([cdict[cid].diam for cid in cdict.cIDs])
    #targetIDs = cdict.cIDs
    #aceIDs = main(targetIDs[:78], cdict, OMATds)
    
    #pf.plot_metrics(aceIDs, cdict, METRICS)
    #pf.plot_medians

#    # Finding ACE THLDs
#    i = 1
#    q = False    
#    while q == False:
#        targetIDs = cdict.cIDs[i:i+10] # IDs to run through main script
#    
#    # Execute main to get crater stats, filter ace craters and then plot metrics
#        aceIDs = main(targetIDs)
#        vpIDs = verify(aceIDs)
#        
#        for cid in list(set(aceIDs) - set(vpIDs)):
#            print('\nExclude '+cdict[cid].name)
#            print('  ACEDOM= '+str(cdict[cid].stats['ACEDOM']))
#            print('  ACERNG= '+str(cdict[cid].stats['ACERNG']))
#    
#            
#        inpt = ''
#        while not inpt:
#            inpt = input('Type c to continue or q to quit: ')
#            if inpt == 'c':
#                continue
#            elif inpt == 'q':
#                q = True
#            else:
#                print('INVALID INPUT')
#                inpt = ''
#        i += 10
      

    
#%% VERIFICATION STEP #
#VERIFY = True
#if VERIFY:
#    start_ind = 4268 # 8km
#    print('\nVerify ace craters')
#    vfIDs, vmfIDs, vnfIDs, anomIDs = verify(LUdict,LUaceIDs[start_ind:],OMATds,PATH) # Verified positive IDs (fresh)
#    #fpIDs = list(set(LUIDs) - set(vpIDs)) # False positive IDs (not fresh)
    
#    print('\nVerify excluded craters')
#    non_aceIDs = list(set(targetIDs) - set(aceIDs)) 
#    fnIDs = verify(non_aceIDs) # False negative IDs (fresh)
#    vnIDs = list(set(non_aceIDs) - set(fnIDs)) 
#    
#    print('\nTotal Craters: {}'.format(len(targetIDs)))
#    print('  Successfully filtered: {}\n  False Positive: {}\n  False Negative: {}\n'.format(
#          len(vpIDs)+len(vnIDs),len(fpIDs),len(fnIDs)))
    
















