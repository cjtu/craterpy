#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:25:37 2016

@author: christian
"""

def init():
    """
    Initialize all global variables.
    """
    import datetime
    now = datetime.datetime.now()    
    
    global RMOON,CPATH,METRICS,FS,FIGSIZE,DPI
    RMOON = 1737400.0 # meters
    #CPATH = "/Users/christian/Google Drive/Research/2016: OMAT & RA (Acerim)/Craterlists/" # Crater database directory
    CPATH = "C:\\Users\\Christian\\Google Drive\\Research\\2016_ OMAT & RA (Acerim)\\craterlists\\"
    METRICS = ['median', 'pct95']
    FS = 12 # FontSize for plotting
    FIGSIZE = (7,12)
    DPI = 100
    
    # NEUKUM PARAMS #
    global a_cfd, a_rfd
    a_cfd = [-3.0876, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058,
         0.019977, 0.086850, -0.005874, -0.006809, 8.25e-4, 5.54e-5]
    a_rfd = [0, 1.375458, 0.1272521, -1.282166, -0.3074558, 0.4149280, 0.1910668,
             -4.26098e-2, -3.976305e-2, -3.18010179e-3, 2.799369e-3, 6.892223e-4,
             2.614385e-6, -1.416178e-5, -1.191124e-6]
    # ACERIM PARAMS #
    global RMAX,DR,WSIZE,EXPDERIV_THLD,ACEDOM_THLD,ACERNG_THLD,ACESAVE_FILE,ACESAVE_TIME
    RMAX = 12 # Max radial extent to consider (in crater radii from center)
    DR = 0.2 #0.25  # Shell thickness in (crater radii)
    WSIZE = 1 # Size of moving window in number of shells
    EXPDERIV_THLD = -0.001 # Threshold for when derivative of exp reaches 0, see getAcedomain
    ACEDOM_THLD = 2.7 # Acerim domain low threshold
    ACERNG_THLD = 0.05 # Ace range low threshold
    ACESAVE_FILE = 'aceDict_'+now.strftime("%d-%b-%y_%H:%M")
    ACESAVE_TIME = 300 
    
    # VERIFIED PARAMS #
    global VERSAVE_FILE, VERSAVE_TIME
    VERSAVE_FILE = 'verDict_'+now.strftime("%d-%b-%y_%H:%M")
    VERSAVE_TIME = 120
    
    # FLAGS #
    global PLOT_ROI,PLOT_ROINF,PLOT_SHELLS,BG_GAUSS
    PLOT_ROI = False  # Set to True to plot ROI of each crater
    PLOT_ROINF = False # Plot roi with no floors (excludes crater interior)
    PLOT_SHELLS = False # Plot each ring/shell roi of each crater
    BG_GAUSS = False  # Set to True to fit a Gaussian to the background data



def init_ds(MODE):
    """
    Contains the information about a dataset required to initialize it. The MODE 
    refers to the specific dataset being imported.
    
    The AceDataset class will call this function to set its attributes, so the 
    correct values must be here under the corresponding MODE block. 
    """
    global PATH,DSNAME,NLAT,WLON,SLAT,ELON,PPD,MPP,PLTMIN,PLTMAX,CMAP
    PATH = "/Users/christian/Desktop/lunar_datasets/"  # Dataset Directory
    
    if MODE == 'OMAT': # For the Kaguya Optical Maturity 50N,50S dataset
        DSNAME = "Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.tif"
        # Extents
        NLAT = 50   # North extent [degrees]
        WLON = -180 # West extent [degrees]
        SLAT = -50  # South extent [degrees]
        ELON = 180  # East extent [degrees]
        # Map resolution / pixel size
        PPD = 512   # resolution [pixels/degree]
        MPP = 60    # pixel size [meters/pixel]
        # Plotting
        PLTMIN = 0  # Standard minimum data value for plotting
        PLTMAX = 0.4 # Standard maximum data value for plotting
        CMAP = 'gray' # colormap to use for plotting

    elif MODE == 'RA':
        DSNAME = "dgdr_ra_slope_cyl_avg_128.tif"
        NLAT = 80  
        WLON = 0
        SLAT = -80
        ELON = 360
        PPD = 128  
        MPP = 240  
        PLTMIN = 0
        PLTMAX = 0.02
        CMAP = 'jet'
        
    else:
        raise Exception('Invalid dataset initialization MODE')

if __name__=='__main__':
    init()