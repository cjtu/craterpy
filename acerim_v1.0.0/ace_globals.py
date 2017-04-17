#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:25:37 2016

@author: christian
"""
RMOON = 1737400.0

# Choose dataset
MODE = 'OMAT'
#MODE = 'RA'
if MODE == 'OMAT':
    # Dataset location and name
    PATH = "/Users/christian/Desktop/Acerim/Craterlists and Datasets/"  # Directory
    DSNAME = "Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.tif"
    # Dataset extents (degrees)
    NLAT = 50  
    WLON = -180
    SLAT = -50
    ELON = 180
    # Pixel size
    PPD = 512  # pixels/degree
    MPP = 60  # meters/pixel
    # Plotting
    PLTMIN = 0
    PLTMAX = 0.4
    CMAP = 'gray'

elif MODE == 'RA':
    # Dataset location and name
    PATH = "/Users/christian/Desktop/Acerim/Craterlists and Datasets/"  # Directory
    DSNAME = "dgdr_ra_slope_cyl_avg_128.tif"
    # Dataset extents (degrees)
    NLAT = 80  
    WLON = 0
    SLAT = -80
    ELON = 360
    # Pixel size
    PPD = 128  # pixels/degree
    MPP = 240  # meters/pixel
    # Plotting
    PLTMIN = 0
    PLTMAX = 0.01
    CMAP = 'jet'