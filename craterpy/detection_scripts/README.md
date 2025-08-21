# Crater Center Detection

The scripts in this folder are designed to help correct crater databases that have small errors in the lat/lon of the crater centers. Specifically, the utilities in this folder were designed to treat the following specific problem: Suppose a crater database exists in which each crater's listed diameter is correct, but its center could be incorrect by circa one crater radius. These scripts are designed to use high-resolution DEMs (or similar files) to extract small azimuthal equidistant tiles centered on the crater (using the erroneous center from the database) and then apply either thresholding or slope-map minimization to find the candidate correct crater center.

## `detect_centers_by_thresholding.py`

This script uses thresholding, and does not rely on the condition that the crater database must have correct crater diameters. The only condition that this script relies on (loosely) is that the "true" crater center is within one crater radius of the listed crater center. However, this script could theoretically be used even in the case where the true crater center is more than one crater radius away from the listed center, simply by increasing the tile size; however, its performance would likely degrade for errors >> 1 crater radius.

Suggested paramters and example usage are provided in the module docstring at the top of the file.

## `detect_centers_using_slope_map.py`

This script uses slope map derivation and simple minimization to try to locate true crater centers. It relies on the fact that the crater database being corrected lists the crater diameter accurately. At a high level, it draws an annulus 15 pixels thick (by default) the same size as the crater being searched for, and a small concentric circle, and slides these two circles around preserving concentricity to try to minimize the values of the pixels they cover. High-value pixels are associated with steep slopes; low-value pixels with flat parts of the DEM. The rational being that around the crater rim, and in the middle of the crater floor, the slope ought to be low (flat). 

### Performance of the methods on example database using the example usage parameters provided in each script's docstring

The thresholding technique outperforms the slope map technique, though additional tuning of the parameters for the slope map technique could improve its performance. Based on a manual review of the first 200 craters in the database, the thresholding approach finds the crater center in 95% of cases. Based on a manual review of the first 25 craters in the same dataset, the slope map approach finds the crater center in 78% of the cases. (Fewer examples were generated and reviewed for the slope map approach due to its significantly higher computational complexity and thus slower runtime.)

Demo output from the two scripts can be easily viewed at: https://drive.google.com/drive/folders/15MZggL6KFPOlrNHH1JczqPIKrHfTAk6f?usp=sharing

## DEM download and processing

The DEM used in the example usage commands can be downloaded here (22 GB): https://astrogeology.usgs.gov/search/map/moon_lro_lola_selene_kaguya_tc_dem_merge_60n60s_59m

Other rasters may be preferable; these scripts were tested with this one, and it is likely that different parameter values would be required to match or exceed the performance mentioned above should a different raster be selected.

