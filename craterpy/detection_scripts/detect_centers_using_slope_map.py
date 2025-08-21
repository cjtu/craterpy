#!/usr/bin/env python3
"""
Dual-Circle Crater Center Detection via Slope Minimization

This script extracts crater tiles, computes slope maps at native resolution, 
and finds optimal crater positions using a dual-circle approach:
1. Large circle (rim detection) - seeks low-slope rim areas
2. Small concentric circle (center detection) - seeks low-slope crater floor

Gradients computed at native DEM resolution before resizing,
combined with dual-circle evaluation for precise crater center location.

Example usage:

poetry run python path/to/detect_centers_using_slope_map.py \
    --raster data/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.vrt \
    --database data/database.geojson \
    --output-folder path/to/output \
    --tile-size 512 
    --rim-thickness 15 \
    --iterative-smooth-rounds 1 \
    --search-radius 80 \
    --step-size 2 \

"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import rasterio.warp
import rasterio.windows
import rasterio.transform
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
from craterpy import CraterDatabase


def compute_extended_finite_differences(dem, kernel_size):
    """Compute gradients using extended finite difference stencils."""
    half_size = kernel_size // 2
    h, w = dem.shape
    
    grad_x = np.zeros_like(dem)
    grad_y = np.zeros_like(dem)
    
    if kernel_size == 3:
        # 3-point central difference
        for i in range(1, h-1):
            for j in range(1, w-1):
                grad_x[i, j] = (dem[i, j+1] - dem[i, j-1]) / 2.0
                grad_y[i, j] = (dem[i+1, j] - dem[i-1, j]) / 2.0
                
    elif kernel_size == 5:
        # 5-point central difference
        for i in range(2, h-2):
            for j in range(2, w-2):
                grad_x[i, j] = (-dem[i, j+2] + 8*dem[i, j+1] - 8*dem[i, j-1] + dem[i, j-2]) / 12.0
                grad_y[i, j] = (-dem[i+2, j] + 8*dem[i+1, j] - 8*dem[i-1, j] + dem[i-2, j]) / 12.0
                
    elif kernel_size == 7:
        # 7-point central difference
        for i in range(3, h-3):
            for j in range(3, w-3):
                grad_x[i, j] = (dem[i, j-3] - 9*dem[i, j-2] + 45*dem[i, j-1] - 45*dem[i, j+1] + 9*dem[i, j+2] - dem[i, j+3]) / 60.0
                grad_y[i, j] = (dem[i-3, j] - 9*dem[i-2, j] + 45*dem[i-1, j] - 45*dem[i+1, j] + 9*dem[i+2, j] - dem[i+3, j]) / 60.0
                
    elif kernel_size >= 9:
        # For larger kernels, use numpy gradient with repeated application for smoothing effect
        dem_smooth = dem.copy()
        for _ in range(kernel_size // 3):
            grad_y_temp, grad_x_temp = np.gradient(dem_smooth.astype(float))
            dem_smooth = dem_smooth - 0.1 * np.sqrt(grad_x_temp**2 + grad_y_temp**2)  # Slight smoothing
        grad_y, grad_x = np.gradient(dem_smooth.astype(float))
    
    return grad_x, grad_y


def compute_gaussian_derivatives(dem, sigma, kernel_size):
    """Compute gradients using Gaussian derivative kernels."""
    # Create 1D Gaussian derivative kernel
    x = np.arange(kernel_size) - kernel_size // 2
    
    # Gaussian function: exp(-x^2 / (2*sigma^2))
    # Gaussian derivative: -x * exp(-x^2 / (2*sigma^2)) / sigma^2
    gaussian = np.exp(-x**2 / (2*sigma**2))
    gaussian_deriv = -x * gaussian / (sigma**2)
    
    # Normalize
    if np.sum(np.abs(gaussian_deriv)) > 0:
        gaussian_deriv = gaussian_deriv / np.sum(np.abs(gaussian_deriv))
    
    # Apply separable filtering
    grad_x = cv2.filter2D(dem, cv2.CV_64F, gaussian_deriv.reshape(1, -1))
    grad_y = cv2.filter2D(dem, cv2.CV_64F, gaussian_deriv.reshape(-1, 1))
    
    return grad_x, grad_y


def compute_extended_sobel(dem, kernel_size):
    """Compute gradients using extended Sobel-like kernels."""
    if kernel_size == 5:
        # 5x5 Sobel-like kernels
        sobel_x = np.array([
            [-1, -2, 0, 2, 1],
            [-2, -4, 0, 4, 2],
            [-4, -8, 0, 8, 4],
            [-2, -4, 0, 4, 2],
            [-1, -2, 0, 2, 1]
        ], dtype=np.float64) / 32.0
        
        sobel_y = sobel_x.T
        
    elif kernel_size == 7:
        # 7x7 Sobel-like kernels
        sobel_x = np.array([
            [-1, -2, -3, 0, 3, 2, 1],
            [-2, -4, -6, 0, 6, 4, 2],
            [-3, -6, -9, 0, 9, 6, 3],
            [-4, -8, -12, 0, 12, 8, 4],
            [-3, -6, -9, 0, 9, 6, 3],
            [-2, -4, -6, 0, 6, 4, 2],
            [-1, -2, -3, 0, 3, 2, 1]
        ], dtype=np.float64) / 84.0
        
        sobel_y = sobel_x.T
        
    elif kernel_size >= 9:
        # Generate larger Sobel-like kernels
        center = kernel_size // 2
        sobel_x = np.zeros((kernel_size, kernel_size))
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                if j < center:
                    sobel_x[i, j] = -(center - j) * max(1, center - abs(i - center))
                elif j > center:
                    sobel_x[i, j] = (j - center) * max(1, center - abs(i - center))
        
        # Normalize
        sobel_x = sobel_x / np.sum(np.abs(sobel_x))
        sobel_y = sobel_x.T
    
    grad_x = cv2.filter2D(dem, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(dem, cv2.CV_64F, sobel_y)
    
    return grad_x, grad_y


def compute_slope_map(dem, method='numpy', kernel_size=3, sigma=1.0, pre_smooth_sigma=0.0, 
                     contrast_gamma=1.0, iterative_smooth_rounds=0, verbose=False):
    """
    Compute slope magnitude from DEM using various gradient methods with pre-processing options.
    
    Args:
        dem: Input DEM array
        method: 'numpy', 'sobel', 'scharr', 'extended_fd', 'gaussian_deriv'
        kernel_size: Size of gradient kernel (3, 5, 7, 9, etc.)
        sigma: Standard deviation for Gaussian derivative method
        pre_smooth_sigma: Gaussian smoothing applied BEFORE gradient computation (0=none)
        contrast_gamma: Gamma correction for contrast enhancement (1.0=none, >1=more contrast)
        iterative_smooth_rounds: Number of iterative smoothing rounds (0=none)
        verbose: Print processing info
    
    Returns:
        slope_magnitude: 2D array of slope values
        dem_processed: DEM after all pre-processing (for visualization)
    """
    # Convert to float and get basic stats
    dem_float = dem.astype(float)
    
    if verbose:
        print(f"    Computing gradients using {method} method")
        print(f"    Kernel: {kernel_size}, σ: {sigma}, Pre-smooth: {pre_smooth_sigma}, γ: {contrast_gamma}")
        print(f"    Original DEM stats: min={dem_float.min():.3f}, max={dem_float.max():.3f}")
    
    # Step 1: Apply heavy pre-smoothing to eliminate blocky artifacts
    if pre_smooth_sigma > 0:
        dem_processed = cv2.GaussianBlur(dem_float, (0, 0), pre_smooth_sigma)
        if verbose:
            print(f"    After pre-smoothing: min={dem_processed.min():.3f}, max={dem_processed.max():.3f}")
    else:
        dem_processed = dem_float.copy()
    
    # Step 2: Apply iterative smoothing if requested
    if iterative_smooth_rounds > 0:
        for round_idx in range(iterative_smooth_rounds):
            # Alternate between Gaussian and bilateral filtering
            if round_idx % 2 == 0:
                dem_processed = cv2.GaussianBlur(dem_processed, (0, 0), pre_smooth_sigma * 0.5)
            else:
                # Convert to uint8 for bilateral filter
                dem_uint8 = ((dem_processed - dem_processed.min()) / 
                           (dem_processed.max() - dem_processed.min()) * 255).astype(np.uint8)
                dem_bilateral = cv2.bilateralFilter(dem_uint8, 9, 50, 50)
                # Convert back to original scale
                dem_processed = dem_bilateral.astype(np.float32) / 255.0 * \
                              (dem_processed.max() - dem_processed.min()) + dem_processed.min()
        
        if verbose:
            print(f"    After {iterative_smooth_rounds} smoothing rounds: "
                  f"min={dem_processed.min():.3f}, max={dem_processed.max():.3f}")
    
    # Step 3: Apply contrast enhancement (gamma correction)
    if contrast_gamma != 1.0:
        # Normalize to 0-1 range
        dem_norm = (dem_processed - dem_processed.min()) / (dem_processed.max() - dem_processed.min())
        # Apply gamma correction
        dem_gamma = np.power(dem_norm, contrast_gamma)
        # Scale back to original range
        dem_processed = dem_gamma * (dem_processed.max() - dem_processed.min()) + dem_processed.min()
        
        if verbose:
            print(f"    After gamma correction: min={dem_processed.min():.3f}, max={dem_processed.max():.3f}")
    
    # Step 4: Compute gradients on the pre-processed DEM
    if method == 'numpy':
        # Standard numpy gradient (1-pixel central difference)
        grad_y, grad_x = np.gradient(dem_processed)
        
    elif method == 'extended_fd':
        # Extended finite differences with larger stencils
        grad_x, grad_y = compute_extended_finite_differences(dem_processed, kernel_size)
        
    elif method == 'sobel':
        # Sobel operators (can be extended to larger kernels)
        if kernel_size == 3:
            grad_x = cv2.Sobel(dem_processed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(dem_processed, cv2.CV_64F, 0, 1, ksize=3)
        else:
            grad_x, grad_y = compute_extended_sobel(dem_processed, kernel_size)
            
    elif method == 'scharr':
        # Scharr operators (3x3 only, but can extend to larger via repeated application)
        if kernel_size == 3:
            grad_x = cv2.Scharr(dem_processed, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(dem_processed, cv2.CV_64F, 0, 1)
        else:
            # For larger kernels, use extended sobel instead
            grad_x, grad_y = compute_extended_sobel(dem_processed, kernel_size)
        
    elif method == 'gaussian_deriv':
        # Gaussian derivative kernels
        grad_x, grad_y = compute_gaussian_derivatives(dem_processed, sigma, kernel_size)
        
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    
    # Compute slope magnitude
    slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    if verbose:
        print(f"    Gradient X: min={grad_x.min():.6f}, max={grad_x.max():.6f}")
        print(f"    Gradient Y: min={grad_y.min():.6f}, max={grad_y.max():.6f}")
        print(f"    Final slope stats: min={slope_magnitude.min():.6f}, max={slope_magnitude.max():.6f}")
    
    return slope_magnitude, dem_processed


def smooth_or_upscale_tile(tile, method='gaussian', sigma=2.0, kernel_size=5, upscale_factor=None, verbose=False):
    """
    Apply smoothing or upscaling to reduce interpolation artifacts.
    
    Args:
        tile: Input tile array
        method: 'gaussian', 'median', 'bilateral', 'upscale', or 'none'
        sigma: Standard deviation for Gaussian blur
        kernel_size: Kernel size for median filter
        upscale_factor: Factor for upscaling (e.g., 2.0)
        verbose: Print processing info
    
    Returns:
        processed_tile: Smoothed/upscaled tile
    """
    if method == 'none':
        return tile
    
    # Convert to float for processing
    tile_float = tile.astype(np.float32)
    
    if verbose:
        print(f"    Applying {method} processing...")
    
    if method == 'gaussian':
        # Gaussian blur to smooth interpolation artifacts
        processed = cv2.GaussianBlur(tile_float, (0, 0), sigma)
        
    elif method == 'median':
        # Median filter to reduce noise while preserving edges
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        processed = cv2.medianBlur(tile_float.astype(np.uint8), kernel_size).astype(np.float32)
        
    elif method == 'bilateral':
        # Bilateral filter - smooths while preserving edges
        # Convert to uint8 for bilateral filter
        tile_uint8 = ((tile_float - tile_float.min()) / (tile_float.max() - tile_float.min()) * 255).astype(np.uint8)
        processed_uint8 = cv2.bilateralFilter(tile_uint8, kernel_size, sigma*20, sigma*20)
        # Convert back to original scale
        processed = processed_uint8.astype(np.float32) / 255.0 * (tile_float.max() - tile_float.min()) + tile_float.min()
        
    elif method == 'upscale':
        if upscale_factor is None:
            upscale_factor = 2.0
        # Upscale using bicubic interpolation then resize back
        h, w = tile_float.shape
        new_size = (int(w * upscale_factor), int(h * upscale_factor))
        upscaled = cv2.resize(tile_float, new_size, interpolation=cv2.INTER_CUBIC)
        # Resize back to original size with smoothing
        processed = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_AREA)
        
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    if verbose:
        diff = np.abs(processed - tile_float)
        print(f"    Max change: {diff.max():.3f}, Mean change: {diff.mean():.3f}")
    
    return processed


def extract_crater_tile_with_native_slopes(cdb, fraster, crater_index, tile_size=1024, roi_radius_factor=2.0,
                                         gradient_method='numpy', gradient_kernel_size=3, gradient_sigma=1.0, 
                                         pre_smooth_sigma=0.0, contrast_gamma=1.0, iterative_smooth_rounds=0, 
                                         verbose=False):
    """
    Extract crater tile and compute slopes at native resolution before resizing.
    
    This is the key innovation: slopes are computed at the DEM's native resolution
    after reprojection but BEFORE resizing to eliminate interpolation artifacts.
    
    Returns:
        tile: 2D numpy array of elevation data (resized to tile_size)
        slope_map: 2D numpy array of slope data (computed at native res, then resized)
        tile_native: 2D numpy array of elevation data at native resolution
        slope_map_native: 2D numpy array of slope data at native resolution
        dem_processed_native: 2D numpy array of pre-processed DEM at native resolution
        crater_info: Dictionary with crater metadata
    """
    # Add ROI circles to database if not present
    roi_column = f'crater_roi_{roi_radius_factor}'
    if roi_column not in cdb.data.columns:
        cdb.add_circles(roi_column, roi_radius_factor)
    
    # Get crater data
    crater_row = cdb.data.iloc[crater_index]
    center = crater_row['_center']
    roi = crater_row[roi_column]
    
    # Create local azimuthal equidistant projection
    local_crs = ProjectedCRS(
        name=f"AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)",
        conversion=AzimuthalEquidistantConversion(center.y, center.x),
        geodetic_crs=cdb._crs,
    )
    
    # Open source raster
    with rasterio.open(fraster) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_res_x = src_transform.a
        src_res_y = abs(src_transform.e)
    
    # Define output grid in projected coordinates
    to_projected = Transformer.from_crs(cdb._crs, local_crs, always_xy=True).transform
    dst_bounds = transform(to_projected, roi).bounds
    
    # Calculate native resolution in the projected space
    # Get a small lat/lon area and see how it maps to projected coordinates
    center_lon, center_lat = center.x, center.y
    test_lons = [center_lon - src_res_x/2, center_lon + src_res_x/2]
    test_lats = [center_lat - src_res_y/2, center_lat + src_res_y/2]
    proj_x, proj_y = to_projected(test_lons, test_lats)
    native_res_meters = (abs(proj_x[1] - proj_x[0]) + abs(proj_y[1] - proj_y[0])) / 2.0
    
    # Calculate high-resolution grid size (preserve native resolution)
    left, bottom, right, top = dst_bounds
    native_width = int(round((right - left) / native_res_meters))
    native_height = int(round((top - bottom) / native_res_meters))
    
    # Clamp to reasonable limits to avoid memory issues
    max_dimension = 5000  # Reasonable limit for processing
    if native_width > max_dimension or native_height > max_dimension:
        scale_factor = max_dimension / max(native_width, native_height)
        native_width = int(native_width * scale_factor)
        native_height = int(native_height * scale_factor)
        native_res_meters = native_res_meters / scale_factor
    
    if verbose:
        print(f"  Native resolution: {native_res_meters:.2f}m/pixel")
        print(f"  High-res grid: {native_width}x{native_height}")
        print(f"  Final tile: {tile_size}x{tile_size}")
    
    # Create high-resolution destination transform
    dst_transform_hires = rasterio.transform.from_bounds(
        *dst_bounds, width=native_width, height=native_height
    )
    
    # Get source window
    try:
        source_bounds_needed = rasterio.warp.transform_bounds(
            local_crs, src_crs, *dst_bounds
        )
        
        with rasterio.open(fraster) as src:
            window = rasterio.windows.from_bounds(
                *source_bounds_needed, transform=src.transform
            )
            tile_raw = src.read(1, window=window)
            tile_transform = src.window_transform(window)
    except Exception as e:
        print(f"Skipping crater {crater_index}: {e}")
        return None, None, None, None, None, None
    
    if tile_raw.shape[0] == 0 or tile_raw.shape[1] == 0:
        print(f"Skipping crater {crater_index}: Invalid tile shape {tile_raw.shape}")
        return None, None, None, None, None, None
    
    # Reproject to high-resolution square tile (preserving native resolution)
    tile_hires = np.empty((native_height, native_width), dtype=tile_raw.dtype)
    
    try:
        rasterio.warp.reproject(
            source=tile_raw,
            destination=tile_hires,
            src_transform=tile_transform,
            src_crs=src_crs,
            dst_transform=dst_transform_hires,
            dst_crs=local_crs,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    except Exception as e:
        print(f"Skipping crater {crater_index}: Reprojection failed: {e}")
        return None, None, None, None, None, None
    
    if verbose:
        print(f"  Computing slopes at native resolution...")
    
    # Compute slopes at native resolution
    slope_map_hires, dem_processed_native = compute_slope_map(
        tile_hires, method=gradient_method, kernel_size=gradient_kernel_size, 
        sigma=gradient_sigma, pre_smooth_sigma=pre_smooth_sigma, 
        contrast_gamma=contrast_gamma, iterative_smooth_rounds=iterative_smooth_rounds, 
        verbose=verbose
    )
    
    # Now resize both elevation and slope to final tile size
    tile_final = cv2.resize(tile_hires.astype(np.float32), (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    slope_map_final = cv2.resize(slope_map_hires.astype(np.float32), (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    
    if verbose:
        print(f"  Resized {native_width}x{native_height} -> {tile_size}x{tile_size}")
        print(f"  Slope stats: min={slope_map_final.min():.6f}, max={slope_map_final.max():.6f}")
    
    # Crater metadata
    crater_info = {
        'crater_index': crater_index,
        'center_lat': center.y,
        'center_lon': center.x,
        'diameter_km': crater_row.get('Diam_m', 0) / 1000.0,
        'roi_radius_factor': roi_radius_factor,
        'tile_size': tile_size,
        'native_resolution_m': native_res_meters,
        'native_size': f"{native_width}x{native_height}",
        'crater_id': crater_row.get('UniqueCraterID', f'crater_{crater_index}')
    }
    
    return tile_final, slope_map_final, tile_hires, slope_map_hires, dem_processed_native, crater_info


def create_dual_circle_mask(center_x, center_y, rim_radius, image_shape, rim_thickness=1, 
                           center_radius_factor=0.15):
    """
    Create dual circle masks: large rim circle + small center circle.
    
    Args:
        center_x, center_y: Circle center coordinates
        rim_radius: Radius of the rim detection circle
        image_shape: Shape of the image
        rim_thickness: Thickness of the rim circle (1=filled, >1=annular)
        center_radius_factor: Center circle radius as fraction of rim radius
    
    Returns:
        rim_mask: Boolean mask for rim circle
        center_mask: Boolean mask for center circle
        center_radius: Actual radius of center circle
    """
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    
    # Create rim circle mask
    if rim_thickness <= 1:
        # Filled circle
        rim_mask = (x - center_x)**2 + (y - center_y)**2 <= rim_radius**2
    else:
        # Annular region (ring)
        inner_radius = max(0, rim_radius - rim_thickness // 2)
        outer_radius = rim_radius + rim_thickness // 2
        
        outer_mask = (x - center_x)**2 + (y - center_y)**2 <= outer_radius**2
        inner_mask = (x - center_x)**2 + (y - center_y)**2 <= inner_radius**2
        rim_mask = outer_mask & ~inner_mask
    
    # Create center circle mask (always filled)
    center_radius = max(1, int(rim_radius * center_radius_factor))
    center_mask = (x - center_x)**2 + (y - center_y)**2 <= center_radius**2
    
    return rim_mask, center_mask, center_radius


def find_optimal_crater_position(slope_map, search_radius, rim_radius, rim_thickness=1, 
                                center_radius_factor=0.15, center_weight=2.0, step_size=2, verbose=False):
    """
    Find the position where dual circles have minimum combined slope using:
    - Large rim circle: seeks low-slope rim areas
    - Small center circle: seeks low-slope crater floor (weighted more heavily)
    
    Args:
        slope_map: 2D array of slope values
        search_radius: How far from center to search
        rim_radius: Radius of the rim detection circle
        rim_thickness: Thickness of rim circle (1=filled, >1=annular ring)
        center_radius_factor: Center circle radius as fraction of rim radius
        center_weight: Weight multiplier for center circle contribution
        step_size: Step size for sliding the circles
        verbose: Print search details
    
    Returns:
        best_center: (x, y) coordinates of optimal position
        min_combined_sum: Minimum combined slope value found
        search_results: Dictionary with detailed search statistics
    """
    h, w = slope_map.shape
    center_x, center_y = w // 2, h // 2
    
    best_center = (center_x, center_y)
    min_combined_sum = float('inf')
    best_rim_mask = None
    best_center_mask = None
    best_center_radius = 0
    
    # Store search results for visualization and analysis
    search_positions = []
    rim_slope_sums = []
    center_slope_sums = []
    combined_slope_sums = []
    rim_pixel_counts = []
    center_pixel_counts = []
    valid_positions = 0
    
    # Calculate effective search bounds considering both circles
    effective_radius = rim_radius + rim_thickness // 2
    
    # Search in a grid around the image center
    for dx in range(-search_radius, search_radius + 1, step_size):
        for dy in range(-search_radius, search_radius + 1, step_size):
            test_x = center_x + dx
            test_y = center_y + dy
            
            # Check if both circles fit within image bounds
            if (test_x - effective_radius < 0 or test_x + effective_radius >= w or
                test_y - effective_radius < 0 or test_y + effective_radius >= h):
                continue
            
            # Create dual circle masks
            rim_mask, center_mask, center_radius = create_dual_circle_mask(
                test_x, test_y, rim_radius, slope_map.shape, rim_thickness, center_radius_factor
            )
            
            # Compute slope sums for both circles
            rim_slope_sum = np.sum(slope_map[rim_mask])
            center_slope_sum = np.sum(slope_map[center_mask])
            
            # Combined evaluation: rim sum + weighted center sum
            combined_sum = rim_slope_sum + (center_weight * center_slope_sum)
            
            # Store results
            search_positions.append((test_x, test_y))
            rim_slope_sums.append(rim_slope_sum)
            center_slope_sums.append(center_slope_sum)
            combined_slope_sums.append(combined_sum)
            rim_pixel_counts.append(np.sum(rim_mask))
            center_pixel_counts.append(np.sum(center_mask))
            valid_positions += 1
            
            if combined_sum < min_combined_sum:
                min_combined_sum = combined_sum
                best_center = (test_x, test_y)
                best_rim_mask = rim_mask
                best_center_mask = center_mask
                best_center_radius = center_radius
    
    # Compute detailed statistics
    if combined_slope_sums:
        rim_slope_array = np.array(rim_slope_sums)
        center_slope_array = np.array(center_slope_sums)
        combined_slope_array = np.array(combined_slope_sums)
        rim_pixel_array = np.array(rim_pixel_counts)
        center_pixel_array = np.array(center_pixel_counts)
        
        # Statistics for each component
        rim_stats = {
            'min': np.min(rim_slope_array),
            'max': np.max(rim_slope_array),
            'mean': np.mean(rim_slope_array),
            'std': np.std(rim_slope_array)
        }
        
        center_stats = {
            'min': np.min(center_slope_array),
            'max': np.max(center_slope_array),
            'mean': np.mean(center_slope_array),
            'std': np.std(center_slope_array)
        }
        
        combined_stats = {
            'min': np.min(combined_slope_array),
            'max': np.max(combined_slope_array),
            'mean': np.mean(combined_slope_array),
            'std': np.std(combined_slope_array)
        }
        
        # Best position detailed stats
        if best_rim_mask is not None and best_center_mask is not None:
            best_rim_slopes = slope_map[best_rim_mask]
            best_center_slopes = slope_map[best_center_mask]
            
            best_position_stats = {
                'rim_pixels': np.sum(best_rim_mask),
                'center_pixels': np.sum(best_center_mask),
                'rim_slope_sum': np.sum(best_rim_slopes),
                'center_slope_sum': np.sum(best_center_slopes),
                'rim_mean_slope': np.mean(best_rim_slopes),
                'center_mean_slope': np.mean(best_center_slopes),
                'rim_std_slope': np.std(best_rim_slopes),
                'center_std_slope': np.std(best_center_slopes),
                'combined_sum': min_combined_sum,
                'center_radius': best_center_radius
            }
        else:
            best_position_stats = {
                'rim_pixels': 0, 'center_pixels': 0, 'rim_slope_sum': 0,
                'center_slope_sum': 0, 'rim_mean_slope': 0, 'center_mean_slope': 0,
                'rim_std_slope': 0, 'center_std_slope': 0, 'combined_sum': 0,
                'center_radius': 0
            }
    else:
        rim_stats = center_stats = combined_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        best_position_stats = {
            'rim_pixels': 0, 'center_pixels': 0, 'rim_slope_sum': 0,
            'center_slope_sum': 0, 'rim_mean_slope': 0, 'center_mean_slope': 0,
            'rim_std_slope': 0, 'center_std_slope': 0, 'combined_sum': 0,
            'center_radius': 0
        }
    
    if verbose:
        print(f"  Dual-Circle Search Statistics:")
        print(f"    Searched {valid_positions} valid positions")
        print(f"    Rim circle: radius={rim_radius}px, thickness={rim_thickness}px")
        print(f"    Center circle: radius={best_center_radius}px (factor={center_radius_factor:.2f}), weight={center_weight:.1f}x")
        if combined_slope_sums:
            print(f"    Rim slope sums: {rim_stats['min']:.2f} - {rim_stats['max']:.2f} (μ={rim_stats['mean']:.2f})")
            print(f"    Center slope sums: {center_stats['min']:.2f} - {center_stats['max']:.2f} (μ={center_stats['mean']:.2f})")
            print(f"    Combined sums: {combined_stats['min']:.2f} - {combined_stats['max']:.2f}")
            print(f"    Best position: rim_sum={best_position_stats['rim_slope_sum']:.2f}, "
                  f"center_sum={best_position_stats['center_slope_sum']:.2f}, "
                  f"combined={best_position_stats['combined_sum']:.2f}")
        print(f"    Optimal position: {best_center}")
    
    search_results = {
        'positions': search_positions,
        'rim_slope_sums': rim_slope_sums,
        'center_slope_sums': center_slope_sums,
        'combined_slope_sums': combined_slope_sums,
        'rim_pixel_counts': rim_pixel_counts,
        'center_pixel_counts': center_pixel_counts,
        'search_radius': search_radius,
        'rim_radius': rim_radius,
        'rim_thickness': rim_thickness,
        'center_radius_factor': center_radius_factor,
        'center_weight': center_weight,
        'step_size': step_size,
        'valid_positions': valid_positions,
        'rim_stats': rim_stats,
        'center_stats': center_stats,
        'combined_stats': combined_stats,
        'best_position_stats': best_position_stats
    }
    
    return best_center, min_combined_sum, search_results


def create_comprehensive_dual_circle_visualization(original_tile, smoothed_tile, tile_native, 
                                                   dem_processed_native, slope_map_native, slope_map_resized, 
                                                   optimal_center, rim_radius, crater_info, 
                                                   smoothing_info=None, search_results=None, output_path=None):
    """
    Create 8-panel visualization: Original | Smoothed | Native DEM | Processed DEM | Native Slope | Resized Slope | Dual Circles | Statistics
    """
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    
    # Normalize all tiles for display
    def normalize_for_display(tile):
        if tile.dtype != np.uint8:
            p1, p99 = np.percentile(tile, [1, 99])
            return np.clip((tile - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
        return tile
    
    display_tile = normalize_for_display(original_tile)
    display_smoothed = normalize_for_display(smoothed_tile)
    display_native = normalize_for_display(tile_native)
    display_processed = normalize_for_display(dem_processed_native)
    
    h, w = display_tile.shape
    h_native, w_native = tile_native.shape
    
    # Panel 1 (top-left): Original DEM
    axes[0,0].imshow(display_tile, cmap='gray', origin='upper')
    axes[0,0].set_title(f"Original DEM ({w}x{h})\nCrater {crater_info['crater_index']} "
                        f"(Ø{crater_info['diameter_km']:.1f}km)")
    axes[0,0].set_xlabel('X (pixels)')
    axes[0,0].set_ylabel('Y (pixels)')
    
    # Add crosshairs
    axes[0,0].axhline(h//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    axes[0,0].axvline(w//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    
    # Panel 2 (top-middle-left): Smoothed DEM
    axes[0,1].imshow(display_smoothed, cmap='gray', origin='upper')
    smoothing_text = smoothing_info if smoothing_info else "Processed"
    axes[0,1].set_title(f"Smoothed DEM ({w}x{h})\n{smoothing_text}")
    axes[0,1].set_xlabel('X (pixels)')
    axes[0,1].set_ylabel('Y (pixels)')
    
    axes[0,1].axhline(h//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    axes[0,1].axvline(w//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    
    # Panel 3 (top-middle-right): Native Resolution DEM
    axes[0,2].imshow(display_native, cmap='gray', origin='upper')
    native_res_text = f"{crater_info.get('native_resolution_m', 'N/A'):.1f}m/px"
    axes[0,2].set_title(f"Native Resolution DEM\n({w_native}x{h_native}, {native_res_text})")
    axes[0,2].set_xlabel('X (pixels)')
    axes[0,2].set_ylabel('Y (pixels)')
    
    axes[0,2].axhline(h_native//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    axes[0,2].axvline(w_native//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    
    # Panel 4 (top-right): Contrast-Enhanced DEM
    axes[0,3].imshow(display_processed, cmap='gray', origin='upper')
    axes[0,3].set_title(f"Contrast Enhanced DEM\n({w_native}x{h_native})")
    axes[0,3].set_xlabel('X (pixels)')
    axes[0,3].set_ylabel('Y (pixels)')
    
    axes[0,3].axhline(h_native//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    axes[0,3].axvline(w_native//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    
    # Panel 5 (bottom-left): Native Resolution Slope Map
    if slope_map_native.max() > slope_map_native.min():
        vmax_native = np.percentile(slope_map_native, 99)
        vmin_native = slope_map_native.min()
    else:
        vmax_native = slope_map_native.max()
        vmin_native = slope_map_native.min()
    
    im5 = axes[1,0].imshow(slope_map_native, cmap='hot', origin='upper', vmin=vmin_native, vmax=vmax_native)
    slope_range_native = f'{slope_map_native.min():.3e} - {slope_map_native.max():.3e}'
    axes[1,0].set_title(f'Native Resolution Slope Map\n({w_native}x{h_native})\nRange: {slope_range_native}')
    axes[1,0].set_xlabel('X (pixels)')
    axes[1,0].set_ylabel('Y (pixels)')
    plt.colorbar(im5, ax=axes[1,0], label='Slope Magnitude', fraction=0.046)
    
    # Panel 6 (bottom-middle-left): Resized Slope Map
    if slope_map_resized.max() > slope_map_resized.min():
        vmax_resized = np.percentile(slope_map_resized, 99)
        vmin_resized = slope_map_resized.min()
    else:
        vmax_resized = slope_map_resized.max()
        vmin_resized = slope_map_resized.min()
    
    im6 = axes[1,1].imshow(slope_map_resized, cmap='hot', origin='upper', vmin=vmin_resized, vmax=vmax_resized)
    slope_range_resized = f'{slope_map_resized.min():.3e} - {slope_map_resized.max():.3e}'
    axes[1,1].set_title(f'Resized Slope Map\n({w}x{h})\nRange: {slope_range_resized}')
    axes[1,1].set_xlabel('X (pixels)')
    axes[1,1].set_ylabel('Y (pixels)')
    plt.colorbar(im6, ax=axes[1,1], label='Slope Magnitude', fraction=0.046)
    
    # Panel 7 (bottom-middle-right): Dual Circle Position
    axes[1,2].imshow(slope_map_resized, cmap='hot', origin='upper', alpha=0.7, vmin=vmin_resized, vmax=vmax_resized)
    
    # Get circle parameters from search results
    rim_thickness = 1
    center_radius_factor = 0.15
    center_weight = 2.0
    center_radius = max(1, int(rim_radius * center_radius_factor))
    
    if search_results:
        rim_thickness = search_results.get('rim_thickness', 1)
        center_radius_factor = search_results.get('center_radius_factor', 0.15)
        center_weight = search_results.get('center_weight', 2.0)
        best_stats = search_results.get('best_position_stats', {})
        center_radius = best_stats.get('center_radius', center_radius)
    
    # Scale line width with rim thickness
    line_width = max(2, min(10, rim_thickness))
    
    # Draw rim circle
    if rim_thickness <= 1:
        # Filled rim circle
        rim_circle = plt.Circle(optimal_center, rim_radius, 
                               fill=False, color='lime', linewidth=3, alpha=0.8)
        axes[1,2].add_patch(rim_circle)
        
        rim_filled = plt.Circle(optimal_center, rim_radius, 
                               fill=True, color='lime', alpha=0.15, linewidth=0)
        axes[1,2].add_patch(rim_filled)
    else:
        # Annular rim region
        inner_radius = max(0, rim_radius - rim_thickness // 2)
        outer_radius = rim_radius + rim_thickness // 2
        
        # Draw filled annular region
        theta = np.linspace(0, 2*np.pi, 100)
        outer_x = optimal_center[0] + outer_radius * np.cos(theta)
        outer_y = optimal_center[1] + outer_radius * np.sin(theta)
        inner_x = optimal_center[0] + inner_radius * np.cos(theta[::-1])
        inner_y = optimal_center[1] + inner_radius * np.sin(theta[::-1])
        
        annular_x = np.concatenate([outer_x, inner_x])
        annular_y = np.concatenate([outer_y, inner_y])
        
        axes[1,2].fill(annular_x, annular_y, color='lime', alpha=0.25)
        
        # Draw rim circle outlines
        outer_circle = plt.Circle(optimal_center, outer_radius, 
                                 fill=False, color='lime', linewidth=3, linestyle='-', alpha=0.9)
        inner_circle = plt.Circle(optimal_center, inner_radius, 
                                 fill=False, color='lime', linewidth=3, linestyle='--', alpha=0.9)
        axes[1,2].add_patch(outer_circle)
        axes[1,2].add_patch(inner_circle)
    
    # Draw center circle (always filled)
    center_circle = plt.Circle(optimal_center, center_radius, 
                              fill=False, color='red', linewidth=max(2, line_width//2), alpha=0.9)
    axes[1,2].add_patch(center_circle)
    
    center_filled = plt.Circle(optimal_center, center_radius, 
                              fill=True, color='red', alpha=0.3, linewidth=0)
    axes[1,2].add_patch(center_filled)
    
    # Draw crosshairs and center marker
    axes[1,2].axhline(h//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    axes[1,2].axvline(w//2, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    
    marker_size = max(12, min(20, 12 + rim_thickness))
    marker_width = max(3, min(8, 3 + rim_thickness // 2))
    axes[1,2].plot(optimal_center[0], optimal_center[1], 'yellow', marker='+', 
                   markersize=marker_size, markeredgewidth=marker_width, alpha=0.9)
    
    # Calculate displacement
    displacement_x = optimal_center[0] - w//2
    displacement_y = optimal_center[1] - h//2
    displacement_mag = np.sqrt(displacement_x**2 + displacement_y**2)
    
    title_text = f'Dual-Circle Detection\n'
    title_text += f'Rim: r={rim_radius}px, Center: r={center_radius}px\n'
    title_text += f'Displacement: ({displacement_x:+.1f}, {displacement_y:+.1f}) = {displacement_mag:.1f}px'
    
    axes[1,2].set_title(title_text)
    axes[1,2].set_xlabel('X (pixels)')
    axes[1,2].set_ylabel('Y (pixels)')
    
    # Panel 8 (bottom-right): Statistics Summary
    axes[1,3].axis('off')  # Turn off axis for text panel
    
    if search_results and 'best_position_stats' in search_results:
        stats = search_results['best_position_stats']
        rim_stats = search_results['rim_stats']
        center_stats = search_results['center_stats']
        combined_stats = search_results['combined_stats']
        
        stats_text = f"Search Statistics\n"
        stats_text += f"Positions checked: {search_results['valid_positions']}\n\n"
        
        stats_text += f"Rim Circle (Lime):\n"
        stats_text += f"  Radius: {rim_radius}px, Thickness: {rim_thickness}px\n"
        stats_text += f"  Pixels: {stats['rim_pixels']}\n"
        stats_text += f"  Slope sum: {stats['rim_slope_sum']:.2f}\n"
        stats_text += f"  Mean slope: {stats['rim_mean_slope']:.2e}\n"
        stats_text += f"  Search range: {rim_stats['min']:.1f} - {rim_stats['max']:.1f}\n\n"
        
        stats_text += f"Center Circle (Red):\n"
        stats_text += f"  Radius: {center_radius}px (factor: {center_radius_factor:.2f})\n"
        stats_text += f"  Weight: {center_weight:.1f}x\n"
        stats_text += f"  Pixels: {stats['center_pixels']}\n"
        stats_text += f"  Slope sum: {stats['center_slope_sum']:.2f}\n"
        stats_text += f"  Weighted contribution: {stats['center_slope_sum'] * center_weight:.2f}\n"
        stats_text += f"  Mean slope: {stats['center_mean_slope']:.2e}\n"
        stats_text += f"  Search range: {center_stats['min']:.1f} - {center_stats['max']:.1f}\n\n"
        
        stats_text += f"Combined Evaluation:\n"
        stats_text += f"  Total score: {stats['combined_sum']:.2f}\n"
        stats_text += f"  Search range: {combined_stats['min']:.1f} - {combined_stats['max']:.1f}\n"
        stats_text += f"  Score = rim_sum + {center_weight:.1f} × center_sum"
        
        axes[1,3].text(0.05, 0.95, stats_text, transform=axes[1,3].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Main title with comprehensive information
    if search_results:
        n_positions = search_results['valid_positions']
        best_stats = search_results.get('best_position_stats', {})
        
        main_title = (f"Dual-Circle Crater Center Detection\n"
                     f"ID: {crater_info['crater_id']} | Searched {n_positions} positions | "
                     f"Best score: {best_stats.get('combined_sum', 0):.1f}")
    else:
        main_title = f"Dual-Circle Crater Center Detection\nID: {crater_info['crater_id']}"
    
    fig.suptitle(main_title, fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved: {output_path}")
    
    return fig


def process_multiple_craters_dual_circle(database_path, raster_path, output_folder, num_craters=100,
                                        tile_size=1024, roi_radius=1.8, search_radius=50, step_size=2,
                                        rim_thickness=1, center_radius_factor=0.15, center_weight=2.0,
                                        smoothing_method='gaussian', smoothing_sigma=2.0, smoothing_kernel=5,
                                        upscale_factor=2.0, gradient_method='numpy', gradient_kernel_size=3, 
                                        gradient_sigma=1.0, pre_smooth_sigma=0.0, contrast_gamma=1.0, 
                                        iterative_smooth_rounds=0, verbose=False):
    """
    Process multiple craters with dual-circle crater center detection.
    
    Uses both rim detection (large circle) and center floor detection (small circle)
    for precise crater center location at native resolution.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load crater database
    print(f"Loading crater database from {database_path}")
    df = gpd.read_file(database_path)
    
    # Ensure diameter and radius columns exist
    if 'radius_km' not in df.columns:
        if 'Diam_m' in df.columns:
            df.insert(1, 'radius_km', df['Diam_m'].apply(lambda x: x / 2000))
        else:
            print("Warning: No diameter information found. Using default radius.")
            df['radius_km'] = 1.0  # Default 1km radius
    
    # Create CraterDatabase
    db = CraterDatabase(df, body='Moon', units='km')
    
    print(f"Processing {num_craters} craters...")
    print(f"Tile size: {tile_size}x{tile_size}, ROI radius: {roi_radius}x crater radius")
    print(f"Smoothing: {smoothing_method} (sigma={smoothing_sigma}, kernel={smoothing_kernel})")
    print(f"Gradient: {gradient_method} (kernel_size={gradient_kernel_size}, sigma={gradient_sigma})")
    print(f"Pre-processing: pre_smooth_sigma={pre_smooth_sigma}, gamma={contrast_gamma}, iter_rounds={iterative_smooth_rounds}")
    print(f"Dual-circle: rim_thickness={rim_thickness}px, center_factor={center_radius_factor:.2f}, center_weight={center_weight:.1f}x")
    print(f"Search: radius={search_radius}px, step_size={step_size}px")
    print(f"*** DUAL-CIRCLE NATIVE RESOLUTION CRATER CENTER DETECTION ***")
    
    # Calculate rim radius based on ROI radius
    rim_radius = int(tile_size / (2 * roi_radius))
    center_radius = max(1, int(rim_radius * center_radius_factor))
    print(f"Calculated rim radius: {rim_radius}px, center radius: {center_radius}px")
    
    results = []
    successful_tiles = 0
    
    for crater_idx in range(min(num_craters, len(df))):
        print(f"\nProcessing crater {crater_idx + 1}/{num_craters} (index {crater_idx})...")
        
        # Extract crater tile with native slope computation
        tile, slope_map_native_resized, tile_native, slope_map_native, dem_processed_native, crater_info = extract_crater_tile_with_native_slopes(
            db, raster_path, crater_idx, tile_size, roi_radius,
            gradient_method, gradient_kernel_size, gradient_sigma, 
            pre_smooth_sigma, contrast_gamma, iterative_smooth_rounds, verbose
        )
        
        if tile is None or slope_map_native_resized is None or tile_native is None or slope_map_native is None or crater_info is None:
            continue
        
        successful_tiles += 1
        
        if verbose:
            print(f"  Crater: {crater_info['crater_id']}, Diameter: {crater_info['diameter_km']:.1f}km")
            print(f"  Native resolution: {crater_info['native_resolution_m']:.2f}m/pixel")
            print(f"  Native size: {crater_info['native_size']}")
        
        # Apply smoothing to elevation tile for visualization
        smoothed_tile = smooth_or_upscale_tile(
            tile, method=smoothing_method, sigma=smoothing_sigma, 
            kernel_size=smoothing_kernel, upscale_factor=upscale_factor, verbose=verbose
        )
        
        # Create smoothing info text
        if smoothing_method == 'gaussian':
            smoothing_info = f"Gaussian (σ={smoothing_sigma})"
        elif smoothing_method == 'median':
            smoothing_info = f"Median (k={smoothing_kernel})"
        elif smoothing_method == 'bilateral':
            smoothing_info = f"Bilateral (σ={smoothing_sigma}, k={smoothing_kernel})"
        elif smoothing_method == 'upscale':
            smoothing_info = f"Upscale ({upscale_factor}x)"
        elif smoothing_method == 'none':
            smoothing_info = "No smoothing"
        else:
            smoothing_info = smoothing_method
        
        # Use the native-resolution slope map for dual-circle detection
        slope_map = slope_map_native_resized
        
        # Find optimal crater position using dual circles
        optimal_center, min_combined_sum, search_results = find_optimal_crater_position(
            slope_map, search_radius, rim_radius, rim_thickness, 
            center_radius_factor, center_weight, step_size, verbose
        )
        
        # Create visualization
        output_path = os.path.join(output_folder, f"crater_{crater_idx:03d}_dual_circle_detection.png")
        fig = create_comprehensive_dual_circle_visualization(
            tile, smoothed_tile, tile_native, dem_processed_native, slope_map_native, slope_map_native_resized,
            optimal_center, rim_radius, crater_info, smoothing_info, search_results, output_path
        )
        plt.close(fig)
        
        # Save raw data including all processing stages
        tile_normalized = ((tile - tile.min()) / (tile.max() - tile.min()) * 255).astype(np.uint8)
        Image.fromarray(tile_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_tile.png")
        )
        
        smoothed_normalized = ((smoothed_tile - smoothed_tile.min()) / (smoothed_tile.max() - smoothed_tile.min()) * 255).astype(np.uint8)
        Image.fromarray(smoothed_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_smoothed.png")
        )
        
        # Save native resolution data
        tile_native_normalized = ((tile_native - tile_native.min()) / (tile_native.max() - tile_native.min()) * 255).astype(np.uint8)
        Image.fromarray(tile_native_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_native_dem.png")
        )
        
        dem_processed_normalized = ((dem_processed_native - dem_processed_native.min()) / (dem_processed_native.max() - dem_processed_native.min()) * 255).astype(np.uint8)
        Image.fromarray(dem_processed_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_processed_dem.png")
        )
        
        # Save slope maps
        if slope_map_native.max() > slope_map_native.min():
            slope_native_normalized = ((slope_map_native - slope_map_native.min()) / (slope_map_native.max() - slope_map_native.min()) * 255).astype(np.uint8)
        else:
            slope_native_normalized = np.full_like(slope_map_native, 128, dtype=np.uint8)
        
        Image.fromarray(slope_native_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_native_slope.png")
        )
        
        if slope_map.max() > slope_map.min():
            slope_normalized = ((slope_map - slope_map.min()) / (slope_map.max() - slope_map.min()) * 255).astype(np.uint8)
        else:
            slope_normalized = np.full_like(slope_map, 128, dtype=np.uint8)
        
        Image.fromarray(slope_normalized).save(
            os.path.join(output_folder, f"crater_{crater_idx:03d}_slope.png")
        )
        
        # Store comprehensive results including dual-circle statistics
        displacement_x = optimal_center[0] - tile_size//2
        displacement_y = optimal_center[1] - tile_size//2
        displacement_mag = np.sqrt(displacement_x**2 + displacement_y**2)
        
        # Extract detailed statistics from search results
        best_stats = search_results.get('best_position_stats', {})
        rim_stats = search_results.get('rim_stats', {})
        center_stats = search_results.get('center_stats', {})
        combined_stats = search_results.get('combined_stats', {})
        
        results.append({
            'crater_index': crater_idx,
            'crater_id': crater_info['crater_id'],
            'center_lat': crater_info['center_lat'],
            'center_lon': crater_info['center_lon'],
            'diameter_km': crater_info['diameter_km'],
            'native_resolution_m': crater_info['native_resolution_m'],
            'native_size': crater_info['native_size'],
            'optimal_center_x': optimal_center[0],
            'optimal_center_y': optimal_center[1],
            'displacement_x': displacement_x,
            'displacement_y': displacement_y,
            'displacement_magnitude': displacement_mag,
            'rim_radius': rim_radius,
            'rim_thickness': rim_thickness,
            'center_radius': best_stats.get('center_radius', center_radius),
            'center_radius_factor': center_radius_factor,
            'center_weight': center_weight,
            'rim_pixel_count': best_stats.get('rim_pixels', 0),
            'center_pixel_count': best_stats.get('center_pixels', 0),
            'rim_slope_sum': best_stats.get('rim_slope_sum', 0),
            'center_slope_sum': best_stats.get('center_slope_sum', 0),
            'combined_slope_sum': best_stats.get('combined_sum', 0),
            'rim_mean_slope': best_stats.get('rim_mean_slope', 0),
            'center_mean_slope': best_stats.get('center_mean_slope', 0),
            'search_positions_checked': search_results.get('valid_positions', 0),
            'rim_search_min': rim_stats.get('min', 0),
            'rim_search_max': rim_stats.get('max', 0),
            'center_search_min': center_stats.get('min', 0),
            'center_search_max': center_stats.get('max', 0),
            'combined_search_min': combined_stats.get('min', 0),
            'combined_search_max': combined_stats.get('max', 0),
            'gradient_method': gradient_method,
            'gradient_params': f"kernel_size={gradient_kernel_size}, sigma={gradient_sigma}",
            'pre_processing_params': f"pre_smooth={pre_smooth_sigma}, gamma={contrast_gamma}, iter={iterative_smooth_rounds}",
            'smoothing_method': smoothing_method,
            'smoothing_params': f"sigma={smoothing_sigma}, kernel={smoothing_kernel}",
            'tile_filename': f"crater_{crater_idx:03d}_tile.png",
            'smoothed_filename': f"crater_{crater_idx:03d}_smoothed.png",
            'native_dem_filename': f"crater_{crater_idx:03d}_native_dem.png",
            'processed_dem_filename': f"crater_{crater_idx:03d}_processed_dem.png",
            'native_slope_filename': f"crater_{crater_idx:03d}_native_slope.png",
            'slope_filename': f"crater_{crater_idx:03d}_slope.png",
            'visualization_filename': f"crater_{crater_idx:03d}_dual_circle_detection.png"
        })
        
        if verbose:
            print(f"  Displacement: ({displacement_x:+.1f}, {displacement_y:+.1f}) = {displacement_mag:.1f}px")
            print(f"  Combined score: {best_stats.get('combined_sum', 0):.2f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "dual_circle_crater_detection_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_tiles}/{num_craters} craters")
    print(f"Results saved to: {csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Dual-Circle Crater Center Detection via Slope Minimization")
    parser.add_argument("--raster", type=str, required=True,
                       help="Path to DEM raster file")
    parser.add_argument("--database", type=str, required=True,
                       help="Path to crater database (GeoJSON)")
    parser.add_argument("--output-folder", type=str, required=True,
                       help="Output folder for results")
    parser.add_argument("--num-craters", type=int, default=100,
                       help="Number of craters to process (default: 100)")
    parser.add_argument("--tile-size", type=int, default=1024,
                       help="Tile size in pixels (default: 1024)")
    parser.add_argument("--roi-radius", type=float, default=1.8,
                       help="ROI radius in crater radii (default: 1.8)")
    parser.add_argument("--search-radius", type=int, default=50,
                       help="Search radius for optimal position in pixels (default: 50)")
    parser.add_argument("--step-size", type=int, default=2,
                       help="Step size for circle sliding in pixels (default: 2)")
    
    # Dual-circle parameters
    parser.add_argument("--rim-thickness", type=int, default=15,
                       help="Rim circle thickness in pixels: 1=filled circle, >1=annular ring (default: 15)")
    parser.add_argument("--center-radius-factor", type=float, default=0.18,
                       help="Center circle radius as fraction of rim radius (default: 0.18)")
    parser.add_argument("--center-weight", type=float, default=2.0,
                       help="Weight multiplier for center circle contribution (default: 2.0)")
    
    # Smoothing/upscaling parameters
    parser.add_argument("--smoothing-method", type=str, default='gaussian',
                       choices=['none', 'gaussian', 'median', 'bilateral', 'upscale'],
                       help="Smoothing method for elevation data visualization (default: gaussian)")
    parser.add_argument("--smoothing-sigma", type=float, default=2.0,
                       help="Sigma parameter for Gaussian/bilateral smoothing (default: 2.0)")
    parser.add_argument("--smoothing-kernel", type=int, default=5,
                       help="Kernel size for median/bilateral filters (default: 5)")
    parser.add_argument("--upscale-factor", type=float, default=2.0,
                       help="Upscaling factor for upscale method (default: 2.0)")
    
    # Gradient computation parameters
    parser.add_argument("--gradient-method", type=str, default='numpy',
                        choices=['numpy', 'sobel', 'scharr', 'extended_fd', 'gaussian_deriv'],
                        help="Gradient computation method (default: numpy)")
    parser.add_argument("--gradient-kernel-size", type=int, default=5,
                        help="Gradient kernel size - larger = smoother gradients (default: 3)")
    parser.add_argument("--gradient-sigma", type=float, default=1.0,
                        help="Sigma for Gaussian derivative method (default: 1.0)")
    
    # Pre-processing parameters to combat blocky artifacts
    parser.add_argument("--pre-smooth-sigma", type=float, default=0.8,
                        help="Gaussian smoothing applied BEFORE gradient computation to reduce blockiness (default: 0.0=none)")
    parser.add_argument("--contrast-gamma", type=float, default=1.5,
                        help="Gamma correction for contrast enhancement: >1=more contrast, <1=less contrast (default: 1.0=none)")
    parser.add_argument("--iterative-smooth-rounds", type=int, default=0,
                        help="Number of iterative smoothing rounds before gradient computation (default: 0=none)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed processing information")
    
    args = parser.parse_args()
    
    # Set environment variable for celestial body projections
    os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'
    
    print("Dual-Circle Crater Center Detection via Slope Minimization")
    print("=" * 65)
    print(f"Database: {args.database}")
    print(f"Raster: {args.raster}")
    print(f"Output: {args.output_folder}")
    print(f"Processing {args.num_craters} craters")
    print(f"Smoothing: {args.smoothing_method} (σ={args.smoothing_sigma}, k={args.smoothing_kernel})")
    print(f"Gradient: {args.gradient_method} (kernel_size={args.gradient_kernel_size}, σ={args.gradient_sigma})")
    print(f"Pre-processing: pre_smooth={args.pre_smooth_sigma}, γ={args.contrast_gamma}, iter={args.iterative_smooth_rounds}")
    print(f"Dual-circle: rim_thickness={args.rim_thickness}px, center_factor={args.center_radius_factor:.2f}, center_weight={args.center_weight:.1f}x")
    print()
    
    try:
        results = process_multiple_craters_dual_circle(
            args.database, args.raster, args.output_folder,
            num_craters=args.num_craters,
            tile_size=args.tile_size,
            roi_radius=args.roi_radius,
            search_radius=args.search_radius,
            step_size=args.step_size,
            rim_thickness=args.rim_thickness,
            center_radius_factor=args.center_radius_factor,
            center_weight=args.center_weight,
            smoothing_method=args.smoothing_method,
            smoothing_sigma=args.smoothing_sigma,
            smoothing_kernel=args.smoothing_kernel,
            upscale_factor=args.upscale_factor,
            gradient_method=args.gradient_method,
            gradient_kernel_size=args.gradient_kernel_size,
            gradient_sigma=args.gradient_sigma,
            pre_smooth_sigma=args.pre_smooth_sigma,
            contrast_gamma=args.contrast_gamma,
            iterative_smooth_rounds=args.iterative_smooth_rounds,
            verbose=args.verbose
        )
        
        if results:
            # Print summary statistics
            displacements = [r['displacement_magnitude'] for r in results]
            combined_scores = [r['combined_slope_sum'] for r in results if r['combined_slope_sum'] > 0]
            
            print(f"\nDisplacement Statistics:")
            print(f"Mean displacement: {np.mean(displacements):.2f} pixels")
            print(f"Median displacement: {np.median(displacements):.2f} pixels")
            print(f"Max displacement: {np.max(displacements):.2f} pixels")
            print(f"Std displacement: {np.std(displacements):.2f} pixels")
            
            # Print native resolution statistics
            native_resolutions = [r['native_resolution_m'] for r in results]
            print(f"\nNative Resolution Statistics:")
            print(f"Mean resolution: {np.mean(native_resolutions):.2f} m/pixel")
            print(f"Resolution range: {np.min(native_resolutions):.2f} - {np.max(native_resolutions):.2f} m/pixel")
            
            # Print dual-circle analysis statistics
            rim_pixel_counts = [r['rim_pixel_count'] for r in results if r['rim_pixel_count'] > 0]
            center_pixel_counts = [r['center_pixel_count'] for r in results if r['center_pixel_count'] > 0]
            
            if rim_pixel_counts and center_pixel_counts:
                print(f"\nDual-Circle Analysis Statistics:")
                print(f"Rim circle pixels: {np.min(rim_pixel_counts):.0f} - {np.max(rim_pixel_counts):.0f} "
                      f"(mean: {np.mean(rim_pixel_counts):.0f})")
                print(f"Center circle pixels: {np.min(center_pixel_counts):.0f} - {np.max(center_pixel_counts):.0f} "
                      f"(mean: {np.mean(center_pixel_counts):.0f})")
                
                if combined_scores:
                    print(f"Combined scores: {np.min(combined_scores):.1f} - {np.max(combined_scores):.1f} "
                          f"(mean: {np.mean(combined_scores):.1f})")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 1
    except Exception as e:
        print(f"Error processing craters: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
