#!/usr/bin/env python3
"""
Crater Center Correction CLI

This script extracts crater tiles from a DEM, detects the actual crater centers,
and corrects the lat/lon coordinates based on the pixel displacement.

Example usage:

poetry run python path/to/detect_centers_by_thresholding.py \
    --mode extract \
    --raster path/to/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.vrt \
    --database path/to/database.geojson \
    --output-folder path/to/output/folder \
    --roi-radius 2 \
    --contrast-gamma 1.5 \
    --prefer-central \
    --num-craters 200 \
    --threshold 0.8

"""

import os
import sys
import argparse
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import rasterio.errors
import rasterio.windows
import rasterio.transform
import rasterio.warp
from pyproj import Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from craterpy import CraterDatabase


def extract_tiles_with_transforms(cdb, fraster, region="", index=None, band=1, shape=(None, None), partial_tiles=False):
    """
    Extract projected tiles from raster with coordinate transformation info.
    
    Returns tiles along with the transformation information needed to convert
    pixel coordinates back to lat/lon.
    """
    if not region:
        region = "_crater_rims"
        if region not in cdb.data.columns:
            cdb.data[region] = cdb._gen_annulus(0, 1, precise=True)
    
    gdf = cdb.data
    if isinstance(index, pd.Index):
        gdf = gdf.loc[index]
    elif isinstance(index, tuple):
        gdf = gdf.iloc[index[0]:index[1]]
    elif isinstance(index, int):
        gdf = gdf.iloc[index:index+1]

    tiles = []
    transform_info = []
    
    for i, row in gdf.iterrows():
        center = row['_center']
        roi = row[region]
        
        # Create local azimuthal equidistant projection
        local_crs = ProjectedCRS(
            name=f"AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)",
            conversion=AzimuthalEquidistantConversion(center.y, center.x),
            geodetic_crs=cdb._crs,
        )
        
        with rasterio.open(fraster) as src:
            src_crs = src.crs
            src_transform = src.transform
        
        # Define output grid
        to_projected = Transformer.from_crs(cdb._crs, local_crs, always_xy=True).transform
        dst_bounds = transform(to_projected, roi).bounds

        dst_width, dst_height = shape
        
        # Determine output shape if not specified
        if dst_height is None or dst_width is None:
            src_res_x = src_transform.a
            src_res_y = abs(src_transform.e)
            one_pixel_lon = [center.x - src_res_x / 2, center.x + src_res_x / 2]
            one_pixel_lat = [center.y - src_res_y / 2, center.y + src_res_y / 2]
            proj_x, proj_y = to_projected(one_pixel_lon, one_pixel_lat)
            
            avg_res_meters = (abs(proj_x[1] - proj_x[0]) + abs(proj_y[1] - proj_y[0])) / 2.0

            left, bottom, right, top = dst_bounds
            dst_width = int(round((right - left) / avg_res_meters))
            dst_height = int(round((top - bottom) / avg_res_meters))

        # Create destination transform
        dst_transform = rasterio.transform.from_bounds(
            *dst_bounds, width=dst_width, height=dst_height
        )

        # Get source bounds and window
        try:
            source_bounds_needed = rasterio.warp.transform_bounds(
                local_crs, src_crs, *dst_bounds
            )
            with rasterio.open(fraster) as src:
                window = rasterio.windows.from_bounds(
                    *source_bounds_needed, transform=src.transform
                )
        except rasterio.errors.WindowError as e:
            if not partial_tiles:
                print(f"Skipping index={int(i)} with error: {e}")
                continue
            window = rasterio.windows.from_bounds(*roi.bounds, transform=src_transform)

        # Read and reproject data
        with rasterio.open(fraster) as src:
            tile = src.read(band, window=window)
            tile_transform = src.window_transform(window)
            
        if tile.shape[0] == 0 or tile.shape[1] == 0:
            print(f"Skipping index={i}. Invalid tile shape: {tile.shape}")
            continue

        tile_projected = np.empty((dst_height, dst_width), dtype=tile.dtype)
        
        try:
            rasterio.warp.reproject(
                source=tile,
                destination=tile_projected,
                src_transform=tile_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=local_crs,
                resampling=rasterio.warp.Resampling.bilinear,
            )
        except Exception as e:
            print(f"Skipping index={i}. Reprojection failed: {e}")
            continue

        tiles.append(tile_projected)
        
        # Store transformation info for coordinate correction
        transform_info.append({
            'index': i,
            'center_lat': center.y,
            'center_lon': center.x,
            'local_crs': local_crs,
            'dst_transform': dst_transform,
            'dst_bounds': dst_bounds,
            'tile_shape': (dst_height, dst_width)
        })
        
        print(f"Processed index {i}: {tile.shape} -> {tile_projected.shape}")
    
    return tiles, transform_info


def find_crater_center(image_array, debug=False, threshold_factor=1.0, contrast_gamma=1.0, require_fully_enclosed=True, verbose=False, prefer_central=False):
    """
    Find the center of the largest dark spot (crater) in the image.
    
    Args:
        image_array: Input image array
        debug: Whether to create debug visualization
        threshold_factor: Multiplier for OTSU threshold (>1 = more restrictive)
        contrast_gamma: Gamma value for exponential contrast enhancement (>1 = more contrast)
        require_fully_enclosed: If True, only consider regions fully enclosed by image boundaries
        verbose: If True, print threshold information
        prefer_central: If True, prefer second-largest region if it's more central than largest
    
    Returns:
        center: (x, y) coordinates of detected center
        original_img: original image for visualization
        contrast_img: contrast-enhanced image for visualization
        overlay: debug visualization if debug=True
    """
    # Ensure we're working with uint8
    if image_array.dtype != np.uint8:
        img = ((image_array - image_array.min()) / 
               (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    else:
        img = image_array.copy()
    
    # Store original for visualization
    original_img = img.copy()
    
    # Apply exponential contrast enhancement if requested
    if contrast_gamma != 1.0:
        # Create lookup table for gamma correction
        gamma_table = np.array([((i / 255.0) ** contrast_gamma) * 255 for i in range(256)]).astype(np.uint8)
        img = cv2.LUT(img, gamma_table)
    
    # Store contrast-enhanced image for visualization
    contrast_img = img.copy()
    
    # Threshold with adjustable factor
    # OTSU automatically finds the optimal threshold value to separate foreground/background
    # by minimizing intra-class variance (i.e., maximizing between-class variance)
    otsu_thresh, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Apply threshold factor: higher values = more restrictive (select fewer/darker pixels)
    adjusted_thresh = otsu_thresh * threshold_factor
    
    # Apply the adjusted threshold with BINARY_INV:
    # - Pixels with intensity <= adjusted_thresh become WHITE (255) - these are "dark" crater pixels
    # - Pixels with intensity > adjusted_thresh become BLACK (0) - these are "bright" background pixels
    _, binary = cv2.threshold(img, adjusted_thresh, 255, cv2.THRESH_BINARY_INV)
    
    if verbose:
        print(f"OTSU threshold: {otsu_thresh:.1f}, Adjusted threshold: {adjusted_thresh:.1f}")
        print(f"Pixels selected as 'crater' (white): {np.sum(binary == 255)} / {binary.size} ({100*np.sum(binary == 255)/binary.size:.1f}%)")
    
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = clean.astype(bool)
    
    # Distance transform & peak finding
    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=20, labels=mask)
    
    if coords.dtype == bool:
        coords = np.column_stack(np.nonzero(coords))
    
    if len(coords) == 0:
        # Fallback: use image center
        center = (img.shape[1] // 2, img.shape[0] // 2)
        if debug:
            overlay = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(overlay, center, (255,0,0), cv2.MARKER_TILTED_CROSS, 20, 2)
            return center, original_img, contrast_img, overlay
        return center, original_img, contrast_img, None
    
    # Create markers
    markers = np.zeros(dist.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    
    # Watershed segmentation
    labels = watershed(-dist, markers, mask=mask)
    
    # Region properties & boundary filtering
    regions = regionprops(labels)
    if not regions:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        if debug:
            overlay = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(overlay, center, (255,0,0), cv2.MARKER_TILTED_CROSS, 20, 2)
            return center, original_img, contrast_img, overlay
        return center, original_img, contrast_img, None
    
    # Filter out regions that touch the boundary if requested
    if require_fully_enclosed:
        img_height, img_width = img.shape
        valid_regions = []
        for region in regions:
            # Check if any pixel of this region touches the boundary
            coords = region.coords
            touches_boundary = np.any((coords[:, 0] == 0) | (coords[:, 0] == img_height - 1) |
                                    (coords[:, 1] == 0) | (coords[:, 1] == img_width - 1))
            if not touches_boundary:
                valid_regions.append(region)
        regions = valid_regions
    
    if not regions:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        if debug:
            overlay = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(overlay, center, (255,0,0), cv2.MARKER_TILTED_CROSS, 20, 2)
            return center, original_img, contrast_img, overlay
        return center, original_img, contrast_img, None
    
    # Sort regions by area (largest first)
    regions_by_area = sorted(regions, key=lambda r: r.area, reverse=True)
    
    # Choose region based on prefer_central flag
    if prefer_central and len(regions_by_area) >= 2:
        largest = regions_by_area[0]
        second_largest = regions_by_area[1]
        
        # Calculate image center
        img_center_x = img.shape[1] / 2
        img_center_y = img.shape[0] / 2
        
        # Calculate distances from image center for both regions
        largest_centroid = largest.centroid  # (row, col) format
        second_centroid = second_largest.centroid
        
        # Distance from image center (note: centroid is in (row, col) but we want (x, y))
        largest_dist = np.sqrt((largest_centroid[1] - img_center_x)**2 + (largest_centroid[0] - img_center_y)**2)
        second_dist = np.sqrt((second_centroid[1] - img_center_x)**2 + (second_centroid[0] - img_center_y)**2)
        
        if second_dist < largest_dist:
            best = second_largest
            if verbose:
                print(f"Preferred central region: 2nd largest (area={second_largest.area}, dist={second_dist:.1f}) over largest (area={largest.area}, dist={largest_dist:.1f})")
        else:
            best = largest
            if verbose:
                print(f"Selected largest region: area={largest.area}, dist={largest_dist:.1f} vs 2nd largest dist={second_dist:.1f}")
    else:
        best = regions_by_area[0]  # Just take the largest
        if verbose and len(regions_by_area) > 1:
            print(f"Selected largest region: area={best.area} (prefer_central=False or only 1 region)")
    rr, cc = best.coords[:, 0], best.coords[:, 1]
    
    # Fit ellipse or circle
    pts = np.stack([cc, rr], axis=1).astype(np.int32)
    if pts.shape[0] >= 5:
        ellipse = cv2.fitEllipse(pts)
        (ex, ey), (ma, mi), angle = ellipse
        center = (int(ex), int(ey))
        draw_fn = lambda vis: cv2.ellipse(vis, center, (int(ma/2), int(mi/2)), angle, 0, 360, (0,255,0), 2)
    else:
        (ex, ey), radius = cv2.minEnclosingCircle(pts)
        center = (int(ex), int(ey))
        draw_fn = lambda vis: cv2.circle(vis, center, int(radius), (0,255,0), 2)
    
    # Create debug visualization
    overlay = None
    if debug:
        overlay = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
        overlay[mask] = (50,50,50)
        for (r, c) in coords:
            cv2.circle(overlay, (c, r), 3, (0,0,255), -1)
        
        # Draw all regions with different colors, highlighting the chosen one
        for i, lbl in enumerate(np.unique(labels)):
            if lbl == 0: continue
            seg = (labels == lbl).astype(np.uint8)*255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find which region this label corresponds to
            region_for_this_label = None
            for region in regions:
                if region.label == lbl:
                    region_for_this_label = region
                    break
            
            # Color coding: chosen region in yellow, others in cyan
            if region_for_this_label == best:
                cv2.drawContours(overlay, cnts, -1, (255,255,0), 2)  # Yellow for chosen
            else:
                cv2.drawContours(overlay, cnts, -1, (255,255,0), 1)  # Thin yellow for others
        
        draw_fn(overlay)
        cv2.drawMarker(overlay, center, (255,0,0), cv2.MARKER_TILTED_CROSS, 20, 2)
    
    return center, original_img, contrast_img, overlay


def pixel_to_latlon(pixel_center, transform_info):
    """
    Convert pixel coordinates to corrected lat/lon coordinates.
    
    Args:
        pixel_center: (x, y) pixel coordinates of detected crater center
        transform_info: Dictionary with transformation information
    
    Returns:
        corrected_lat, corrected_lon: Corrected coordinates
    """
    # Get image center in pixels
    img_height, img_width = transform_info['tile_shape']
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Calculate pixel displacement
    dx_pixels = pixel_center[0] - img_center_x
    dy_pixels = pixel_center[1] - img_center_y
    
    # Convert pixel displacement to projected coordinate displacement
    dst_transform = transform_info['dst_transform']
    
    # dst_transform maps from projected coords to pixels
    # We need the inverse to go from pixels to projected coords
    pixel_size_x = dst_transform.a  # x pixel size in projected units
    pixel_size_y = -dst_transform.e  # y pixel size in projected units (e is negative)
    
    # Convert pixel displacement to projected coordinate displacement
    dx_projected = dx_pixels * pixel_size_x
    dy_projected = dy_pixels * pixel_size_y
    
    # Get the center of the image in projected coordinates
    left, bottom, right, top = transform_info['dst_bounds']
    proj_center_x = (left + right) / 2
    proj_center_y = (bottom + top) / 2
    
    # Add displacement to get new projected coordinates
    new_proj_x = proj_center_x + dx_projected
    new_proj_y = proj_center_y + dy_projected
    
    # Convert from local projected CRS back to geographic coordinates
    local_crs = transform_info['local_crs']
    # Use the original lunar CRS from the crater database instead of EPSG:4326
    lunar_crs = local_crs.geodetic_crs  # This should be the lunar coordinate system
    transformer = Transformer.from_crs(local_crs, lunar_crs, always_xy=True)
    corrected_lon, corrected_lat = transformer.transform(new_proj_x, new_proj_y)
    
    return corrected_lat, corrected_lon


def process_existing_images(input_folder, output_folder, threshold_factor=1.0, contrast_gamma=1.0, require_fully_enclosed=True, verbose=False, prefer_central=False):
    """Process existing PNG files in a folder."""
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    results = []
    
    os.makedirs(output_folder, exist_ok=True)
    
    for png_file in png_files:
        print(f"Processing {png_file}")
        
        # Load image
        img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load {png_file}")
            continue
        
        # Find crater center
        center, original_img, contrast_img, overlay = find_crater_center(
            img, debug=True, threshold_factor=threshold_factor, 
            contrast_gamma=contrast_gamma, require_fully_enclosed=require_fully_enclosed,
            verbose=verbose, prefer_central=prefer_central
        )
        
        # Save triptych visualization
        filename = Path(png_file).stem
        vis_path = os.path.join(output_folder, f"{filename}_center_detection.png")
        
        if overlay is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            
            # Original image
            ax1.imshow(original_img, cmap='gray')
            ax1.set_title(f"Original: {filename}")
            ax1.axis('off')
            
            # Contrast-enhanced image
            ax2.imshow(contrast_img, cmap='gray')
            gamma_text = f"γ={contrast_gamma}" if contrast_gamma != 1.0 else "No enhancement"
            ax2.set_title(f"Contrast Enhanced ({gamma_text})")
            ax2.axis('off')
            
            # Annotated image
            ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            thresh_text = f"thresh×{threshold_factor}"
            ax3.set_title(f"Detection ({thresh_text}): {center}")
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        results.append({
            'filename': filename,
            'detected_center_x': center[0],
            'detected_center_y': center[1],
            'image_shape': img.shape
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Crater Center Correction CLI")
    parser.add_argument("--mode", choices=["extract", "process"], required=True,
                       help="Mode: 'extract' to create new tiles, 'process' to analyze existing images")
    parser.add_argument("--raster", type=str, help="Path to DEM raster file (required for extract mode)")
    parser.add_argument("--database", type=str, help="Path to crater database (GeoJSON, required for extract mode)")
    parser.add_argument("--input-folder", type=str, help="Input folder with PNG files (for process mode)")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder for results")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024)")
    parser.add_argument("--num-craters", type=int, default=500, help="Number of craters to process (default: 500)")
    parser.add_argument("--partial-tiles", action="store_true", help="Include partial tiles")
    parser.add_argument("--threshold-factor", type=float, default=0.8, 
                       help="Multiplier for OTSU threshold - lower = more restrictive (default: 0.8)")
    parser.add_argument("--roi-radius", type=float, default=2, 
                       help="Radius of region of interest around crater center measured in crater radii (default: 2)")
    parser.add_argument("--contrast-gamma", type=float, default=1.5,
                       help="Gamma for exponential contrast enhancement - >1 increases contrast (default: 1.5)")
    parser.add_argument("--require-fully-enclosed", action="store_true", default=True,
                       help="Only consider regions fully enclosed by image boundaries (default: True)")
    parser.add_argument("--allow-boundary-regions", action="store_true",
                       help="Allow regions that touch image boundaries (overrides --require-fully-enclosed)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed threshold information during processing")
    parser.add_argument("--prefer-central", action="store_true",
                       help="If multiple regions found, prefer second-largest if it's more central than largest")
    
    args = parser.parse_args()
    
    # Handle boundary region logic
    require_fully_enclosed = args.require_fully_enclosed and not args.allow_boundary_regions
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.mode == "extract":
        if not args.raster or not args.database:
            print("Error: --raster and --database are required for extract mode")
            return 1
        
        print("Loading crater database...")
        df = gpd.read_file(args.database)
        df.insert(1, 'radius_km', df['Diam_m'].apply(lambda x: x / 2000))
        db = CraterDatabase(df, body='Moon', units='km')
        db.add_circles('crater', args.roi_radius)
        
        print(f"Extracting tiles for first {args.num_craters} craters...")
        index = (0, args.num_craters)
        tiles, transform_info = extract_tiles_with_transforms(
            db, args.raster, region="crater", index=index, 
            partial_tiles=args.partial_tiles, shape=(args.tile_size, args.tile_size)
        )
        
        print(f"Processing {len(tiles)} extracted tiles...")
        results = []
        
        for idx, (tile, tinfo) in enumerate(zip(tiles, transform_info)):
            # Normalize and save tile
            p1, p99 = np.percentile(tile, [1, 99])
            tile_normalized = np.clip((tile - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            
            tile_path = os.path.join(args.output_folder, f"tile_{idx:03d}.png")
            Image.fromarray(tile_normalized).save(tile_path)
            
            # Find crater center
            center, original_img, contrast_img, overlay = find_crater_center(
                tile_normalized, debug=True, threshold_factor=args.threshold_factor,
                contrast_gamma=args.contrast_gamma, require_fully_enclosed=require_fully_enclosed,
                verbose=args.verbose, prefer_central=args.prefer_central
            )
            
            # Save triptych visualization
            vis_path = os.path.join(args.output_folder, f"tile_{idx:03d}_center_detection.png")
            if overlay is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                
                # Original tile
                ax1.imshow(original_img, cmap='gray')
                ax1.set_title(f"Tile {idx}: Original")
                ax1.axis('off')
                
                # Contrast-enhanced tile
                ax2.imshow(contrast_img, cmap='gray')
                gamma_text = f"γ={args.contrast_gamma}" if args.contrast_gamma != 1.0 else "No enhancement"
                ax2.set_title(f"Contrast Enhanced ({gamma_text})")
                ax2.axis('off')
                
                # Annotated tile
                ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                thresh_text = f"thresh×{args.threshold_factor}"
                ax3.set_title(f"Detection ({thresh_text}): {center}")
                ax3.axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Calculate corrected coordinates
            corrected_lat, corrected_lon = pixel_to_latlon(center, tinfo)
            
            # Get original data
            crater_idx = tinfo['index']
            original_row = db.data.iloc[crater_idx]
            
            results.append({
                'tile_idx': idx,
                'crater_index': crater_idx,
                'UniqueCraterID': original_row.get('UniqueCraterID', ''),
                'original_lat': tinfo['center_lat'],
                'original_lon': tinfo['center_lon'],
                'detected_center_x': center[0],
                'detected_center_y': center[1],
                'corrected_lat': corrected_lat,
                'corrected_lon': corrected_lon,
                'displacement_pixels_x': center[0] - args.tile_size/2,
                'displacement_pixels_y': center[1] - args.tile_size/2,
                'tile_filename': f"tile_{idx:03d}.png",
                'visualization_filename': f"tile_{idx:03d}_center_detection.png"
            })
            
            print(f"Tile {idx}: Original ({tinfo['center_lat']:.6f}, {tinfo['center_lon']:.6f}) -> "
                  f"Corrected ({corrected_lat:.6f}, {corrected_lon:.6f})")
    
    elif args.mode == "process":
        if not args.input_folder:
            print("Error: --input-folder is required for process mode")
            return 1
        
        results = process_existing_images(
            args.input_folder, args.output_folder, 
            threshold_factor=args.threshold_factor,
            contrast_gamma=args.contrast_gamma,
            require_fully_enclosed=require_fully_enclosed,
            verbose=args.verbose,
            prefer_central=args.prefer_central
        )
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_folder, "crater_center_corrections.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    print(f"Processing complete! Generated {len(results)} result entries.")
    return 0


if __name__ == "__main__":
    os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'
    sys.exit(main())
