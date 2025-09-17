"""This file contains various helper functions for craterpy"""

import contextlib
import warnings
from pathlib import Path

import antimeridian
import joblib
import numpy as np
import pandas as pd
import pyproj
from joblib import Parallel, delayed
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform
from tqdm.autonotebook import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=antimeridian.FixWindingWarning)
warnings.filterwarnings("ignore", "Setting nodata.*", module=r".*rasterstats")
warnings.filterwarnings("ignore", "Setting masked.*", module=r".*rasterstats")
warnings.filterwarnings("ignore", ".*lose important proj.*", module=r".*pyproj")


# Geospatial helpers
def bbox2extent(bbox):
    """Convert rasterio/geopandas bounding box to matplotlib extent.

    Parameters
    ----------
    bbox : array-like of float
        Bounding box in the form (minx, miny, maxx, maxy).

    Returns
    -------
    tuple of float
        Extent in the form (xmin, xmax, ymin, ymax).
    """
    return bbox[0], bbox[2], bbox[1], bbox[3]


def lon360(lon):
    """Return longitude in range [0, 360).

    Parameters
    ----------
    lon : float or array-like
        Longitude values in degrees.

    Returns
    -------
    float or array-like
        Corresponding longitude values in the [0, 360] range.
    """
    return (lon + 360) % 360


def lon180(lon):
    """Return longitude in range (-180, 180].

    Parameters
    ----------
    lon : float or array-like
        Longitude values in degrees.

    Returns
    -------
    float or array-like
        Corresponding longitude values in the (-180, 180] range.
    """
    return ((lon + 180) % 360) - 180


def deg2pix(degrees, ppd):
    """Return degrees converted to pixels at ppd pixels/degree.

    Parameters
    ----------
    degrees : float
        Angular distance in degrees.
    ppd : float
        Pixels per degree resolution.

    Returns
    -------
    int
        Equivalent number of pixels.
    """
    return int(degrees * ppd)


def get_ind(value, array):
    """Return closest index of a value from array.

    Parameters
    ----------
    value : float
        Target value to find.
    array : array-like
        Array of values.

    Returns
    -------
    int
        Index of the nearest array element to `value`.
    """
    ind = np.abs(array - value).argmin()
    return int(ind)


def km2deg(dist, mpp, ppd):
    """Return dist converted from kilometers to degrees.

    Parameters
    ----------
    dist : float
        Distance in kilometers.
    mpp : float
        Meters per pixel.
    ppd : float
        Pixels per degree.

    Returns
    -------
    float
        Equivalent angular distance in degrees.
    """
    return 1000 * dist / (mpp * ppd)


def km2pix(dist, mpp):
    """Return dist converted from kilometers to pixels

    Parameters
    ----------
    dist : float
        Distance in kilometers.
    mpp : float
        Meters per pixel.

    Returns
    -------
    int
        Equivalent number of pixels.
    """
    return int(1000 * dist / mpp)


def greatcircdist(lat1, lon1, lat2, lon2, radius):
    """Return great circle distance between two points on a spherical body.

    Uses Haversine formula for great circle distances.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.
    radius : float
        Radius of the sphere (same units for output).

    Returns
    -------
    float
        Great circle distance between the points.

    Examples
    --------
        >>> greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
        2887.259950607111

    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    # Haversine
    dlat, dlon = abs(lat2 - lat1), abs(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    theta = 2 * np.arcsin(np.sqrt(a))
    return radius * theta


def inglobal(lat, lon):
    """True if lat and lon within global coordinates.

    Default coords: lat in (-90, 90) and lon in (-180, 180).
    mode='pos': lat in (-90, 90) and lon in (0, 360).

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.

    Returns
    -------
    bool
        True if latitude is between -90 and 90 and longitude normalized to 0-360.

    Examples
    --------
        >>> inglobal(0, 0)
        True
        >>> inglobal(91, 0)
        False
        >>> inglobal(0, -50)
        True

    """
    return (-90 <= lat <= 90) and (0 <= lon360(lon) <= 360)


# Shape geometry helpers
def gen_annuli(centers: list, rads: list, inner: float, outer: float, crs, **kwargs):
    """
    Generate annular polygons for a set of crater centers using a local azimuthal equidistant projection.

    Each annulus is generated by projecting to a local coordinate system, buffering, and then
    transforming back to geodetic coordinates.

    Processes in parallel with joblib.

    Parameters
    ----------
    centers : list of shapely.geometry.Point
        List of crater center points (latitude/longitude).
    rads : list of float
        List of crater radii (in projection units).
    inner : float
        Inner radius scaling factor (relative to crater radius).
    outer : float
        Outer radius scaling factor (relative to crater radius).
    crs : pyproj.CRS
        Geodetic coordinate reference system for output polygons.
    **kwargs
        Additional keyword arguments passed to annulus creation (e.g., n_jobs, nvert).

    Returns
    -------
    list of shapely.geometry.Polygon
        List of annular polygons in geodetic coordinates.
    """
    # Use joblib to run _create_single_annulus on all available CPU cores minus 1.
    n_jobs = kwargs.get("n_jobs", -2)
    tasks = (
        delayed(create_single_annulus)(center, rad, inner, outer, crs, **kwargs)
        for center, rad in zip(centers, rads, strict=True)
    )
    with tqdm_joblib(tqdm(desc="Generating polygons", total=len(centers))):
        return Parallel(n_jobs=n_jobs, return_as="list")(tasks)


def create_single_annulus(
    center: Point, rad: float, inner: float, outer: float, geodetic_crs, **kwargs
) -> Polygon | MultiPolygon:
    """
    Create a geodetic annulus polygon for a single crater center.

    Projects the center to a local azimuthal equidistant CRS, generates an annular buffer,
    and transforms the result back to the geodetic CRS. Handles antimeridian wrapping if needed.

    Parameters
    ----------
    center : shapely.geometry.Point
        Crater center point (latitude/longitude).
    rad : float
        Crater radius in projection units.
    inner : float
        Inner radius scaling factor (relative to crater radius).
    outer : float
        Outer radius scaling factor (relative to crater radius).
    geodetic_crs : pyproj.CRS
        Geodetic coordinate reference system for output polygon.
    **kwargs
        Additional keyword arguments for buffer creation.

    Returns
    -------
    shapely.geometry.Polygon
        Annulus polygon in geodetic coordinates.
    """
    # Create a local azimuthal equidistant projection for the crater
    local_crs = ProjectedCRS(
        name=f"AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)",
        conversion=AzimuthalEquidistantConversion(center.y, center.x),
        geodetic_crs=geodetic_crs,
    )

    # Generate the buffer in the local, flat projection
    # Assumes self._get_annular_buffer is available or moved here
    buf = get_annular_buffer(Point(0, 0), rad, inner, outer, **kwargs)

    # Create transformer and unproject the buffer to the geodetic CRS
    to_geodetic = pyproj.Transformer.from_crs(
        local_crs, geodetic_crs, always_xy=True
    ).transform
    annulus = transform(to_geodetic, buf)

    # Fix antimeridian wrapping if necessary
    if annulus.bounds[2] - annulus.bounds[0] >= 300:
        annulus = fix_antimeridian_wrap(
            annulus, rad, outer, geodetic_crs, local_crs, **kwargs
        )

    return annulus


def get_annular_buffer(point: Point, rad: float, inner: float, outer: float, **kwargs):
    """
    Generate an annular buffer (ring-shaped polygon) around a point.

    The annulus is defined by an outer and inner radius, both scaled by the crater radius.
    The polygon is created in a projected (flat) coordinate system.

    Parameters
    ----------
    point : shapely.geometry.Point
        Center point for the buffer (usually (0, 0) in local projection).
    rad : float
        Crater radius in projection units.
    inner : float
        Inner radius scaling factor (relative to crater radius).
    outer : float
        Outer radius scaling factor (relative to crater radius).
    **kwargs
        nvert : int, optional
            Number of vertices for the circular polygon (default: 32).

    Returns
    -------
    shapely.geometry.Polygon
        Annular buffer polygon.
    """
    # Num vertices to make each circular poly
    nvert = kwargs.get("nvert", 32)
    outer_scaled = rad * outer
    inner_scaled = rad * inner
    outer_poly = point.buffer(outer_scaled, quad_segs=nvert // 4)
    if inner_scaled <= 0:
        return outer_poly
    inner_poly = point.buffer(inner_scaled, quad_segs=nvert // 4)
    return outer_poly.difference(inner_poly)


def fix_antimeridian_wrap(
    annulus, rad: float, outer: float, geodetic_crs, local_crs, **kwargs
) -> Polygon | MultiPolygon:
    """
    Corrects polygons that wrap incorrectly across the antimeridian.

    Checks if the generated annulus covers the North or South pole and applies
    antimeridian fixing logic to ensure valid geometry.

    Parameters
    ----------
    annulus : shapely.geometry.Polygon
        Annulus polygon in geodetic coordinates.
    rad : float
        Crater radius in projection units.
    outer : float
        Outer radius scaling factor (relative to crater radius).
    geodetic_crs : pyproj.CRS
        Geodetic coordinate reference system.
    local_crs : pyproj.CRS
        Local projected coordinate reference system.
    **kwargs
        Additional keyword arguments for buffer creation (e.g. nvert).

    Returns
    -------
    shapely.geometry.Polygon
        Corrected polygon with proper antimeridian handling.
    """
    to_projected = pyproj.Transformer.from_crs(
        geodetic_crs, local_crs, always_xy=True
    ).transform

    # Create a solid buffer (from radius 0 to outer) for pole checking
    solid_buffer = get_annular_buffer(Point(0, 0), rad, 0, outer, **kwargs)

    # Check if North or South pole is contained within the buffer
    covers_north_pole = solid_buffer.contains(transform(to_projected, Point(0, 90)))
    covers_south_pole = solid_buffer.contains(transform(to_projected, Point(0, -90)))

    return antimeridian.fix_polygon(
        annulus,
        force_north_pole=covers_north_pole,
        force_south_pole=covers_south_pole,
    )


def compute_zonal_stats(geometries, raster, suffix: str = "", **kwargs) -> pd.DataFrame:
    """
    Compute zonal stats (e.g., mean, std, count) for each geometry in the raster.

    See rasterstats.zonal_stats.

    Parameters
    ----------
    geometries : geopandas.GeoSeries
        GeoSeries of geometry objects (e.g., shapely.Polygons).
    raster : str or rasterio.DatasetReader
        Raster file path or open raster dataset.
    suffix : str, optional
        Suffix to append to output column names (default: "").
    **kwargs
        Additional keyword arguments to pass to rasterstats.zonal_stats (e.g., stats, nodata).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing computed statistics for each geometry.
    """
    # Perform the core calculation
    zstats = zonal_stats(geometries, raster, all_touched=True, **kwargs)
    df = pd.DataFrame(zstats, index=geometries.index)

    # Return only requested stat columns with suffix, if given
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return df.add_suffix(suffix)


def get_stats(
    gdf,
    rasters,
    regions,
    stats=("mean", "std", "count"),
    nodata=None,
    n_jobs=-2,
):
    """
    Compute zonal stats on polygons in a GeoDataFrame for all rasters and regions in parallel.

    Return a single DataFrame with columns <stat_raster_region> for all combinations.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon regions as columns.
    rasters : str, rasterio.DatasetReader, or dict of {str: str or rasterio.DatasetReader}
        Single raster path or open raster dataset, or a mapping of names to rasters.
    regions : str or list of str
        Name or list of names of geometry columns in gdf to use as regions.
    stats : tuple of str, optional
        Statistics to compute for each raster-region combination. Default: ("mean", "std", "count").
    nodata : number, None, or dict of {str: number}, optional
        Nodata value to use for masking, or mapping of raster names to their nodata values.
        If None, no nodata masking is applied.
    n_jobs : int, optional
        Number of parallel worker processes to use. Default is 1.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original crater columns along with appended
        statistics columns for each raster and region combination.
    """

    def _name_raster(i, raster):
        return Path(raster).name if isinstance(raster, str) else f"raster{i}"

    if isinstance(rasters, dict):
        rdict = rasters
    elif isinstance(rasters, (tuple, list)):
        # Set up dict for column naming. Choose filename if str, else "raster{i}" (e.g., if given file descriptors)
        rdict = {
            _name_raster(i, raster): Path(raster).as_posix()
            for i, raster in enumerate(rasters)
        }
    else:
        rdict = {"_": rasters}
    if isinstance(regions, str):
        regions = [regions]
    if not isinstance(nodata, dict):
        nodata = dict.fromkeys(rdict, nodata)

    total_tasks = len(rdict) * len(regions)
    # Create a list of tasks, where each task has all the info for one worker call
    tasks = []
    for region_name in regions:
        for raster_name, raster_file in rdict.items():
            tasks.append(
                delayed(compute_zonal_stats)(
                    geometries=gdf[region_name],
                    raster=raster_file,
                    stats=stats,
                    nodata=nodata.get(raster_name),  # Safely get nodata value
                    suffix=f"{raster_name}_{region_name}".strip("_"),
                )
            )
    # Compute in parallel
    with tqdm_joblib(tqdm(desc="Computing Zonal Stats", total=total_tasks)):
        all_stats = Parallel(n_jobs=n_jobs)(tasks)
        return pd.concat(all_stats, axis=1)


# Misc
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    See: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# DataFrame helpers
def findcol(df, names, exact=False):
    """Return first matching column in df matching a string given in names.

    Case insensitive, ignores whitespace. Raises ValueError if none found.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe object.
    names : str or list of str
        Names to check against columns in df.
    exact : bool
        Exact matches only (case-insensitive). Otherwise match on column
        substring (default: False).

    Returns
    -------
    str
        Name of the first matching column.

    Raises
    ------
    ValueError
        If no matching column is found.

    Examples
    --------
        >>> df = pd.DataFrame({'Lat': [10, -20., 80.0],
                            'Lon': [14, -40.1, 317.2],
                            'Diam': [2, 12., 23.7]})
        >>> findcol(df, ['Latitude', 'Lat'])
        'Lat'
        >>> findcol(df, ['Radius'])
        Traceback (most recent call last):
        ...
        ValueError: No column containing ['Radius'] found.
        >>> findcol(df, 'diam')
        'Diam'

    """
    if isinstance(names, str):
        names = [names]
    # Ignore spaces, convert "(xyz)" to "_xyz" so that Ex. d_m will match d (m)
    # Note: can match with regex but then calling func needs to escape chars like ()
    cols = df.columns.str.replace(" ", "")
    cols = cols.str.replace("(", "_", regex=False).str.replace(")", " ", regex=False)
    for name in names:
        if exact:
            matches = df.columns[cols.str.fullmatch(name, case=False)]
        else:
            matches = df.columns[cols.str.contains(name, case=False)]
        if any(matches):
            # Exit early at first match found
            return matches[0]
    raise ValueError(f"No column containing {names} found.")


def find_rad_or_diam_col(df):
    """Return the name of the radius or diameter column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to search for radius/diameter column names.

    Returns
    -------
    str
        Name of the column representing radius or diameter.

    Raises
    ------
    ValueError
        If no suitable column is found.
    """
    # First search these names (case-insensitive) in order for exact matches
    names = "radius,diameter,rad,diam,r_km,d_km,r_m,d_m,r,d"
    try:
        col = findcol(df, names.split(","), exact=True)
    except ValueError:
        try:
            # Unlikely that inexact matches for r,d are correct so exclude
            names_contains = names.replace(",r,d", "")
            col = findcol(df, names_contains.split(","), exact=False)
        except ValueError:
            raise ValueError("Could not identify radius or diameter column.") from None
    return col


def latlon_to_cartesian(lat, lon, radius):
    """Convert lat, lon to cartesian (x, y, z)

    Parameters
    ----------
    lat : float or array-like
        Latitude in degrees.
    lon : float or array-like
        Longitude in degrees.
    radius : float
        Radius of the sphere.

    Returns
    -------
    tuple of array-like or float
        Cartesian coordinates (x, y, z).

    Examples
    --------
        >>> latlon_to_cartesian(0, 0, 1)
        (1.0, 0.0, 0.0)

    """
    lat, lon = np.radians(lat), np.radians(lon)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def merge(
    df1,
    df2,
    rbody=1737.4e3,
    rtol=0.5,
    latcol="_lat",
    loncol="_lon",
    radcol="_radius_m",
):
    """Spatial merge of craters in df1 and df2.

    If a crater matches, keep lat, lon, rad and other col values from df1.
    If not a match, append to the end with lat, lon, rad from df2.
    Retains all columns from both dfs and fills in values if present.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Left dataframe whose values will be preserved for any matches
    df2 : pandas.DataFrame
        Right dataframe to merge
    rbody : float
        Radius of the body in meters (default: Moon's radius)
    latcol : str
        Name of latitude column (default: 'lat')
    loncol : str
        Name of longitude column (default: 'lon')
    radcol : str
        Name of radius column (default: 'rad')
    rtol : float
        Relative tolerance for radius matching (default: 0.5)

    Returns
    -------
    pandas.DataFrame
        Merged dataframe with lat/lon/rad from df1
    """

    def match(result, df2_rad):
        """Return matching index if radii differ by less than rtol."""
        if result:
            for match in result:
                df1_rad = df1[radcol].iloc[match]
                if (df2_rad - df1_rad) / df1_rad < rtol:
                    return match
        return np.nan

    coords1 = list(
        zip(*latlon_to_cartesian(df1[latcol], df1[loncol], rbody), strict=True)
    )
    coords2 = list(
        zip(*latlon_to_cartesian(df2[latcol], df2[loncol], rbody), strict=True)
    )
    tree = cKDTree(coords1)
    results = tree.query_ball_point(coords2, r=df2[radcol])
    df2["df1_idx"] = np.nan
    for (i, row), result in zip(df2.iterrows(), results, strict=True):
        df2.at[i, "df1_idx"] = match(result, row[radcol])

    # Append non-matching rows (includes columns from both df1 and df2)
    merged = pd.concat([df1, df2[df2.df1_idx.isna()]], ignore_index=True)

    # Find and add missing data from columns unique to df2 to the matched rows
    df2_uniq_cols = list(set(df2.columns) - set(df1.columns))
    df2_match_idx = df2.df1_idx.dropna().values
    merged.loc[df2_match_idx, df2_uniq_cols] = df2.loc[
        ~df2.df1_idx.isna(), df2_uniq_cols
    ].values
    merged.drop(columns="df1_idx", inplace=True)
    return merged.reset_index(drop=True)
