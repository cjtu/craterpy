"""This file contains various helper functions for craterpy"""

from functools import partial
import numpy as np
import pandas as pd
import fiona
from fiona.transform import transform_geom
from shapely.geometry import mapping, shape, Polygon


# Geospatial helpers
def lon360(lon):
    """Return longitude in range [0, 360)."""
    return (lon + 360) % 360


def lon180(lon):
    """Return longitude in range (-180, 180]."""
    return ((lon + 180) % 360) - 180


def deg2pix(degrees, ppd):
    """Return degrees converted to pixels at ppd pixels/degree."""
    return int(degrees * ppd)


def get_ind(value, array):
    """Return closest index of a value from array."""
    ind = np.abs(array - value).argmin()
    return int(ind)


def km2deg(dist, mpp, ppd):
    """Return dist converted from kilometers to degrees."""
    return 1000 * dist / (mpp * ppd)


def km2pix(dist, mpp):
    """Return dist converted from kilometers to pixels"""
    return int(1000 * dist / mpp)


def greatcircdist(lat1, lon1, lat2, lon2, radius):
    """Return great circle distance between two points on a spherical body.

    Uses Haversine formula for great circle distances.

    Examples
    --------
    >>> greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
    2887.259950607111
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    # Haversine
    dlat, dlon = abs(lat2 - lat1), abs(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    theta = 2 * np.arcsin(np.sqrt(a))
    return radius * theta


def inglobal(lat, lon):
    """True if lat and lon within global coordinates.

    Default coords: lat in (-90, 90) and lon in (-180, 180).
    mode='pos': lat in (-90, 90) and lon in (0, 360).

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


def get_spheroid_rad_from_wkt(wkt):
    """Return body radius from Well-Known Text coordinate reference system."""
    return float(wkt.lower().split("spheroid")[1].split(",")[1])


# DataFrame helpers
def findcol(df, names):
    """Return first instance of a column from df containing string in names.
    Case insensitive. Raise error if none found.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe object.
    names : str or list of str
        Names to check against columns in df.

    Examples
    --------
    >>> df = pd.DataFrame({'Lat': [10, -20., 80.0],
                           'Lon': [14, -40.1, 317.2],
                           'Diam': [2, 12., 23.7]})
    >>> findcol(df, ['Latitude', 'Lat'])
    'Lat'
    >>> findcol(df, ['Radius'])
    >>> findcol(df, 'diam')
    'Diam'
    """
    if isinstance(names, str):
        names = [names]
    for column in df.columns:
        if any(
            name.lower() in column.lower().replace(" ", "").replace("_", "")
            for name in names
        ):
            return column
    raise ValueError(f"No column containing {names} found.")


def get_crater_cols(df):
    """Return name of latitude, longitude, and radius columns from df"""
    latcol = findcol(df, ["Latitude", "Lat"])
    loncol = findcol(df, ["Longitude", "Lon"])
    try:
        radcol = findcol(df, ["Radius", "Rad", "R(km)", "R(m)"])
    except ValueError as e:
        try:
            diamcol = findcol(df, ["Diameter", "Diam", "D(km)", "D(m)"])
            df["Radius"] = pd.to_numeric(df[diamcol]) / 2
            radcol = "Radius"
        except ValueError:
            raise ValueError("No Radius or Diameter column found.") from e
    return latcol, loncol, radcol


def diam2radius(df, diamcol=None):
    """Return dataframe with diameter column converted to radius."""
    if not diamcol:
        diamcol = findcol(df, ["Diameter", "Diam"])
    df.update(df[diamcol] / 2)
    df.rename(columns={diamcol: "Radius"}, inplace=True)
    return df

def fix_antimeridian(geoms):
    """Fix geometries that cross the antimeridian.
    
    Polygons that also cross a pole might not be fixable but it tries.
    """
    if isinstance(geoms, Polygon):
        minx, miny, maxx, maxy = geoms.bounds
    else:
        minx, miny, maxx, maxy = geoms.bounds.values.T

    # Polygon wrapped around whole globe
    isa = maxx - minx > 330  
    
    # TODO: antimeridian can try to fix pole crossing polys but we need to
    #  know which pole. problem with doing this from the geom is that a crater
    #  at the pole that is 10 degrees diameter will have maxy of 80
    #  maybe need to do fix_antimeridian_north pole and south pole separate?
    isn = maxy + miny > 0  # in the north force_north_pole=True
    # iss = miny < -89.9  # force_south_pole=True
    geoms.loc[isa] = antimeridian.fix_polygon()


def reproject_split_meridian(gdf, dst_crs):
    """Reproject gdf from its geometry (must have crs) to dst_crs, 
    splitting geometries that cross the antimeridian."""
    # See https://gist.github.com/snowman2/2142fc217c983c42a4ed440007438b13
    transformer = partial(
        lambda geom, src_crs, dst_crs: shape(
            transform_geom(src_crs,dst_crs,mapping(geom),
                antimeridian_cutting=True,
                antimeridian_offset=2
            )
        ), src_crs=gdf.crs, dst_crs=dst_crs)
    
    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION=True):
        new_geom = gdf.geometry.apply(transformer)
    new_geom.crs = dst_crs
    return new_geom


def unproject_split_meridian(gdf, src_crs, dst_crs):
    """
    Fix geometries that cross the antimeridian. Slow, use only where needed.

    See https://gist.github.com/snowman2/2142fc217c983c42a4ed440007438b13
    """

    def base_transformer(geom, src_crs, dst_crs):
        return shape(
            transform_geom(
                src_crs=src_crs,
                dst_crs=dst_crs,
                geom=mapping(geom),
                antimeridian_cutting=True,
            )
        )

    reverse_transformer = partial(
        base_transformer, src_crs=dst_crs.to_wkt(), dst_crs=src_crs.to_wkt()
    )
    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION=True):
        if hasattr(gdf, 'geometry'): # GeoDataFrame or GeoSeries
            return gdf.geometry.apply(reverse_transformer)
    # Otherwise assume it is a single instance
    # print(src_crs)
        return reverse_transformer(gdf)
