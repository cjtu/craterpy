"""This file contains various helper functions for craterpy"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


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
        Corresponding longitude values in the [0, 360) range.
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
    .. code-block:: doctest

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
    .. code-block:: doctest

        >>> inglobal(0, 0)
        True
        >>> inglobal(91, 0)
        False
        >>> inglobal(0, -50)
        True

    """
    return (-90 <= lat <= 90) and (0 <= lon360(lon) <= 360)


def get_spheroid_rad_from_wkt(wkt):
    """Return body radius from Well-Known Text coordinate reference system.

    Parameters
    ----------
    wkt : str
        WKT string of the coordinate reference system.

    Returns
    -------
    float
        Semi-major axis of the spheroid in the WKT.
    """
    return float(wkt.lower().split("spheroid")[1].split(",")[1])


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
    .. code-block:: doctest

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
    cols = cols.str.replace("(", "_", regex=False).str.replace(
        ")", " ", regex=False
    )
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
            raise ValueError("Could not identify radius or diameter column.")
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
    .. code-block:: doctest

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
    latcol="lat",
    loncol="lon",
    radcol="rad",
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

    coords1 = list(zip(*latlon_to_cartesian(df1[latcol], df1[loncol], rbody)))
    coords2 = list(zip(*latlon_to_cartesian(df2[latcol], df2[loncol], rbody)))
    tree = cKDTree(coords1)
    results = tree.query_ball_point(coords2, r=df2[radcol])
    df2["df1_idx"] = np.nan
    for (i, row), result in zip(df2.iterrows(), results):
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
