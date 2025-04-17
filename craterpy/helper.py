"""This file contains various helper functions for craterpy"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Geospatial helpers
def bbox2extent(bbox):
    """Convert rasterio/geopandas bounding box to matplotlib extent."""
    return bbox[0], bbox[2], bbox[1], bbox[3]


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
            name.lower() in column.lower().replace(" ", "") for name in names
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


def latlon_to_cartesian(lat, lon, radius):
    """Convert lat, lon to cartesian (x, y, z)"""
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
