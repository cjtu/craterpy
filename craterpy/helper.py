"""This file contains various helper functions for craterpy"""
from __future__ import division, print_function, absolute_import
from craterpy.exceptions import LatLongOutOfBoundsError
import numpy as np


# Geospatial helpers
def deg2pix(degrees, ppd):
    """Return degrees converted to pixels at ppd pixels/degree."""
    return int(degrees*ppd)


def get_ind(value, array):
    """Return closest index of a value from array."""
    ind = np.abs(array-value).argmin()
    return int(ind)


def km2deg(dist, mpp, ppd):
    """Return dist converted from kilometers to degrees."""
    return 1000*dist/(mpp*ppd)


def km2pix(dist, mpp):
    """Return dist converted from kilometers to pixels"""
    return int(1000*dist/mpp)


def greatcircdist(lat1, lon1, lat2, lon2, radius):
    """Return great circle distance between two points on a spherical body.

    Uses Haversine formula for great circle distances.

    Examples
    --------
    >>> greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
    2887.259950607111
    """
    if not all(map(inglobal, (lat1, lon1), (lat2, lon2))):
        raise LatLongOutOfBoundsError("Latitude or longitude out of bounds.")
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    # Haversine
    dlat, dlon = abs(lat2 - lat1), abs(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    theta = 2 * np.arcsin(np.sqrt(a))
    return radius*theta


def inglobal(lat, lon, mode=None):
    """True if lat and lon within global coordinates.

    Default coords: lat in (-90, 90) and lon in (-180, 180).
    mode='pos': lat in (-90, 90) and lon in (0, 360).

    Examples
    --------
    >>> lat = -10
    >>> lon = -10
    >>> inbounds(lat, lon)
    True
    >>> inbounds(lat, lon, 'pos')
    False
    """
    if mode == 'pos':
        return (-90 <= lat <= 90) and (0 <= lon <= 360)
    else:
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)


# DataFrame helpers
def findcol(df, names):
    """Return first instance of a column from df matching an unformated string
    in names. If none found, return None

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
    cols = df.columns.values
    for name in names:
        for col in cols:
            if name.strip().lower() == col.strip().lower():
                return col
    return None


def diam2radius(df, diamcol=None):
    """Return dataframe with diameter column converted to radius."""
    if not diamcol:
        diamcol = findcol(df, ['diam', 'diameter'])
    df.update(df[diamcol]/2)
    df.rename(columns={diamcol: "Radius"}, inplace=True)
    return df


# def downshift_lon(df):
#    """Shift longitudes from (0, 360) -> (-180, 180) convention."""
#    pass
