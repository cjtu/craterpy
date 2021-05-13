"""This file contains various helper functions for craterpy"""
import numpy as np


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


def get_crater_cols(df):
    """Return name of latitude, longitude, and radius columns from df"""
    latcol = findcol(df, ["Latitude", "Lat"])
    loncol = findcol(df, ["Longitude", "Lon"])
    radcol = findcol(df, ["Radius", "Rad"])
    if not all((latcol, loncol, radcol)):
        e = "Unable to read latitude, longitude and/or radius from DataFrame"
        raise RuntimeError(e)
    return latcol, loncol, radcol


def diam2radius(df, diamcol=None):
    """Return dataframe with diameter column converted to radius."""
    if not diamcol:
        diamcol = findcol(df, ["diam", "diameter"])
    df.update(df[diamcol] / 2)
    df.rename(columns={diamcol: "Radius"}, inplace=True)
    return df


# def downshift_lon(df):
#    """Shift longitudes from (0, 360) -> (-180, 180) convention."""
#    pass
