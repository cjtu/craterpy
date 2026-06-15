"""Planetary coordinate reference systems for craterpy.

Standard planetocentric/planetographic CRS come from :func:`planetarypy.crs.body_crs`.
This module adds the cases planetarypy does not produce: Vesta Claudia frames and
the west-positive planetographic for bodies with no IAU ographic code (see
``_SPECIAL_CRS``).

``body_crs`` is always called with an integer NAIF id (``_BODY_NAIF``), never a
name, to avoid planetarypy's name-resolution path importing
``planetarypy.constants`` (which can trigger a one-time archive download).
"""

from planetarypy.crs import body_crs
from pyproj import CRS
from pyproj.exceptions import CRSError

# Body name -> NAIF id. The IAU CRS code is naif_id * 100 + variant offset
# (offset 0 = planetocentric, 1 = planetographic where IAU defines it).
_BODY_NAIF = {
    "mercury": 199,
    "venus": 299,
    "moon": 301,
    "earth": 399,
    "mars": 499,
    "ceres": 2000001,
    "vesta": 2000004,
    "europa": 502,
    "ganymede": 503,
    "callisto": 504,
    "enceladus": 602,
    "tethys": 603,
    "dione": 604,
    "rhea": 605,
    "iapetus": 608,
    "pluto": 999,
}

# Default CRS for each body, else assume planetocentric.
DEFAULT_CRS = {
    "ceres": "planetographic",
    "ganymede": "planetographic",
    "enceladus": "planetographic",
    "tethys": "planetographic",
    "dione": "planetographic",
    "rhea": "planetographic",
    "iapetus": "planetographic",
    "mercury": "planetographic",
}

ALL_BODIES = list(_BODY_NAIF)

# craterpy alias -> planetarypy `body_crs` system name.
_SYSTEM_MAP = {"planetocentric": "ocentric", "planetographic": "ographic"}

# Bodies with no IAU planetographic code: craterpy fabricates one by flipping
# the planetocentric CRS to a west-positive (wnu) axis order.
_WNU_PLANETOGRAPHIC = {"mercury", "enceladus", "tethys", "dione", "rhea"}


def _special_crs(body: str, system: str, naif: int) -> CRS | None:
    """Return a craterpy-specific CRS not produced by planetarypy, else None."""
    if body == "vesta":
        if system == "claudia_dp":
            # Claudia double prime == the IAU_2015 standard for Vesta.
            return body_crs(naif, "ocentric")
        if system == "claudia_p":
            # Claudia Prime, Claudia crater at 136 E (10 deg W offset).
            return CRS.from_proj4("+proj=longlat +R=255000 +lon_0=+10 +no_defs")
        if system == "dawn_claudia":
            # Dawn Claudia, Claudia crater at 356 E (210 deg E offset).
            return CRS.from_proj4("+proj=longlat +R=255000 +lon_0=-210 +no_defs")
    if system == "planetographic" and body in _WNU_PLANETOGRAPHIC:
        return CRS.from_proj4(body_crs(naif, "ocentric").to_proj4() + " +axis=wnu")
    return None


def get_crs(body: str, system: str | CRS = "default") -> CRS:
    """Retrieve a CRS for a body, or pass through any valid pyproj CRS.

    Parameters
    ----------
    body : str
        Planetary body name (case-insensitive), e.g. 'Moon', 'mars', 'Vesta'.
    system : str or pyproj.CRS
        A system alias ('planetocentric', 'planetographic', 'default', or a
        body-specific alias like 'claudia_dp'), or any valid pyproj CRS object
        or string (e.g. an EPSG code), which is returned as-is.
    """
    # Pass through any input that is already a valid CRS (object or string).
    try:
        return CRS.from_user_input(system)
    except CRSError:
        pass

    body = body.lower()
    system = system.lower()
    if body not in _BODY_NAIF:
        raise ValueError(
            f"Body '{body}' is not supported. Choose one of {ALL_BODIES} "
            "or open a feature request."
        )
    if system == "default":
        system = DEFAULT_CRS.get(body, "planetocentric")

    naif = _BODY_NAIF[body]
    special = _special_crs(body, system, naif)
    if special is not None:
        return special

    if system not in _SYSTEM_MAP:
        raise ValueError(
            f"Unknown Planetary body and system combo: '{body}', '{system}'."
        )
    try:
        # Delegate the standard ocentric/ographic CRS to planetarypy. Raises
        # for bodies that define no planetographic IAU code (e.g. pluto).
        return body_crs(naif, _SYSTEM_MAP[system])
    except ValueError as err:
        raise ValueError(
            f"Unknown Planetary body and system combo: '{body}', '{system}'."
        ) from err
