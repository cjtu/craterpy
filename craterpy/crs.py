from pyproj import CRS
from pyproj.exceptions import CRSError

# CRS for units / coord transformations (convention is only Ocentric)
BODIES = {
    "mercury": 199,
    "venus": 299,
    "moon": 301,
    "earth": 399,
    "phobos": 401,
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
    "titan": 606,
    "iapetus": 608,
    "triton": 801,
    "charon": 901,
    "pluto": 999,
}


PLANETARY_CRS = {
    "mercury": {
        "planetocentric": CRS("IAU_2015:19900"),
        "planetographic": CRS("IAU_2015:19901"),
    },
    "venus": {
        "planetocentric": CRS("IAU_2015:29900"),
    },
    "moon": {
        "planetocentric": CRS("IAU_2015:30100"),
    },
    "earth": {
        "planetocentric": CRS("IAU_2015:39900"),
        "planetographic": CRS("IAU_2015:39901"),
    },
    "mars": {
        "planetocentric": CRS("IAU_2015:49900"),
        "planetographic": CRS("IAU_2015:49901"),
    },
    "vesta": {
        # Planetographic, Positive West, Claudia Double Prime at 146 E
        "claudia_dp": CRS.from_proj4("+proj=longlat +R=255000 +lon_0=150 +no_defs"),
        # Planetographic, Positive West, Claudia Prime at 136 E
        "claudia_p": CRS.from_proj4("+proj=longlat +R=255000 +lon_0=160 +no_defs"),
        # Planetographic, Positive West, Dawn-Claudia at 356 E
        "dawn_claudia": CRS.from_proj4("+proj=longlat +R=255000 +lon_0=356 +no_defs"),
        # Internal Standard (Planetocentric)
        "planetocentric": CRS("IAU_2015:200000400"),
    },
}

DEFAULT_CRS = {
    "mercury": "planetographic",
    "venus": "planetocentric",
    "moon": "planetocentric",
    "earth": "planetographic",
    "mars": "planetographic",
    "vesta": "claudia_dp",
}

ALL_BODIES = list(PLANETARY_CRS.keys())


def get_crs(body: str, system: str | CRS = "default") -> CRS:
    """Retrieves a CRS object from the registry or parse system if valid pyproj CRS."""
    # Note CRSes need always_xy to interpret the lat,lon as y,x in Proj geometries
    body = body.lower()
    system = system.lower() if isinstance(system, str) else system
    try:
        # Should fail for any keys in PLANETARY_CRS
        return CRS.from_user_input(system)
    except CRSError:
        try:
            system = DEFAULT_CRS[body] if system == "default" else str(system)
            return CRS(PLANETARY_CRS[body][system])
        except KeyError as err:
            raise ValueError(f"Unknown body '{body}' or system '{system}'.") from err
