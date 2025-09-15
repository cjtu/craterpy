from pyproj import CRS
from pyproj.exceptions import CRSError

# Planetary CRS registry
PLANETARY_CRS = {
    "mercury": {
        "planetocentric": CRS("IAU_2015:19900"),
        "planetographic": CRS.from_proj4(
            CRS("IAU_2015:19900").to_proj4() + " +axis=wnu"
        ),
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
    "ceres": {
        "planetocentric": CRS("IAU_2015:200000100"),
        "planetographic": CRS("IAU_2015:200000101"),
    },
    "vesta": {
        # Planetocentric, Claudia crater at 146 E
        "planetocentric": CRS("IAU_2015:200000400"),
        # Claudia double prime (same as IAU_2015)
        "claudia_dp": CRS("IAU_2015:200000400"),
        # Claudia Prime, Claudia crater at 136 E
        "claudia_p": CRS.from_proj4("+proj=longlat +R=255000 +lon_0=-10 +no_defs"),
        # Dawn Claudia, Claudia crater at 356 E
        "dawn_claudia": CRS.from_proj4("+proj=longlat +R=255000 +lon_0=210 +no_defs"),
        # Not Implemented: IAU-2000, Claudia crater at 4.3N, 145 E
    },
    "europa": {
        "planetocentric": CRS("IAU_2015:50200"),
    },
    "ganymede": {
        "planetocentric": CRS("IAU_2015:50300"),
        "planetographic": CRS("IAU_2015:50301"),
    },
    "callisto": {
        "planetocentric": CRS("IAU_2015:50400"),
        "planetographic": CRS("IAU_2015:50401"),
    },
    "enceladus": {
        "planetocentric": CRS("IAU_2015:60200"),
        "planetographic": CRS.from_proj4(
            CRS("IAU_2015:60200").to_proj4() + " +axis=wnu"
        ),
    },
    "tethys": {
        "planetocentric": CRS("IAU_2015:60300"),
        "planetographic": CRS.from_proj4(
            CRS("IAU_2015:60300").to_proj4() + " +axis=wnu"
        ),
    },
    "dione": {
        "planetocentric": CRS("IAU_2015:60400"),
        "planetographic": CRS.from_proj4(
            CRS("IAU_2015:60400").to_proj4() + " +axis=wnu"
        ),
    },
    "rhea": {
        "planetocentric": CRS("IAU_2015:60500"),
        "planetographic": CRS.from_proj4(
            CRS("IAU_2015:60500").to_proj4() + " +axis=wnu"
        ),
    },
    "iapetus": {
        "planetocentric": CRS("IAU_2015:60800"),
        "planetographic": CRS("IAU_2015:60801"),
    },
    "pluto": {
        "planetocentric": CRS("IAU_2015:99900"),
    },
}

# Set the default CRS fro the body, else assume planetocentric
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
            system = (
                DEFAULT_CRS.get(body, "planetocentric")
                if system == "default"
                else str(system)
            )
            return CRS(PLANETARY_CRS[body][system])
        except KeyError as err:
            if body not in PLANETARY_CRS:
                raise ValueError(
                    f"Body '{body}' is not supported. Choose one of {ALL_BODIES} or open a feature request."
                ) from err
            raise ValueError(
                f"Unknown Planetary body and system combo: '{body}', '{system}'."
            ) from err
