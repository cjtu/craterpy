"""craterpy module"""

from importlib import metadata, resources
from craterpy.classes import CraterDatabase, CRS_DICT

sample_data = {
    k: str(resources.files(__name__).joinpath("data", k))
    for k in (
        "moon.tif",
        "moon_craters.csv",
        "vesta.tif",
        "vesta_craters.csv",
        "moon_dem.tif",
    )
}
all_bodies = list(CRS_DICT.keys())

__version__ = metadata.version(__package__)
del metadata
