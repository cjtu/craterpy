"""craterpy module"""

from importlib import metadata, resources
from craterpy.classes import CraterDatabase

sample_data = {
    k: resources.files(__package__).joinpath(f"data/{k}")
    for k in ("moon.tif", "moon_craters.csv", "vesta.tif", "vesta_craters.csv")
}

__version__ = metadata.version(__package__)
del metadata
