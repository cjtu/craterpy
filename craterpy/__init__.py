"""craterpy module"""

from importlib import metadata, resources

from craterpy.classes import CraterDatabase
from craterpy.crs import ALL_BODIES

sample_data = {k.name: k for k in resources.files(__name__).joinpath("data").iterdir()}
all_bodies = ALL_BODIES
__all__ = ["CraterDatabase", "sample_data"]

__version__ = metadata.version(__package__)
del metadata
