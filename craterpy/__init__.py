"""craterpy module"""

from importlib import metadata, resources

from craterpy.classes import CraterDatabase
from craterpy.crs import ALL_BODIES

data_dir = resources.files(__name__).joinpath("data")
sample_data = {p.name: p for p in data_dir.rglob("*") if p.suffix in (".csv", ".tif")}
all_bodies = ALL_BODIES
__all__ = ["CraterDatabase", "sample_data"]

__version__ = metadata.version(__package__)
del metadata
