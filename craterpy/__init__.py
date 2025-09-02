"""craterpy module"""

from importlib import metadata, resources

from craterpy.classes import BODIES, CraterDatabase

sample_data = {k.name: k for k in resources.files(__name__).joinpath("data").iterdir()}
all_bodies = list(BODIES.keys())
__all__ = ["CraterDatabase", "sample_data"]

__version__ = metadata.version(__package__)
del metadata
