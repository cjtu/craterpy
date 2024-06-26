"""craterpy module"""

from importlib import metadata
from craterpy.roi import CraterRoi
from craterpy.classes import CraterDatabase

__version__ = metadata.version(__package__)
del metadata
