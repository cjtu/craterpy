"""craterpy module"""
from importlib import metadata
from craterpy.dataset import CraterpyDataset
from craterpy.roi import CraterRoi

__version__ = metadata.version(__package__)
del metadata
