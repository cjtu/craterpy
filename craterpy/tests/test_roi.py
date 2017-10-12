from __future__ import division, print_function, absolute_import
import os.path as p
import unittest
from craterpy.dataset import CraterpyDataset
from craterpy.roi import CraterRoi


class TestCraterRoi(unittest.TestCase):
    """Test CraterpyDataset object"""
    def setUp(self):
        import craterpy
        self.data_path = p.join(craterpy.__path__[0], 'data')
        self.moon_tif = p.join(self.data_path, 'moon.tif')
        self.cds = CraterpyDataset(self.moon_tif, radius=1737)
        self.roi = CraterRoi(self.cds, 0, 0, 100)

    def test_roi_import(self):
        """Test import"""
        roi = CraterRoi(self.cds, 0, 0, 100)
        self.assertIsNotNone(roi)
