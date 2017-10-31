from __future__ import division, print_function, absolute_import
import os.path as p
import unittest
import numpy as np
from craterpy.dataset import CraterpyDataset
from craterpy.roi import CraterRoi
from craterpy.masking import circle_mask

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

    def test_get_extent(self):
        """Test _get_extent"""
        roi = CraterRoi(self.cds, 0, 0, 16)
        actual = roi._get_extent()
        expected = (-0.5278, 0.5278, -0.5278, 0.5278)
        np.testing.assert_almost_equal(actual, expected, 4)
        roi = CraterRoi(self.cds, 20, 20, 16)
        actual = roi._get_extent()
        expected = (19.4384, 20.5616, 19.4722, 20.5278)
        np.testing.assert_almost_equal(actual, expected, 4)

    def test_get_roi(self):
        roi = CraterRoi(self.cds, 0, 0, 50)
        arr = roi.get_roi()
        self.assertIsNotNone(arr)
        roi180 = CraterRoi(self.cds, 0, 180, 50)
        arr180 = roi.get_roi()
        self.assertIsNotNone(arr180)

    def test_filter(self):
        roi = self.roi
        roi.filter(0, 100)
        self.assertTrue(roi.roi.max() <= 100)
        roi.filter(0, 50, strict=True)
        self.assertTrue(roi.roi.max() < 50)

    # TODO: fix this test
    # def test_mask(self):
    #     shape, rad = self.roi.roi.shape, self.roi.rad
    #     mask = circle_mask(shape, rad)
    #     masked_roi = self.roi.mask(mask)
    #     self.assertTrue(np.isnan(self.roi.roi[*(shape/2)]))

    def test_plot(self):
        self.assertIsNotNone(self.roi.plot())

