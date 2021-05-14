"""Unittest roi.py"""
import os.path as p
import unittest
import numpy as np
import craterpy
import craterpy.roi as croi
from craterpy.dataset import CraterpyDataset


class TestCraterRoi(unittest.TestCase):
    """Test CraterpyDataset object"""

    def setUp(self):
        self.data_path = p.join(craterpy.__path__[0], "data")
        self.moon_tif = p.join(self.data_path, "moon.tif")
        self.cds = CraterpyDataset(self.moon_tif)
        self.roi = croi.CraterRoi(self.cds, 0, 0, 100)

    def test_roi_import(self):
        """Test import"""
        roi = croi.CraterRoi(self.cds, 0, 0, 100)
        self.assertIsNotNone(roi)

    def test_get_extent(self):
        """Test _get_extent"""
        actual = croi.get_extent(self.cds, 0, 0, 40)
        expected = np.array([-1.3191, 1.3191, -1.3191, 1.3191])
        np.testing.assert_almost_equal(actual, expected, 4)
        actual = croi.get_extent(self.cds, 0, 0, 40, wsize=5)
        expected = 5 * expected
        np.testing.assert_almost_equal(actual, expected, 4)

    def test_get_roi(self):
        """Test get_roi"""
        actual = croi.get_roi_latlon(self.cds, 0, 0.5, 0, 0.5)
        expected = np.array([[41, 43], [41, 41]])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_get_roi_wrap_360(self):
        """Test roi that extends across dataset bounds"""
        # Test wrap right
        actual = croi.get_roi_latlon(self.cds, 179.5, 180.25, 0, 0.5)
        expected = np.concatenate(
            [
                croi.get_roi_latlon(self.cds, 179.5, 180, 0, 0.5),
                croi.get_roi_latlon(self.cds, -180, -179.75, 0, 0.5),
            ],
            axis=1,
        )
        np.testing.assert_equal(actual, expected)

        # Test wrap left is the same
        actual = croi.get_roi_latlon(self.cds, -180.5, -179.75, 0, 0.5)
        np.testing.assert_equal(actual, expected)

    def test_get_roi_oob(self):
        """Test get_roi with extent out of bounds for cds"""
        with self.assertRaises(ValueError):
            _ = croi.get_roi_latlon(self.cds, 0, 0, 91, 0)

    def test_string_repr(self):
        """Test string representation of CraterRoi"""
        actual = str(self.roi)
        expected = "CraterRoi at (0N, 0E) with radius 100 km"
        self.assertEqual(actual, expected)

    def test_filter(self):
        """Test filter method"""
        roi = croi.CraterRoi(self.cds, 0, 0, 10)
        # Filter non-inclusive
        roi.filter(36, 39)
        actual = roi.roi
        expected = np.array(([36, np.nan], [36, 38]))
        np.testing.assert_array_equal(actual, expected)

        # Filter inclusive (strict)
        roi.filter(36, 39, strict=True)
        actual = roi.roi
        expected = np.array(([np.nan, np.nan], [np.nan, 38]))
        np.testing.assert_array_equal(actual, expected)

    def test_mask_nan(self):
        """Test mask"""
        roi = croi.CraterRoi(self.cds, 0, 0, 10)
        mask = np.array([[True, False], [True, True]])
        roi.mask(mask)
        actual = roi.roi
        expected = np.array([[np.nan, 40], [np.nan, np.nan]])
        np.testing.assert_array_equal(actual, expected)

    def test_mask_outside(self):
        """Test mask inverted"""
        roi = croi.CraterRoi(self.cds, 0, 0, 10)
        mask = np.array([[True, False], [True, True]])
        roi.mask(mask, outside=True)
        actual = roi.roi
        expected = np.array([[36, np.nan], [36, 38]])
        np.testing.assert_array_equal(actual, expected)

    def test_mask_fill(self):
        """Test mask with fillvalue"""
        roi = croi.CraterRoi(self.cds, 0, 0, 10)
        mask = np.array([[True, False], [True, True]])
        roi.mask(mask, fillvalue=100)
        actual = roi.roi
        expected = np.array([[100, 40], [100, 100]])
        np.testing.assert_array_equal(actual, expected)
