"""Unittest masking.py."""
import os.path as p
import unittest
import numpy as np
import craterpy
from craterpy import masking as cm
from craterpy.roi import CraterRoi
from craterpy.dataset import CraterpyDataset

# Test ROI manipulation functions
class Test_circle_mask(unittest.TestCase):
    """Test ring_mask function"""

    def test_radius0(self):
        """Test radius 0"""
        actual = cm.circle_mask((3, 3), 0)
        expected = np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_radius1(self):
        """Test radius 1"""
        actual = cm.circle_mask((3, 3), 1)
        expected = np.array(
            [[False, True, False], [True, True, True], [False, True, False]]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_odd(self):
        """Test roi with odd side length"""
        actual = cm.circle_mask((5, 5), 2)
        expected = np.array(
            [
                [False, False, True, False, False],
                [False, True, True, True, False],
                [True, True, True, True, True],
                [False, True, True, True, False],
                [False, False, True, False, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_even(self):
        """Test roi with even side length"""
        actual = cm.circle_mask((4, 4), 2)
        expected = np.array(
            [
                [False, True, True, False],
                [True, True, True, True],
                [True, True, True, True],
                [False, True, True, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_offcenter(self):
        """Test specifying off center location"""
        actual = cm.circle_mask((5, 5), 2, center=(2, 3))
        expected = np.array(
            [
                [False, False, False, True, False],
                [False, False, True, True, True],
                [False, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, True, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


class Test_ellipse_mask(unittest.TestCase):
    """Test ellipse mask function"""

    def test_radius_equal(self):
        """Test radius equal"""
        actual = cm.ellipse_mask((5, 5), 2, 2)
        expected = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        )
        np.testing.assert_equal(actual, expected)

    def test_y_gt_x(self):
        """Test ysize > xsize"""
        actual = cm.ellipse_mask((5, 5), 2, 1)
        expected = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        )
        np.testing.assert_equal(actual, expected)

    def test_x_gt_y(self):
        """Test xsize > ysize"""
        actual = cm.ellipse_mask((5, 5), 1, 2)
        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        np.testing.assert_equal(actual, expected)

    def test_high_eccentricity(self):
        """Test ysize > xsize"""
        actual = cm.ellipse_mask((3, 9), 1, 4)
        expected = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        print(actual)
        print(expected)
        np.testing.assert_equal(actual, expected)


class Test_ring_mask(unittest.TestCase):
    """Test ring_mask function"""

    def test_trivial(self):
        """Test fully masked."""
        actual = cm.ring_mask((3, 3), 0, 0)
        expected = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_odd(self):
        """Test roi with odd side length"""
        actual = cm.ring_mask((5, 5), 1, 2)
        expected = np.array(
            [
                [False, False, True, False, False],
                [False, True, False, True, False],
                [True, False, False, False, True],
                [False, True, False, True, False],
                [False, False, True, False, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_even(self):
        """Test roi with even side length"""
        actual = cm.ring_mask((4, 4), 1.5, 2)
        expected = np.array(
            [
                [False, True, True, False],
                [True, False, False, True],
                [True, False, False, True],
                [False, True, True, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_offcenter(self):
        """Test specifying off center location"""
        actual = cm.ring_mask((5, 5), 1, 2, center=(2, 3))
        expected = np.array(
            [
                [False, False, False, True, False],
                [False, False, True, False, True],
                [False, True, False, False, False],
                [False, False, True, False, True],
                [False, False, False, True, False],
            ]
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


class Test_crater_floor_mask(unittest.TestCase):
    """Test crater floor mask function"""

    def setUp(self):
        self.data_path = p.join(craterpy.__path__[0], "data")
        self.moon_tif = p.join(self.data_path, "moon.tif")
        self.cds = CraterpyDataset(self.moon_tif, radius=1737)

    def test_floor_mask_equator(self):
        """Test floor mask at equator"""
        lat, lon = (0, 0)
        rad = 2 * self.cds.calc_mpp(lat) / 1000  # 2 pixel radius
        croi = CraterRoi(self.cds, lat, lon, rad, 1)
        actual = cm.crater_floor_mask(croi, 1)
        expected = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
            dtype=bool,
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_floor_mask_mid_lat(self):
        """Test floor mask at mid latitude"""
        lat, lon = (45, 0)
        rad = 2.5 * self.cds.calc_mpp(lat) / 1000  # 2.5 pixel radius
        croi = CraterRoi(self.cds, lat, lon, rad, 1)
        actual = cm.crater_floor_mask(croi, 1)
        expected = np.array(
            [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool
        )
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


# class Test_crater_ring_mask(unittest.TestCase):
#     """Test crater ring mask"""
#     pass  # TODO: implement


# class Test_polygon_mask(unittest.TestCase):
#     """Test polygon mask"""
#     pass  # TODO: implement
