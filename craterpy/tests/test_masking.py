"""
Suite of unittests for functions found in /craterpy/acefunctions.py.
"""
from __future__ import division, print_function, absolute_import
import unittest
import numpy as np
from craterpy import masking as cm


# Test ROI manipulation functions
class Test_circle_mask(unittest.TestCase):
    """Test ring_mask function"""
    def test_radius0(self):
        """Test radius 0"""
        actual = cm.circle_mask((3, 3), 0)
        expected = np.array([[False, False, False],
                             [False, True,  False],
                             [False, False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_radius1(self):
        """Test radius 1"""
        actual = cm.circle_mask((3, 3), 1)
        expected = np.array([[False, True, False],
                             [True,  True,  True],
                             [False, True, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_odd(self):
        """Test roi with odd side length"""
        actual = cm.circle_mask((5, 5), 2)
        expected = np.array([[False, False, True,  False, False],
                             [False, True,  True,  True,  False],
                             [True,  True,  True,  True,  True],
                             [False, True,  True,  True,  False],
                             [False, False, True,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_even(self):
        """Test roi with even side length"""
        actual = cm.circle_mask((4, 4), 2)
        expected = np.array([[False,  True,  True, False],
                             [True,  True,  True,  True],
                             [True,  True,  True,  True],
                             [False,  True,  True, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_offcenter(self):
        """Test specifying off center location"""
        actual = cm.circle_mask((5, 5), 2, center=(2, 3))
        expected = np.array([[False, False, False, True,  False],
                             [False, False, True,  True,  True],
                             [False, True,  True,  True,  True],
                             [False, False, True,  True,  True],
                             [False, False, False, True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


class Test_ellipse_mask(unittest.TestCase):
    """Test ellipse mask function"""
    pass  # TODO: implement


class Test_ring_mask(unittest.TestCase):
    """Test ring_mask function"""
    def test_trivial(self):
        actual = cm.ring_mask((3, 3), 0, 0)
        expected = np.array([[False, False, False],
                             [False, False, False],
                             [False,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_odd(self):
        """Test roi with odd side length"""
        actual = cm.ring_mask((5, 5), 1, 2)
        expected = np.array([[False, False, True,  False, False],
                             [False, True,  False, True,  False],
                             [True,  False, False, False, True],
                             [False, True,  False, True,  False],
                             [False, False, True,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_even(self):
        """Test roi with even side length"""
        actual = cm.ring_mask((4, 4), 1.5, 2)
        expected = np.array([[False, True,  True,  False],
                             [True,  False, False, True],
                             [True,  False, False, True],
                             [False, True,  True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_offcenter(self):
        """Test specifying off center location"""
        actual = cm.ring_mask((5, 5), 1, 2, center=(2, 3))
        expected = np.array([[False, False, False, True,  False],
                             [False, False, True,  False, True],
                             [False, True,  False, False, False],
                             [False, False, True,  False, True],
                             [False, False, False, True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


class Test_crater_floor_mask(unittest.TestCase):
    """Test crater floor mask function"""
    pass  # TODO: implement


class Test_crater_ring_mask(unittest.TestCase):
    """Test crater ring mask"""
    pass  # TODO: implement


class Test_polygon_mask(unittest.TestCase):
    """Test polygon mask"""
    pass  # TODO: implement
