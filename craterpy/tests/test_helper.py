"""
Suite of unittests for functions found in /craterpy/acefunctions.py.
"""
from __future__ import division, print_function, absolute_import
import unittest
from craterpy import helper as ch


# Image Helper Functions
class Test_km2deg(unittest.TestCase):
    """Test km2deg functions"""
    def test_basic(self):
        """Test simple"""
        actual = ch.km2deg(0.4, 10, 20)
        expected = 2.0
        self.assertEqual(actual, expected)

    def test_float(self):
        """Test float"""
        actual = ch.km2deg(1.5, 4.0, 0.25)
        expected = 1500.0
        self.assertEqual(actual, expected)
