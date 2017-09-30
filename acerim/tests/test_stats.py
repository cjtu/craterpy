"""
Suite of unittests for classes found in /acerim/acestats.py.
"""
from __future__ import division, print_function, absolute_import
import unittest
import numpy as np
from acerim import acestats as acs


class Test_ace_stats(unittest.TestCase):
    """Test the basic stats found in acestats.py"""
    data_arr = np.array([4, 4, 8, 3, 4, -1, 0, -5, 1, -10, 3])

    def test_size(self):
        """Test acestats.size function"""
        expected = 11
        actual = acs.size(self.data_arr)
        self.assertEqual(actual, expected)

    def test_mean(self):
        """Test acestats.mean function"""
        expected = 1.0
        actual = acs.mean(self.data_arr)
        self.assertEqual(actual, expected)

    def test_median(self):
        """Test acestats.median function"""
        expected = 3.0
        actual = acs.median(self.data_arr)
        self.assertEqual(actual, expected)

    def test_mode(self):
        """Test acestats.mode function"""
        expected = 4.0
        actual = acs.mode(self.data_arr)
        self.assertEqual(actual, expected)

    def test_std(self):
        """Test acestats.std function"""
        expected = 4.9598387
        actual = acs.std(self.data_arr)
        self.assertAlmostEqual(actual, expected)

    def test_maximum(self):
        """Test acestats.maximum function"""
        expected = 8.0
        actual = acs.maximum(self.data_arr)
        self.assertEqual(actual, expected)

    def test_minimum(self):
        """Test acestats.minimum function"""
        expected = -10.0
        actual = acs.minimum(self.data_arr)
        self.assertEqual(actual, expected)

    def test_q1(self):
        """Test acestats.q1 function"""
        expected = -0.5
        actual = acs.q1(self.data_arr)
        self.assertEqual(actual, expected)

    def test_q3(self):
        """Test acestats.q3 function"""
        expected = 4
        actual = acs.q3(self.data_arr)
        self.assertEqual(actual, expected)

    def test_pct95(self):
        """Test acestats.pct95 function"""
        expected = 6
        actual = acs.pct95(self.data_arr)
        self.assertEqual(actual, expected)
