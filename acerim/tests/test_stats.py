"""
Suite of unittests for classes found in /acerim/acestats.py.
"""
from __future__ import division, print_function, absolute_import
import unittest
import numpy as np
from acerim import acestats as acs


class Test_protected(unittest.TestCase):
    """Test the protected functions in acestats.py"""

    def test_listStats(self):
        """Test the _listStats function"""
        expected = np.array(["maximum", "mean", "median"])
        actual = acs._listStats()[:3]
        np.testing.assert_array_equal(actual, expected)

    def test_getFunctions_single(self):
        """Test the _getFunctions function with one stat specified"""
        expected_name = "maximum"
        stat_func = acs._getFunctions("maximum")
        actual_name = stat_func[0][0]
        actual_func = stat_func[0][1]
        self.assertEqual(actual_name, expected_name)
        self.assertEqual(callable(actual_func), True)

    def test_getFunctions_multi(self):
        """Test the _getFunctions function with multiple stats specified"""
        expected_names = ["maximum", "mean", "median"]
        stats_funcs = acs._getFunctions(("maximum", "mean", "median"))
        actual_names = [pair[0] for pair in stats_funcs]
        funcs_callable = [callable(pair[1]) for pair in stats_funcs]
        self.assertEqual(actual_names, expected_names)
        self.assertEqual(funcs_callable, [True, True, True])

    def test_getFuctions_undefined_stat(self):
        """Test that _getFunctions raises Exception if passed undefined stat"""
        self.assertRaises(ValueError, acs._getFunctions, "Unknown_Stat")


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
