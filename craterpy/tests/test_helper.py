"""
Suite of unittests for functions found in /craterpy/acefunctions.py.
"""
from __future__ import division, print_function, absolute_import
import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from craterpy import helper as ch


class Test_geo_helpers(unittest.TestCase):
    """Test Geospatial Helper Functions"""
    def test_deg2pix(self):
        """Test deg2pix function"""
        actual = ch.deg2pix(10, 20)
        expected = 200
        self.assertEqual(actual, expected)
        actual = ch.deg2pix(5.5, 2.5)
        expected = 13
        self.assertEqual(actual, expected)

    def test_get_ind(self):
        """Test get_ind function"""
        actual = ch.get_ind(2, np.arange(5))
        expected = 2
        self.assertEqual(actual, expected)
        actual = ch.get_ind(3.7, np.arange(5))
        expected = 4
        self.assertEqual(actual, expected)

    def test_km2deg(self):
        """Test km2deg functions"""
        actual = ch.km2deg(0.4, 10, 20)
        expected = 2.0
        self.assertEqual(actual, expected)
        actual = ch.km2deg(128, 4000, 32)
        expected = 1.0
        self.assertEqual(actual, expected)

    def test_km2pix(self):
        """Test km2pix function"""
        actual = ch.km2pix(1, 100)
        expected = 10
        self.assertEqual(actual, expected)
        actual = ch.km2pix(1.5, 1500)
        expected = 1
        self.assertEqual(actual, expected)
        actual = ch.km2pix(0.5, 2.5)
        expected = 200
        self.assertEqual(actual, expected)

    def test_greatcircdist(self):
        """Test greatcircdist"""
        actual = ch.greatcircdist(36.12, -86.67, 33.94, -118.40, 6372.8)
        expected = 2887.259950607111
        self.assertAlmostEqual(actual, expected)

    def test_inglobal(self):
        """Test inglobal"""
        self.assertTrue(ch.inglobal(0, 0))
        self.assertTrue(ch.inglobal(90.0, 180.0))
        self.assertTrue(ch.inglobal(-5.5, 360, 'pos'))
        self.assertFalse(ch.inglobal(91, 0))
        self.assertFalse(ch.inglobal(0, 181))
        self.assertFalse(ch.inglobal(0, -1, 'pos'))


class Test_dataframe_helpers(unittest.TestCase):
    """Test pandas.DataFrame helper functions"""
    def setUp(self):
        self.df = pd.DataFrame({'Lat': [10, -20., 80.0],
                                'Lon': [14, -40.1, 317.2],
                                'Diam': [2, 12., 23.7]})

    def test_findcol(self):
        """Test findcol"""
        expected = 'Lat'
        actual = ch.findcol(self.df, 'Lat')
        self.assertEqual(actual, expected)
        actual = ch.findcol(self.df, ['lat'])
        self.assertEqual(actual, expected)
        actual = ch.findcol(self.df, ['Latitude', 'Lat'])
        self.assertEqual(actual, expected)
        actual = ch.findcol(self.df, 'slat')
        self.assertIsNone(actual)

    def test_diam2radius(self):
        """Test diam2radius"""
        # Find Diam col
        expected = pd.Series([1.0, 6.0, 11.85], name="Radius")
        actual = ch.diam2radius(self.df)['Radius']
        pdt.assert_series_equal(actual, expected)
        # Find Diameter col
        expected = pd.Series([1.0, 6.0, 11.85], name="Radius")
        df = pd.DataFrame({'Lat': [10, -20., 80.0],
                           'Lon': [14, -40.1, 317.2],
                           'Diameter': [2, 12., 23.7]})
        actual = ch.diam2radius(df)['Radius']
        pdt.assert_series_equal(actual, expected)
        # Supply Lat col
        df = pd.DataFrame({'Lat': [10, -20., 80.0],
                           'Lon': [14, -40.1, 317.2],
                           'Diam': [2, 12., 23.7]})
        expected = pd.Series([5, -10., 40.0], name="Radius")
        actual = ch.diam2radius(df, 'Lat')['Radius']
        pdt.assert_series_equal(actual, expected)
