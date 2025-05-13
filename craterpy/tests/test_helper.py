"""Unittest helper.py."""

import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdt
from craterpy import helper as ch


class Test_geo_helpers(unittest.TestCase):
    """Test Geospatial Helper Functions"""

    def test_lon360(self):
        """Test lon360 function"""
        actual = ch.lon360(0)
        expected = 0
        self.assertEqual(actual, expected)
        actual = ch.lon360(360)
        expected = 0
        self.assertEqual(actual, expected)
        actual = ch.lon360(20)
        expected = 20
        self.assertEqual(actual, expected)
        actual = ch.lon360(185)
        expected = 185
        self.assertEqual(actual, expected)
        actual = ch.lon360(355)
        expected = 355
        self.assertEqual(actual, expected)
        actual = ch.lon360(-175)
        expected = 185
        self.assertEqual(actual, expected)
        actual = ch.lon360(-5)
        expected = 355
        self.assertEqual(actual, expected)

    def test_lon180(self):
        """Test lon180 function"""
        actual = ch.lon180(0)
        expected = 0
        self.assertEqual(actual, expected)
        actual = ch.lon180(360)
        expected = 0
        self.assertEqual(actual, expected)
        actual = ch.lon180(20)
        expected = 20
        self.assertEqual(actual, expected)
        actual = ch.lon180(185)
        expected = -175
        self.assertEqual(actual, expected)
        actual = ch.lon180(355)
        expected = -5
        self.assertEqual(actual, expected)
        actual = ch.lon180(-175)
        expected = -175
        self.assertEqual(actual, expected)
        actual = ch.lon180(-5)
        expected = -5
        self.assertEqual(actual, expected)

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
        self.assertTrue(ch.inglobal(-5.5, 360))
        self.assertFalse(ch.inglobal(91, 0))
        self.assertTrue(ch.inglobal(0, 181))
        self.assertTrue(ch.inglobal(0, -1))


class Test_dataframe_helpers(unittest.TestCase):
    """Test pandas.DataFrame helper functions"""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Lat": [10, -20.0, 80.0],
                "Lon": [14, -40.1, 317.2],
                "Diam": [2, 12.0, 23.7],
            }
        )

    def test_findcol(self):
        """Test findcol"""
        expected = "Lat"
        actual = ch.findcol(self.df, "Lat")
        self.assertEqual(actual, expected)
        actual = ch.findcol(self.df, ["lat"])
        self.assertEqual(actual, expected)
        actual = ch.findcol(self.df, ["Latitude", "Lat"])
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            actual = ch.findcol(self.df, "slat")

    def test_find_rad_or_diam_col(self):
        """Test find_rad_or_diam_col"""
        # Test with a column named "radius"
        df = pd.DataFrame({"radius": [1, 2, 3], "other": [4, 5, 6]})
        expected = "radius"
        self.assertEqual(ch.find_rad_or_diam_col(df), expected)

        # Test with a column named "rad"
        df = pd.DataFrame({"rad": [1, 2, 3], "other": [4, 5, 6]})
        expected = "rad"
        actual = ch.find_rad_or_diam_col(df)
        self.assertEqual(actual, expected)

        # Test with a column named "D"
        df = pd.DataFrame({"D": [1, 2, 3], "other": [4, 5, 6]})
        expected = "D"
        actual = ch.find_rad_or_diam_col(df)
        self.assertEqual(actual, expected)

        # Test with multiple possible matches, exact match should take precedence
        df = pd.DataFrame({"radius": [1, 2, 3], "rad": [4, 5, 6]})
        expected = "radius"
        actual = ch.find_rad_or_diam_col(df)
        self.assertEqual(actual, expected)

        # Test case insensitive
        df = pd.DataFrame({"Radius (km)": [1, 2, 3], "other": [4, 5, 6]})
        expected = "Radius (km)"
        actual = ch.find_rad_or_diam_col(df)
        self.assertEqual(actual, expected)

        # Test no exact match
        df = pd.DataFrame({"circ_radius_m": [1, 2, 3], "other": [4, 5, 6]})
        expected = "circ_radius_m"
        actual = ch.find_rad_or_diam_col(df)
        self.assertEqual(actual, expected)

        # Test with no matching column
        df = pd.DataFrame({"other": [1, 2, 3]})
        with self.assertRaises(ValueError):
            ch.find_rad_or_diam_col(df)

    def test_merge(self):
        """Test spatial merge of crater dataframes."""
        # Create test dataframes
        df1 = pd.DataFrame(
            {
                "lat": [0, 10, -10],
                "lon": [0, 10, -10],
                "rad": [1, 2, 3],
                "a": [1, 2, 3],
            }
        )
        df2 = pd.DataFrame(
            {
                "lat": [0, 20, -20],  # First crater matches df1
                "lon": [0, 20, -20],
                "rad": [1.1, 4, 5],  # Within rtol=0.5 of df1
                "b": [4, 5, 6],
            }
        )

        # Test basic merge
        merged = ch.merge(df1, df2)
        self.assertEqual(len(merged), 5)  # 3 from df1 + 2 unique from df2
        self.assertTrue("a" in merged.columns and "b" in merged.columns)

        # Test value preservation from df1 for matching crater
        self.assertEqual(merged.iloc[0]["rad"], 1)  # Keep df1 value
        self.assertEqual(merged.iloc[0]["b"], 4)  # Get df2 value

        # Test rtol parameter
        merged_strict = ch.merge(df1, df2, rtol=0.05)
        self.assertEqual(
            len(merged_strict), 6
        )  # No matches with stricter rtol
