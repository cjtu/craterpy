"""Unittest helper.py."""

import unittest

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

from craterpy import helper as ch


class TestGeoHelpers(unittest.TestCase):
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


class TestShapeGeometryHelpers(unittest.TestCase):
    """Test shape geometry helper functions."""

    def test_get_annular_buffer(self):
        """Test get_annular_buffer with different parameters."""
        point = Point(0, 0)

        # Test with inner=0 (circle)
        buf = ch.get_annular_buffer(point, 1, inner=0, outer=1)
        self.assertTrue(isinstance(buf, shapely.geometry.Polygon))

        # Test with inner>0 (annulus)
        buf = ch.get_annular_buffer(point, 1, inner=0.5, outer=1)
        self.assertTrue(isinstance(buf, shapely.geometry.Polygon))
        self.assertTrue(buf.interiors)  # Should have an inner ring

        # Test with different number of vertices
        buf = ch.get_annular_buffer(point, 1, inner=0, outer=1, nvert=16)
        self.assertEqual(
            len(buf.exterior.coords), 17
        )  # nvert=16 means 16 segments + closing point

    def test_gen_annuli(self):
        """Test gen_annuli with parallel processing."""
        # Setup test data
        centers = [Point(0, 0), Point(45, 45), Point(90, 0)]
        rads = [10, 15, 20]

        # Generate annuli
        annuli = ch.gen_annuli(
            centers, rads, inner=0.5, outer=1, crs="IAU_2015:30100", n_jobs=2
        )

        self.assertEqual(len(list(annuli)), 3)
        self.assertTrue(all(isinstance(a, shapely.geometry.Polygon) for a in annuli))

    def test_create_single_annulus(self):
        """Basic annulus: a valid holed polygon in geodetic coords."""
        annulus = ch.create_single_annulus(
            Point(0, 0), rad=1000, inner=0.5, outer=1, geodetic_crs="IAU_2015:30100"
        )
        self.assertIsInstance(annulus, shapely.geometry.Polygon)
        self.assertTrue(annulus.is_valid)
        self.assertTrue(annulus.interiors)  # inner radius -> hole preserved

    def test_create_single_annulus_keeps_hole_off_antimeridian(self):
        """A holed annulus that does not wrap keeps its hole (no fix applied)."""
        annulus = ch.create_single_annulus(
            Point(0, 0), rad=100000, inner=0.5, outer=1, geodetic_crs="IAU_2015:30100"
        )
        self.assertIsInstance(annulus, shapely.geometry.Polygon)
        self.assertTrue(annulus.is_valid)
        self.assertEqual(len(annulus.interiors), 1)

    def test_create_single_annulus_antimeridian(self):
        """A crater on the antimeridian splits into a valid multipolygon."""
        annulus = ch.create_single_annulus(
            Point(179.9, 0),
            rad=10000,
            inner=0.5,
            outer=1,
            geodetic_crs="IAU_2015:30100",
        )
        self.assertIsInstance(annulus, shapely.geometry.MultiPolygon)
        self.assertTrue(annulus.is_valid)
        # Pieces land in both hemispheres, all within valid longitude bounds.
        self.assertLess(annulus.bounds[0], -170)
        self.assertGreater(annulus.bounds[2], 170)
        self.assertGreaterEqual(annulus.bounds[0], -180)
        self.assertLessEqual(annulus.bounds[2], 180)

    def test_create_single_annulus_poles(self):
        """A crater over either pole yields a valid polar-cap polygon."""
        north = ch.create_single_annulus(
            Point(0, 89.9), rad=10000, inner=0, outer=1, geodetic_crs="IAU_2015:30100"
        )
        self.assertTrue(north.is_valid)
        self.assertEqual(north.bounds[0], -180.0)
        self.assertEqual(north.bounds[2], 180.0)
        self.assertEqual(north.bounds[3], 90.0)

        south = ch.create_single_annulus(
            Point(0, -89.9), rad=10000, inner=0, outer=1, geodetic_crs="IAU_2015:30100"
        )
        self.assertTrue(south.is_valid)
        self.assertEqual(south.bounds[0], -180.0)
        self.assertEqual(south.bounds[2], 180.0)
        self.assertEqual(south.bounds[1], -90.0)


class TestDataframeHelpers(unittest.TestCase):
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
        merged = ch.merge(df1, df2, latcol="lat", loncol="lon", radcol="rad")
        self.assertEqual(len(merged), 5)  # 3 from df1 + 2 unique from df2
        self.assertTrue("a" in merged.columns and "b" in merged.columns)

        # Test value preservation from df1 for matching crater
        self.assertEqual(merged.iloc[0]["rad"], 1)  # Keep df1 value
        self.assertEqual(merged.iloc[0]["b"], 4)  # Get df2 value

        # Test rtol parameter
        merged_strict = ch.merge(
            df1, df2, rtol=0.05, latcol="lat", loncol="lon", radcol="rad"
        )
        self.assertEqual(len(merged_strict), 6)  # No matches with stricter rtol
