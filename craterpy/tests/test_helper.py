"""Unittest helper.py."""

import unittest

import numpy as np
import pandas as pd
import shapely
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
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

    def test_fix_antimeridian_wrap(self):
        """Test fix_antimeridian_wrap with various edge cases."""
        geodetic_crs = "IAU_2015:30100"  # Moon CRS
        local_crs = ProjectedCRS(
            name="AzimuthalEquidistant(0N, 180E)",
            conversion=AzimuthalEquidistantConversion(0, 180),
            geodetic_crs=geodetic_crs,
        )

        # Test antimeridian wrapping
        antimeridian_annulus = Point(180, 0).buffer(20)  # 20 degree buffer at 180°E
        fixed = ch.fix_antimeridian_wrap(
            antimeridian_annulus, 20, 1.0, geodetic_crs, local_crs
        )
        # Check the polygon was properly split at antimeridian
        self.assertLess(fixed.bounds[0], -170)  # minx should be in western hemisphere
        self.assertGreater(fixed.bounds[2], 170)  # maxx should be in eastern hemisphere

        # Test North pole crossing
        north_local_crs = ProjectedCRS(
            name="AzimuthalEquidistant(85N, 0E)",
            conversion=AzimuthalEquidistantConversion(85, 0),
            geodetic_crs=geodetic_crs,
        )
        pole_annulus = Point(0, 85).buffer(10)  # 10 degree buffer near North pole
        fixed = ch.fix_antimeridian_wrap(
            pole_annulus, 10, 1.0, geodetic_crs, north_local_crs
        )
        # Check that the polygon extends to the pole
        self.assertGreater(fixed.bounds[3], 89)  # maxy should be near 90°N

        # Test South pole crossing
        south_local_crs = ProjectedCRS(
            name="AzimuthalEquidistant(-85S, 0E)",
            conversion=AzimuthalEquidistantConversion(-85, 0),
            geodetic_crs=geodetic_crs,
        )
        pole_annulus = Point(0, -85).buffer(10)  # 10 degree buffer near South pole
        fixed = ch.fix_antimeridian_wrap(
            pole_annulus, 10, 1.0, geodetic_crs, south_local_crs
        )
        # Check that the polygon extends to the pole
        self.assertLess(fixed.bounds[1], -89)  # miny should be near 90°S

        # Control case - polygon that doesn't need fixing
        normal_annulus = Point(0, 0).buffer(5)  # Small buffer at prime meridian
        fixed = ch.fix_antimeridian_wrap(
            normal_annulus, 5, 1.0, geodetic_crs, local_crs
        )
        # Check that the polygon bounds are unchanged
        self.assertTrue(
            abs(normal_annulus.bounds[0] - fixed.bounds[0]) < 1e-10
            and abs(normal_annulus.bounds[2] - fixed.bounds[2]) < 1e-10
        )

        # Test both antimeridian and pole crossing
        complex_local_crs = ProjectedCRS(
            name="AzimuthalEquidistant(85N, 180E)",
            conversion=AzimuthalEquidistantConversion(85, 180),
            geodetic_crs=geodetic_crs,
        )
        complex_annulus = Point(180, 85).buffer(
            15
        )  # Large buffer near pole and antimeridian
        fixed = ch.fix_antimeridian_wrap(
            complex_annulus, 15, 1.0, geodetic_crs, complex_local_crs
        )
        # Check both pole and antimeridian conditions
        self.assertGreater(fixed.bounds[3], 89)  # Extends to North pole
        self.assertLess(fixed.bounds[0], -170)  # Crosses antimeridian westward
        self.assertGreater(fixed.bounds[2], 170)  # Crosses antimeridian eastward

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
        """Test create_single_annulus with different scenarios."""
        geodetic_crs = "IAU_2015:30100"

        # Basic case
        center = Point(0, 0)
        annulus = ch.create_single_annulus(
            center, rad=1000, inner=0.5, outer=1, geodetic_crs=geodetic_crs
        )
        self.assertTrue(isinstance(annulus, shapely.geometry.Polygon))

        # Antimeridian case
        center = Point(179.9, 0)
        annulus = ch.create_single_annulus(
            center, rad=10000, inner=0.5, outer=1, geodetic_crs=geodetic_crs
        )
        self.assertTrue(isinstance(annulus, shapely.geometry.MultiPolygon))

        # Pole case
        center = Point(0, 89.9)
        annulus = ch.create_single_annulus(
            center, rad=10000, inner=0, outer=1, geodetic_crs=geodetic_crs
        )
        self.assertEqual(annulus.bounds[0], -180.0)
        self.assertGreater(annulus.bounds[1], 89.5)
        self.assertEqual(annulus.bounds[2], 180.0)
        self.assertEqual(annulus.bounds[3], 90.0)


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
