"""Unittest crs.py"""

import unittest

import pyproj
from pyproj import CRS

from craterpy.crs import ALL_BODIES, PLANETARY_CRS, get_crs


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def test_all_crs_default(self):
        """Test that every defined CRS loads."""
        for body in ALL_BODIES:
            crs = get_crs(body, "default")
            self.assertIsInstance(crs, pyproj.CRS)

    def test_crs_nondefault(self):
        """Test other defined CRSes."""
        # Test Mars planetographic CRS
        crs = get_crs("mars", "planetographic")
        self.assertIsInstance(crs, pyproj.CRS)
        self.assertEqual(crs.to_authority(), ("IAU_2015", "49901"))

        # Test custom CRS string
        custom_crs = CRS.from_epsg(4326)
        crs = get_crs("mars", custom_crs)
        self.assertEqual(crs, custom_crs)

        # Test invalid CRS raises ValueError
        with self.assertRaises(ValueError):
            get_crs("mars", "invalid_crs")

    def test_unknown_body_or_system(self):
        """Test raises error for unknown crs."""
        with self.assertRaises(ValueError):
            get_crs("pluto", "planetographic")

        with self.assertRaises(ValueError):
            get_crs("krypton")

        with self.assertRaises(ValueError):
            get_crs("moon", "asdf")

    def test_vesta_coords(self):
        """Test longitude conversion between the various vesta coord systems."""
        # Test Claudia Double Prime (IAU_2015 standard, 0° offset)
        crs_cdp = get_crs("vesta", "claudia_dp")
        self.assertEqual(crs_cdp, PLANETARY_CRS["vesta"]["claudia_dp"])

        # Test Claudia Prime (10° W offset)
        crs_cp = get_crs("vesta", "claudia_p")
        self.assertIn("+lon_0=-10", crs_cp.to_proj4())

        # Test Dawn Claudia (210° E offset)
        crs_dc = get_crs("vesta", "dawn_claudia")
        self.assertIn("+lon_0=210", crs_dc.to_proj4())

        # Test invalid vesta system raises ValueError
        with self.assertRaises(ValueError):
            get_crs("vesta", "invalid_system")
