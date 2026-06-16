"""Unittest crs.py"""

import unittest

import pyproj
from pyproj import CRS

from craterpy.crs import ALL_BODIES, get_crs


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def test_all_crs_default(self):
        """Test that every defined CRS loads."""
        for body in ALL_BODIES:
            crs = get_crs(body, "default")
            self.assertIsInstance(crs, pyproj.CRS)

    def test_standard_body_delegates_to_planetarypy(self):
        """Standard ocentric/ographic CRS come from planetarypy's IAU authority."""
        # Mars planetocentric -> IAU_2015 ocentric code (naif*100 + 0)
        self.assertEqual(
            get_crs("mars", "planetocentric").to_authority(), ("IAU_2015", "49900")
        )
        # Moon default is planetocentric
        self.assertEqual(get_crs("moon").to_authority(), ("IAU_2015", "30100"))

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

    def test_input_crs_passthrough_forms(self):
        """The CRS forms accepted as CraterDatabase input_crs pass through get_crs."""
        expected = CRS.from_user_input("IAU_2015:200000100")  # Ceres ocentric

        # 1. Authority code string
        self.assertEqual(get_crs("ceres", "IAU_2015:200000100"), expected)

        # 2. pyproj.CRS object
        self.assertEqual(get_crs("ceres", expected), expected)

        # 3. proj4 string (the Ceres sphere, R=487300 m)
        crs = get_crs("ceres", "+proj=longlat +R=487300 +no_defs")
        self.assertIsInstance(crs, pyproj.CRS)
        self.assertEqual(crs.ellipsoid.semi_major_metre, 487300)

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
        self.assertEqual(crs_cdp.to_authority(), ("IAU_2015", "200000400"))

        # Test Claudia Prime (10° W offset)
        crs_cp = get_crs("vesta", "claudia_p")
        self.assertIn("+lon_0=10", crs_cp.to_proj4())

        # Test Dawn Claudia (210° E offset)
        crs_dc = get_crs("vesta", "dawn_claudia")
        self.assertIn("+lon_0=-210", crs_dc.to_proj4())

        # Test invalid vesta system raises ValueError
        with self.assertRaises(ValueError):
            get_crs("vesta", "invalid_system")
