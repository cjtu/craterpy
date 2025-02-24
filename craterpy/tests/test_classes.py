"""Unittest classes.py."""

import warnings
from pathlib import Path
import unittest
import pyproj
import shapely
from shapely.testing import assert_geometries_equal
import craterpy
from craterpy.classes import CraterDatabase, CRS_DICT


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def setUp(self):
        data_dir = Path(craterpy.__path__[0], "data")
        self.moon_tif = data_dir / "moon.tif"
        self.crater_list = data_dir / "craters.csv"

    def test_add_annuli(self):
        """Test adding annular shapefiles to CraterDataBase."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli(1, 2, "ejecta")
        # Check that ejecta appears in the string repr
        self.assertIn("ejecta", str(cdb))
        # Test that ejecta was registered as a propety and contains a shapely geom
        self.assertIsInstance(cdb.ejecta[0], shapely.geometry.Polygon)

    def test_annuli_precision(self):
        """Test that simple and precise annuli mostly agree."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli(0, 1, "precise", precise=True)
        cdb.add_annuli(0, 1, "simple", precise=False)
        assert_geometries_equal(cdb.precise, cdb.simple, tolerance=1)

    def test_get_stats(self):
        """Test getting statistics on a region for a raster."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli(1, 1.1, "rim")
        stats = cdb.get_stats(self.moon_tif, "rim", ["count"])
        self.assertIn("count_rim", stats.columns)

    def test_get_stats_parallel(self):
        """Test parallellization of get_stats for multiple rasters/regions."""
        pass

    def test_plot(self):
        """Test CraterDatabase summary plot."""
        pass

    def test_body_crs_all(self):
        """Test that every defined CRS loads."""
        for body in CRS_DICT.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Vesta*")
                cdb = CraterDatabase(self.crater_list, body)
                for crs in [
                    v for k, v in cdb.__dict__.items() if k.startswith("_crs")
                ]:
                    self.assertIsInstance(crs, pyproj.CRS)

    def test_vesta_coord_correction(self):
        """Test Vesta's various coordinate systems."""
        pass

    def test_import_dataframe(self):
        """Test importing from a dataframe."""
        pass

    def test_units(self):
        """Test importing with radii in m or km."""
        pass

    def test_rad_vs_diam(self):
        """Test importing with radii and diam columns."""
        pass

    def test_gen_annulus_simple(self):
        """Test generating simple annuli."""
        pass

    def test_gen_annulus_precise(self):
        """Test generating precise annuli."""
        pass

    def test_antimeridian_splitting(self):
        """Test generating annuli when split over the antimeridian."""
        pass

    def test_pole_crossing(self):
        """Test annuli that cross the North or South pole."""
        pass
