from __future__ import division, print_function, absolute_import
import os.path as p
import unittest
import numpy as np
import rasterio as rio
from craterpy.dataset import CraterpyDataset
from craterpy.exceptions import DataImportError


class TestCraterpyDataset(unittest.TestCase):
    """Test CraterpyDataset object"""
    def setUp(self):
        import craterpy
        self.data_path = p.join(craterpy.__path__[0], 'data')
        self.moon_tif = p.join(self.data_path, 'moon.tif')
        self.cds = CraterpyDataset(self.moon_tif, radius=1737)

    def test_file_import(self):
        """Test import"""
        self.assertIsNotNone(CraterpyDataset(self.moon_tif))

    def test_import_error(self):
        """Test that importing invalid dataset fails"""
        self.assertRaises(rio.errors.RasterioIOError, CraterpyDataset, "?")

    def test_set_attrs(self):
        """Test supplying optional attributes to Craterpydataset"""
        pass  # TODO: implement

    def test_get_rasterioDataset_attrs(self):
        """Test that wrapped rasterio Dataset attrs are accessible"""
        self.assertIsNotNone(self.cds.read(1))
        with self.assertRaises(AttributeError):
            self.cds.lat

    def test_repr(self):
        expected = 'CraterpyDataset with extent (90.0N, -90.0N), '
        expected += '(-180.0E, 180.0E), radius 1737 km, '
        expected += 'xres 4.0 ppd, and yres 4.0 ppd'
        actual = self.cds.__repr__()
        self.assertEqual(actual, expected)

    def test_get_geotiff_info(self):
        """Test _get_info() method for reading geotiff info"""
        actual = self.cds._get_geotiff_info()
        expected = (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0, 4.0)
        self.assertEqual(actual, expected)

    def test_calc_mpp(self):
        """Test .calc_mpp method"""
        cds = self.cds
        xpix = cds.width  # [pix]
        # Test at equator
        expected = 1000*2*np.pi*cds.radius/xpix  # [m/pix] at lat=0
        self.assertAlmostEqual(cds.calc_mpp(), expected, 5)
        # Test at 50 degrees lat
        lat = 50
        expected = 1000*2*np.pi*cds.radius*np.cos(lat*np.pi/180)/xpix
        self.assertAlmostEqual(cds.calc_mpp(lat), expected, 5)
        # Test at -40 degrees lat
        lat = -40
        expected = 1000*2*np.pi*cds.radius*np.cos(lat*np.pi/180)/xpix
        self.assertAlmostEqual(cds.calc_mpp(lat), expected, 5)
        # Test at 90
        self.assertAlmostEqual(cds.calc_mpp(90), 0)
        # Test at out of bounds lat
        lat = -91
        self.assertRaises(ValueError, cds.calc_mpp, lat)

    def test_inbounds(self):
        """Test inbounds method"""
        # Non-global
        cds = CraterpyDataset(self.moon_tif, 20, -20, 10, 180)
        self.assertTrue(cds.inbounds(10, 15))
        self.assertTrue(cds.inbounds(20, 20))
        self.assertFalse(cds.inbounds(0, 200))
        # Global
        cds = CraterpyDataset(self.moon_tif, 90, -90, -180, 180)
        self.assertTrue(cds.inbounds(50, 100))
        self.assertTrue(cds.inbounds(-50.1, -100.5))
        self.assertTrue(cds.inbounds(90.0, 180.0))
        self.assertTrue(cds.inbounds(90.0, -180.5))
        self.assertFalse(cds.inbounds(90.1, 100))

    def test_is_global(self):
        """Test .is_global method"""
        is_global = CraterpyDataset(self.moon_tif, wlon=0,
                                    elon=360).is_global()
        self.assertTrue(is_global)
        not_global = CraterpyDataset(self.moon_tif, wlon=0,
                                     elon=180).is_global()
        self.assertFalse(not_global)

    def test_wrap_roi_360(self):
        """Test wrap_roi_360 method"""
        pass  # TODO: implement

    def test_get_roi(self):
        """Test get_roi method"""
        extent = (0, 10, 0, 91)
        self.assertRaises(DataImportError, self.cds.get_roi, *extent)
        extent = (170, 190, -10, 10)
        actual = self.cds.get_roi(*extent)
        # TODO: Finish this
        extent = (-190, 170, -10, 10)
        actual = self.cds.get_roi(*extent)
        # TODO: Finish this
