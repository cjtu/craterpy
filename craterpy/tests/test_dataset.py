from __future__ import division, print_function, absolute_import
import os.path as p
import unittest
import numpy as np
import gdal
from craterpy.dataset import CraterpyDataset


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

    def test_import_gdal_Dataset(self):
        """Test importing from gdal.Dataset object"""
        ds = gdal.Open(self.moon_tif)
        self.assertIsNotNone(CraterpyDataset(ds))

    def test_import_error(self):
        """Test that importing invalid dataset fails"""
        self.assertRaises(RuntimeError, CraterpyDataset, "Invalid_DS")
        self.assertRaises(RuntimeError, CraterpyDataset, [1, 2])

    def test_set_attrs(self):
        """Test supplying optional attributes to Craterpydataset"""
        pass  # TODO: implement

    def test_get_gdalDataset_attrs(self):
        """Test that wrapped gdal Dataset attrs are accessible"""
        pass  # TODO: implement

    def test_repr(self):
        expected = 'CraterpyDataset with extent (90.0N, -90.0N), '
        expected += '(-180.0E, 180.0E), radius 1737 km, and '
        expected += 'resolution 4.0 ppd'
        actual = self.cds.__repr__()
        self.assertEqual(actual, expected)

    def test_get_geotiff_info(self):
        """Test _get_info() method for reading geotiff info"""
        actual = self.cds._get_geotiff_info()
        expected = (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        self.assertEqual(actual, expected)

    def test_calc_mpp(self):
        """Test .calc_mpp method"""
        cds = self.cds
        xpix = cds.RasterXSize  # [pix]
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
        cds = CraterpyDataset(self.moon_tif, 90, -90, -180, 180)
        self.assertTrue(cds.inbounds(50, 100))
        self.assertTrue(cds.inbounds(-50.1, -100.5))
        self.assertTrue(cds.inbounds(90.0, 180.0))
        self.assertFalse(cds.inbounds(90.1, 100))
        self.assertFalse(cds.inbounds(90.0, -180.5))

    def test_is_global(self):
        """Test .is_global method"""
        is_global = CraterpyDataset(self.moon_tif, wlon=0,
                                    elon=360).is_global()
        self.assertTrue(is_global)

        not_global = CraterpyDataset(self.moon_tif, wlon=0,
                                     elon=180).is_global()
        self.assertFalse(not_global)

    def test_get_roi(self):
        """Test get_roi method"""
        pass  # TODO: Implement

    def test_wrap_roi_360(self):
        """Test wrap_roi_360 method"""
        pass  # TODO: implement
