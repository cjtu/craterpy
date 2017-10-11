from __future__ import division, print_function, absolute_import
import os
import unittest
import numpy as np
import pandas as pd
import gdal
import acerim
from acerim import aceclasses as ac

DATA_PATH = os.path.join(acerim.__path__[0], 'sample')


class TestAceDataset(unittest.TestCase):
    """Test AceDataset object"""
    test_dataset = os.path.join(DATA_PATH, 'moon.tif')
    ads = ac.AceDataset(test_dataset, radius=1737)

    def test_file_import(self):
        """Import test tif '/tests/moon.tif'"""
        self.assertIsNotNone(self.ads)

    def test_import_gdal_Dataset(self):
        """Test instantiating AceDataset from gdal.Dataset"""
        gds = gdal.Open(self.test_dataset)
        self.assertIsNotNone(ac.AceDataset(gds))

    def test_import_error(self):
        """Test that importing invalid dataset fails"""
        self.assertRaises(RuntimeError, ac.AceDataset, "Invalid_DS")
        self.assertRaises(ImportError, ac.AceDataset, [1, 2])

    def test_get_gdalDataset_attrs(self):
        """Test that wrapped gdal Dataset attrs are accessible"""
        pass

    def test_get_AceDataset_attrs(self):
        pass

    def test_repr(self):
        expected = 'AceDataset object with latitude (90.0, -90.0)N, '
        expected += 'longitude (-180.0, 180.0)E, radius 1737 km, and '
        expected += 'resolution 4.0 ppd'
        actual = self.ads.__repr__()
        self.assertEqual(actual, expected)

    def test_is_global(self):
        """Test .is_global method"""
        is_global = ac.AceDataset(self.test_dataset,
                                  wlon=0, elon=360).is_global()
        self.assertTrue(is_global)

        not_global = ac.AceDataset(self.test_dataset,
                                   wlon=0, elon=180).is_global()
        self.assertFalse(not_global)

    def test_calc_mpp0(self):
        """Test .calc_mpp method at equator"""
        ads = self.ads
        circum = 2*np.pi*ads.radius  # [m] circumference at lat=0
        xpix = ads.RasterXSize  # [pix]
        expected = circum/xpix  # [m/pix]
        actual = ads.calc_mpp()
        self.assertAlmostEqual(actual, expected, 5)

    def test_calc_mpp50(self):
        """Test .calc_mpp method at 50 degrees latitude"""
        ads = self.ads
        circum = 2*np.pi*np.cos(50*(np.pi/180))*ads.radius
        xpix = ads.RasterXSize
        expected = circum/xpix
        actual = ads.calc_mpp(50)
        self.assertAlmostEqual(actual, expected, 5)

    def test_calc_mpp_gt90(self):
        """Test if calc_mpp() fails above 90 or below -90 degrees latitude"""
        ads = self.ads
        self.assertRaises(ValueError, ads.calc_mpp, 90)
        self.assertRaises(ValueError, ads.calc_mpp, 100)
        self.assertRaises(ValueError, ads.calc_mpp, -100)

    def test_get_info(self):
        """Test _get_info() method for reading geotiff info"""
        ads = self.ads
        actual = ads._get_info()
        expected = (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        self.assertEqual(actual, expected)

    def test_get_roi(self):
        """Test get_roi method"""
        pass

    def test_wrap_lon(self):
        pass