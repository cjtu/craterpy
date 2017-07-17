# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:17:06 2017

@author: Christian
"""

import os
import pandas as pd
import unittest
import numpy as np
import acerim
from acerim import aceclasses as ac

DATA_PATH = os.path.join(acerim.__path__[0], 'sample')

class TestCraterDataFrame(unittest.TestCase):
    """Test CraterDataFrame object"""
    crater_csv = os.path.join(DATA_PATH,'craters.csv')
    cdict = {'Lat' : [10, -20., 80.0],
             'Lon' : [14, -40.1, 317.2],
             'Diam' : [2, 12., 23.7]}
    
    def test_file_import(self):
        """Import from test file '/tests/craters.csv'"""
        cdf = ac.CraterDataFrame(self.crater_csv)
        self.assertIsNotNone(cdf)
        
    def test_dict_import(self):
        """Import from dict"""
        cdf = ac.CraterDataFrame(self.cdict)
        self.assertIsNotNone(cdf)
    
    def test_pandas_dataframe_import(self):
        """Import from pandas.DataFrame object"""
        pdf = pd.DataFrame(pd.read_csv(self.crater_csv))
        cdf = ac.CraterDataFrame(pdf)
        self.assertIsNotNone(cdf)
        
    def test_specify_index(self):
        """Test defining custom index"""
        cdf = ac.CraterDataFrame(self.cdict, index=['A', 'B', 'C'])
        self.assertIn('A', cdf.index)
        
    def test_specify_columns(self):
        """Test defining custom index"""
        cdf = ac.CraterDataFrame(self.cdict, columns=['Diameter', 'Lat','Lon'])
        self.assertIn('Diameter', cdf.columns)   
        
    def test_find_latcol(self):
        """Find latitude column in loaded data"""
        cdf = ac.CraterDataFrame(self.cdict)
        self.assertEqual(cdf.latcol, 'Lat')
        
    def test_find_loncol(self):
        """Find longitude column in loaded data"""
        cdf = ac.CraterDataFrame(self.cdict)
        self.assertEqual(cdf.loncol, 'Lon')

    def test_make_radcol(self):
        """Make radius column if diameter is specified in loaded data"""
        cdf = ac.CraterDataFrame(self.cdict)
        self.assertEqual(cdf.radcol, '_Rad')
        actual = cdf.loc[0, '_Rad']
        expected = 0.5*cdf.loc[0, 'Diam']
        self.assertEqual(actual, expected)
        
        
class TestAceDataset(unittest.TestCase):
    """Test AceDataset object"""
    test_dataset = os.path.join(DATA_PATH, 'moon.tif')
    ads = ac.AceDataset(test_dataset, radius=1737)
    
    def test_file_import(self):
        """Import test tif '/tests/moon.tif'"""
        self.assertIsNotNone(self.ads)
        
    def test_get_info(self):
        """Test .get_info method for reading geotiff info"""
        ads = self.ads
        actual = ads.get_info()
        expected = (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        self.assertEqual(actual, expected)
        
    def test_is_global(self):
        """Test .is_global method for checking if dataset has 360 degrees of lon""" 
        is_global = ac.AceDataset(self.test_dataset, wlon=0, elon=360).is_global()
        self.assertTrue(is_global)
        
        not_global = ac.AceDataset(self.test_dataset, wlon=0, elon=180).is_global()
        self.assertFalse(not_global)
        
    def test_calc_mpp0(self):
        """Test .calc_mpp method at equator"""
        ads = self.ads
        circum = 2*np.pi*ads.radius # [m] circumference at lat=0
        xpix = ads.RasterXSize # [pix]
        expected = circum/xpix # [m/pix]
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
        
    def test_calc_mpp90(self):
        """Test .calc_mpp method fails at 90 degrees latitude"""
        ads = self.ads
        self.assertRaises(ValueError, ads.calc_mpp, 90)
        
    def test_get_roi(self):
        """Test get_roi method"""
        pass
        
        
        
        