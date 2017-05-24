# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:17:06 2017

@author: Christian
"""

import os
import pandas as pd
import acerim
import acerim.classes as ac
import unittest

test_path = os.path.join(acerim.__path__[0], 'tests')

class TestCraterDataFrame(unittest.TestCase):
    """Test CraterDataFrame object"""
    crater_csv = os.path.join(test_path,'craters.csv')
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
    test_data = os.path.join(test_path, 'moon.tif')
    ads = ac.AceDataset(test_data, radius=1737)
    
    def test_file_import(self):
        """Import test tif '/tests/moon.tif'"""
        self.assertIsNotNone(self.ads)
        
    def test_geotiff_info(self):
        """Test getDSinfo method for reading geotiff info"""
        ads = self.ads
        actual = ads._getDSinfo()
        expected = (90.0, -90.0, -180.0, 180.0, 6378.137, 4.0)
        self.assertEqual(actual, expected)
        
    def test_isGlobal(self):
        """Test isGlobal method for checking if dataset has 360 degrees of lon"""
        
        is_global = ac.AceDataset(self.test_data, wlon=0, elon=360).isGlobal()
        self.assertTrue(is_global)
        
        not_global = ac.AceDataset(self.test_data, wlon=0, elon=180).isGlobal()
        self.assertFalse(not_global)
        
        
        
        
        