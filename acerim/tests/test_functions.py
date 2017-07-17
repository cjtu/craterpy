# -*- coding: utf-8 -*-
"""
Created on Fri May 26 07:33:17 2017

@author: Christian
"""
import os
import unittest
import numpy as np
import acerim
from acerim import acefunctions as af
from acerim import aceclasses as ac

DATA_PATH = os.path.join(acerim.__path__[0], 'sample')

#%% Test Compute Stats
class Test_compute_stats(unittest.TestCase):
    """Test computeStats function"""
    crater_csv = os.path.join(DATA_PATH,'craters.csv')
    cdf = ac.CraterDataFrame(crater_csv)
    test_dataset = os.path.join(DATA_PATH, 'moon.tif')
    ads = ac.AceDataset(test_dataset, radius=1737)

    def test_first_mean(self):
        """Test mean on first crater in cdf"""
        #af.compute_stats(self.cdf, self.ads, 'mean', self.cdf.index[0:5])
        # TODO: fix this test (choose appropriate indices)

#%% Test ROI manipulation functions
class Test_circle_mask(unittest.TestCase):
    """Test ring_mask function"""
    def test_trivial(self):
        """Test radius 0"""
        actual = af.circle_mask(np.ones((3,3)), 0)
        expected = np.array([[False, False, False],
                             [False, False, False],
                             [False,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
    
    def test_odd(self):
        """Test roi with odd side length"""
        actual = af.circle_mask(np.ones((5,5)), 2)
        expected = np.array([[False, False, True,  False, False],
                             [False, True,  True,  True,  False],
                             [True,  True,  True,  True, True],
                             [False, True,  True,  True,  False],
                             [False, False, True,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
        
    def test_even(self):
        """Test roi with even side length"""
        actual = af.circle_mask(np.ones((4,4)), 2)
        expected = np.array([[False,  True,  True, False],
                             [ True,  True,  True,  True],
                             [ True,  True,  True,  True],
                             [False,  True,  True, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
        
    def test_offcenter(self):
        """Test specifying off center location"""
        actual = af.circle_mask(np.ones((5,5)), 2, center=(3,2))
        expected = np.array([[False, False, False, True,  False],
                             [False, False, True,  True,  True],
                             [False, True,  True,  True,  True],
                             [False, False, True,  True,  True],
                             [False, False, False, True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
     

class Test_ring_mask(unittest.TestCase):
    """Test ring_mask function"""
    def test_trivial(self):
        actual = af.ring_mask(np.ones((3,3)), 0, 0)
        expected = np.array([[False, False, False],
                             [False, False, False],
                             [False,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
    
    def test_odd(self):
        """Test roi with odd side length"""
        actual = af.ring_mask(np.ones((5,5)), 1, 2)
        expected = np.array([[False, False, True,  False, False],
                             [False, True,  False, True,  False],
                             [True,  False, False, False, True],
                             [False, True,  False, True,  False],
                             [False, False, True,  False, False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
        
    def test_even(self):
        """Test roi with even side length"""
        actual = af.ring_mask(np.ones((4,4)), 1.5, 2)
        expected = np.array([[False, True,  True,  False],
                             [True,  False, False, True],
                             [True,  False, False, True],
                             [False, True,  True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
        
    def test_offcenter(self):
        """Test specifying off center location"""
        actual = af.ring_mask(np.ones((5,5)), 1, 2, center=(3,2))
        expected = np.array([[False, False, False, True,  False],
                             [False, False, True,  False, True],
                             [False, True,  False, False, False],
                             [False, False, True,  False, True],
                             [False, False, False, True,  False]])
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))
         

#%% Image Helper Functions
class Test_m2deg(unittest.TestCase):
    """Test m2deg functions"""
    def test_basic(self):
        """Test simple"""
        actual = af.m2deg(400, 10, 20)
        expected = 2.0
        self.assertEqual(actual, expected)
    
    def test_float(self):
        """Test float"""
        actual = af.m2deg(1500.0, 4.0, 0.25)
        expected = 1500.0
        self.assertEqual(actual, expected)
        