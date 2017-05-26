# -*- coding: utf-8 -*-
"""
Created on Fri May 26 07:33:17 2017

@author: Christian
"""
import os
import unittest
import acerim
import functions as f

test_path = os.path.join(acerim.__path__[0], 'tests')

# Image Helper Functions
class Test_m2deg(unittest.TestCase):
    """Test m2deg functions"""
    
    def test_m2deg(self):
        """Test simple"""
        actual = f.m2deg(400, 10, 20)
        expected = 2.0
        self.assertEqual(actual, expected)
    
    def test_m2deg(self):
        """Test float"""
        actual = f.m2deg(1500.5, 4.0, 0.25)
        expected = 1502.0
        self.assertEqual(actual, expected)
        