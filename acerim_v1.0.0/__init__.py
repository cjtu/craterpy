# -*- coding: utf-8 -*-
"""
Initialize the ACERIM package. Import all functions for use at top level.
Makes top level functions available through acerim_v2.function() when the 
acerim_v2 package is imported.

Created on Wed Nov 30 21:49:08 2016

@author: christian
"""
from main import acerim, compare, verify

if __name__=='__main__':
    acerim() # test script here