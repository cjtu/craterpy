# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:05:26 2016

@author: christian
"""
import ace_settings
from ace_classes import CraterDict, AceDataset
  


if __name__=='__main__':
    ace_settings.init()
    # Testing
    OMATds = AceDataset('OMAT') 
    RAds = AceDataset('RA') 
    cpath = ace_settings.CPATH # Path to craterlists
    LUdict = CraterDict(cpath+'LUcraters.csv')
    SARAdict = CraterDict(cpath+'SARAcraters.csv')   
    OMATdict = CraterDict(cpath+'OMATcraters.csv')   
#    OMATdict.acerim(OMATds)
#    ACEdict=OMATdict.getSubdict(OMATdict.cIDs[:101])
#    ACEdict.acerim(OMATds)
#    LUlt30 = LUdict.getSubdict(drange=(0,30)) # Subdict of craters with 8 <= diam < 30  
#    LUgt30 = LUdict.getSubdict(drange=(100,1000))
#    sub.compare([OMATds,RAds])
#    sub.acerim(OMATds)
#    sub.verify([OMATds,RAds])

#    LUlt30.acerim(OMATds)
#    LUlt30.verify()


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        