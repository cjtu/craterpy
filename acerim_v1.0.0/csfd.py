# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 23:16:52 2017

Endeavour into pandas and databases

@author: Christian
"""
import numpy as np
import pandas as pd
import os
#CPATH = "/Users/christian/Google Drive/Research/2016: OMAT & RA (Acerim)/Craterlists/" 
CPATH = "/Users/christian/Desktop/lunar_datasets/cdb" 
os.chdir(CPATH)

def save2diam(dataframe,fname='untitled.diam',area=38000000):
    """
    Generate craterstats .diam files from a Pandas dataframe.
    Dataframme must have 'diam' as a column. Diam must be in km
    """
    if (dataframe['diam'] > 1000).any():
        raise Exception('Diameter too high')
    if len(dataframe['diam']) == 0:
        raise Exception('No diameters')
    with open(fname,'w') as f:
        f.write('area = {}\n'.format(area))
        f.write('crater = {diameter\n')
        for diam in dataframe['diam']:
            f.write('{}\n'.format(diam))
        f.write('}')

def find_matching(df1,df2):
    matching = np.zeros(len(df1))
    for i in df1.index:
        j = 0
        lats = df2['lat'] - df1['lat'][i]
        lons = df2['lon'] - df1['lon'][i]
        diams = (df2['diam'] - df1['diam'][i])/df1['diam'][i]
        if (np.sqrt(lats**2 + lons**2 + diams**2) < 1).any():
            j = np.argmin(np.sqrt(lats**2 + lons**2 + diams**2))
        matching[i] = j
    return matching
    
###IMPORTS###
lu = pd.read_csv('LUcraters.csv')
lpi = pd.read_csv('LPIcraters.csv')
sai = pd.read_csv('SAIcraters.csv')
omat = pd.read_csv('OMAT_CTU.csv')
ra = pd.read_csv('RAcraters.csv')

###Clean up formats
lons = sai['lon']
lons[lons>=180] -= 360
sai['lon'] = lons
        
###Subset dataframes
omat_m = omat[(omat['tag']=='standard') & (omat['mare']!=-1)]
omat_hl = omat[(omat['tag']=='standard') & (omat['mare']==-1)]

ra_m = ra[ra['mare'] != 'H'] 
ra_hl = ra[ra['mare'] == 'H'] 
ra_lt266 = ra[ra['age'] <= 266] # younger than 266 Mya
ra_gt266 = ra[ra['age'] > 266] # older than 266 Mya

ra_lt200 = ra[ra['age'] <= 200] # younger than 266 Mya
ra_lt600 = ra[(ra['age'] > 200) & (ra['age'] <= 600)] # older than 266 Mya
ra_gt600 = ra[ra['age'] > 600] # younger than 266 Mya

###Save .diam files
save2diam(lu, 'LU.diam',37932328) # Global
save2diam(omat,'OMAT.diam',33102197) # 50N50S
save2diam(omat_m,'OMAT_m.diam',8275500)
save2diam(omat_hl,'OMAT_hl.diam',26814497)
save2diam(ra,'RA.diam',37356049) #70N70S
save2diam(ra_m,'RA_m.diam',7097649)
save2diam(ra_hl,'RA_hl.diam',30258400)
save2diam(ra_lt266,'RA_lt266.diam',37356049)
save2diam(ra_gt266,'RA_gt266.diam',37356049)
             
save2diam(ra_lt200,'RA_lt200.diam',37356049)
save2diam(ra_lt600,'RA_lt600.diam',37356049)                          
save2diam(ra_gt600,'RA_gt600.diam',37356049)  
#omat[:5]['Diam']
#omat[:5]['m/h'] == 'm'
#
#is_mare = omat['m/h'] == 'm'
#omat[is_mare]
#
#omat[is_mare]['Diam']
#
#test = pd.read_excel(CPATH+'Fresh Craters.xlsx')
#
#efile = pd.ExcelFile(CPATH+'Fresh Craters.xlsx')
#v2 = pd.read_excel(efile, 'v2.0')

#omat = pd.read_csv('CRATER_OMAT_pt8.csv')

#LU = pd.read_csv('LUcraters2.csv')