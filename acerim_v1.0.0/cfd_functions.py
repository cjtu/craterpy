# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:42:08 2017

@author: christian
"""

def wcsd(dlims,alpha):
    C = getC(dlims, alpha)
    G = getG(dlims, alpha, C)
    diams, csd = getND(dlims, alpha, C, G)
    return diams, G[0]*csd
    
def getND(dlims, alpha, C, G):
    """normalized csd eq (17)"""
    N = len(dlims) # num intervals
    for i in range(N-1):
        diams = np.arange(dlims[i],dlims[i+1])
        n = len(diams)
        csd = np.zeros(n)
        for j in range(n):
            csd[j] =(C[i]*(diams[j]**(-alpha[i])-dlims[i+1]**(-alpha[i])) + G[i+1])/G[0]
        
        if i == 0:
            all_diams = diams
            all_csd = csd
        else:
            all_diams = np.concatenate((all_diams,diams))
            all_csd = np.concatenate((all_csd,csd))
    return all_diams, all_csd
    
def getG(dlims, alpha, C):
    """ G, eq (19)"""
    N = len(dlims)
    g = np.zeros(N)
    for i in range(N-1):
        tot = 0
        for j in range(i,N-1):
            tot += C[j]*(dlims[j]**(-alpha[j]) - dlims[j+1]**(-alpha[j]))
        g[i] = tot
    return g
    
def getC(dlims, alpha):
    """C, eq (20)"""
    N = len(dlims)
    c = np.ones(N-1)
    for i in range(1,N-1):
        tot = alpha[0]/alpha[i]
        for j in range(1,i):
            tot *= dlims[j]**(alpha[j]-alpha[j-1])
        c[i] = tot
    return c

def isochron(t):
    return 5.44e-14 * (np.exp(6.93*t)-1) + 8.38e-4 * t

def getCdiams(cdict,fname,area):
    """get formatted file of crater diameters for a size frequency"""
    outfile = fname
    if fname[-5:] != '.diam':
        outfile = outfile + '.diam'
    with open(outfile,'w') as f:
        f.write('area = '+str(area)+'\n')
        f.write('crater = {diameter\n')
        for cID in cdict.cIDs:
            f.write(str(cdict[cID].diam)+'\n')
        f.write('}')
        
        

###

#    LUdict.cfd(fig='cfd', label='Goran',mark='+')
#    SARAdict.cfd(fig='cfd', label='Sara',mark='.',NPF=True)
#    LUdict.rfd(fig='rfd',NPF=True,mark='+')
#    

#    SARAdict.rfd(fig='rfd')
#    LUgt30.cfd(fig='cfd') 
#    LUlt30.cfd(fig='cfd')

#    # Wang et al
#    import numpy as np
#    alpha1 = [1.17, 1.88, 3.17, 1.4]
#    dlims1 = [10,49,120,251,2500]
#    
#    alpha2 = [1.96]
#    dlims2 = [10,100]
#    
#    wd1, wcsd1 = wcsd(dlims1, alpha1)
#    wd2, wcsd2 = wcsd(dlims2, alpha2)