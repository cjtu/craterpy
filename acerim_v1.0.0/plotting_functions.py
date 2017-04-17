#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:00:12 2016

@author: christian
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# GLOBALS #
RMOON = 1737.400 #km

def plot_metrics(cIDs, cdict, METRICS):   
    M = 1
    NPLOTS = len(METRICS)
    m_fig, axarr = plt.subplots(NPLOTS, sharex=True, num=3)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(c) for c in np.linspace(0, 1, len(cIDs))]
    for i, cid in enumerate(cIDs):    
        c = cdict[cid]
        radii = c.stats['radii']
        clabel = c.name + ' dia={:.3}km'.format(c.diam)

        for j, m in enumerate(METRICS):
            if m+'_acerim' in c.stats:
                clabel = '"{}" {:.3}km ace={:.3}'.format(c.name, c.diam, c.stats[m+'_acerim'])
            # Plot Metrics (even index plots)
            #axarr[M*j].plot(radii,c.stats[METRICS[j]],'-',color=COLORS[i])
            axarr[M*j].plot(radii,c.stats[m+'_shift'],'+',color=colors[i]) # Downshifted metric
            axarr[M*j].plot(c.stats['fit_'+m+'_xarr'],c.stats['fit_'+m+'_shift'],color=colors[i],label=clabel) #Downshifted fit
            axarr[M*j].set_title('{0} vs RADIUS'.format(m.upper()))
            axarr[M*j].legend(ncol=2,loc='upper right')
            axarr[M*j].set_xlabel('Radius (# of crater radii)')
    
#            if DMETRICS: # Plot dMetrics (odd index plots)
#                axarr[M*j+1].plot(radii[1:],c.stats[DMETRICS[j]],color=colors[i],label=clabel)
#        #        axarr[2*j+1].plot(metrics[FDMETRICS[j]+'_xarr'],metrics[FDMETRICS[j]],color=COLORS[i],label=clabel)    
#                axarr[M*j+1].set_title('{0} vs RADIUS'.format(DMETRICS[j].upper()))
#                axarr[M*j+1].legend(ncol=2,loc='upper right')
#                axarr[M*j+1].set_xlabel('Radius (# of crater radii)')
#            
#            if D2METRICS: # Plot d2Metrics
#                axarr[M*j+2].plot(radii[2:],c.stats[D2METRICS[j]],color=colors[i],label=clabel) 
#                axarr[M*j+2].set_title('{0} vs RADIUS'.format(D2METRICS[j].upper()))
#                axarr[M*j+2].legend(ncol=2,loc='upper right')
#                axarr[M*j+2].set_xlabel('Radius (# of crater radii)')        
    
    m_fig.set_figheight(M*20)
    m_fig.set_figwidth(15)
    axarr[0].set_xlim((1, c.stats['radii'][-1]))
#    axarr[0].set_ylim((-0.02,0.09))
#    axarr[1].set_ylim((-0.02,0.08))
    plt.setp([a.get_xticklabels() for a in axarr], visible=True)
    plt.show()

def plotSfd(diams,sizefreq,fig=0,label='',mark='o'):    
    # Plot size freq distribution
    if not fig:
        plt.figure()
    else:
        plt.figure(fig)
    if label:
        plt.plot(diams, sizefreq, '*', label=label+', N='+str(len(sizefreq)),marker=mark,fillstyle='none')
        plt.legend(loc='best')
        plt.title('Size-frequency')
    else:
        plt.plot(diams, sizefreq, '--')
        plt.title('Size-frequency, N={}'.format(len(sizefreq)))
    plt.yscale('log')
    plt.xscale('log')
#    plt.xlim([8,40])
#    plt.ylim((min(sizefreq),max(sizefreq)))
    plt.xlabel('diameter (km)')
    plt.ylabel('frequency ')  
    
    
def plot_sfd(cdict, cIDs, norm=1, mode='fullmoon'):
    """
    Plot the Size-Frequency distribution of the craters listed in cIDs from
    cdict where norm is the normalization factor (defaults to 1 for whole Moon).
    """
    clons = np.array([cdict[cid].lon for cid in cIDs])
    clons[clons > 180] -= 360
    if mode == 'nearside':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-180 <= clons[i] < -90) or (90 <= clons[i] < 180)]
        norm /= 2
    elif mode == 'farside':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-90 <= clons[i] < 90)]
        norm /= 2
    elif mode == 'leading':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (-180 <= clons[i] < 0)]
        norm /= 2
    elif mode == 'trailing':
        cIDs = [cIDs[i] for i in range(len(cIDs)) if (0 <= clons[i] < 180)]
        norm /= 2    
                
    a_moon = 4*np.pi*(RMOON**2) # Area of the moon
    diams = np.array([cdict[cid].diam for cid in cIDs])
    hist, bins = np.histogram(diams, bins = int(max(diams)-min(diams)))  
    sfd_diams = bins[:-1][hist>0]
    hist = hist[hist>0]
    norm_hist = norm*(hist/a_moon)
    sizefreq = np.cumsum(norm_hist[::-1])[::-1]

    # Plot size freq distribution
    plt.figure()
    plt.plot(sfd_diams, sizefreq, 'b*')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([min(diams),max(diams)])
    plt.ylim([min(sizefreq),max(sizefreq)])
    plt.title('Size-frequency, N={} ({})'.format(len(cIDs), mode))
    plt.xlabel('diameter (km)')
    plt.ylabel('frequency ')  


def plot_lfd(cdict, cIDs):
    """
    Plot the longitudinal frequency distribution of cIDs craters in cdict.
    Fit=True will fit a sinusoid to the distribution and return the apex:antapex
    ratio.
    """
    dbin = 10 # bin width (deg)
    bins = np.arange(-180,180+dbin,dbin)
    clons = np.array([cdict[cid].lon for cid in cIDs])
    clons[clons > 180] -= 360
    hist,bins = np.histogram(clons, bins = bins)
    
    #TODO Calculate apex:antapex ratio (Change to leastsq w/ sin function)
    apex = hist[bins[:-1] == -90.][0]
    antapex = hist[bins[:-1] == 90.][0]
    aaratio = apex/antapex
    norm_hist = (hist/np.sum(hist))
    
    plt.figure()
    plt.plot(bins[:-1],norm_hist,drawstyle='steps')
    plt.xlim([-180,180])
    plt.title('Longitudinal distribution, N={}, apex:antapex={}'.format(len(clons),aaratio))
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Fraction of craters')
    
    
def plot_stats(cIDs, cdict, stats):
    def getMarker(c):
        if c.loc == 'Mare':
            cmarker = '^'
        elif c.loc == 'Near Highlands':
            cmarker = 'D'
        elif c.loc == 'Near Mare':
            cmarker = 's'
        elif c.loc == 'Highlands':
            cmarker = 'p'
        else:
            cmarker = '+'
        return cmarker
        
    def getColor(c):
        if c.omat == 'young':
            ccol = 'blue'
        elif c.omat == 'intermediate':
            ccol = 'green'
        elif c.omat == 'old':
            ccol = 'red'
        else:
            ccol = 'black'
        return ccol
        
    def getHandles():
        young = mpatches.Patch(color='blue',label='young OMAT')
        inter = mpatches.Patch(color='green',label='inter OMAT')
        old = mpatches.Patch(color='red',label='old OMAT')
        mare = mlines.Line2D([],[],color='black',marker='^',label='Mare')
        nearhl = mlines.Line2D([],[],color='black',marker='D',label='Near HL')
        nearmare = mlines.Line2D([],[],color='black',marker='s',label='Near Mare')
        hl = mlines.Line2D([],[],color='black',marker='p',label='HL')

        return [young,inter,old,mare,nearhl,nearmare,hl]
        
    NPLOTS = 2*len(stats)
    fig, axarr = plt.subplots(nrows=NPLOTS)
    for i,cid in enumerate(cIDs):
        c = cdict[cid]
        clabel = c.name
        ccol = getColor(c)
        cmark = getMarker(c)
        handles = getHandles()
        for j,stat in enumerate(stats):
            k = j + len(stats) 
            axarr[j].plot(c.diam, c.stats[stat], label=clabel,color=ccol,marker=cmark,markersize=15)
            axarr[j].set_title('Crater {} vs Diameter'.format(stat))
            axarr[j].set_xlabel('Diameter')
            axarr[j].set_ylabel(stat)
            axarr[j].legend(handles=handles)
            #axarr[j].legend(ncol=2,loc='upper right')
            
            axarr[k].plot(c.age, c.stats[stat], label=clabel,color=ccol,marker=cmark,markersize=15)
            axarr[k].set_title('Crater {} vs Age'.format(stat))
            axarr[k].set_xlabel('Age')
            axarr[k].set_ylabel(stat) 
            axarr[k].legend(handles=handles)
            #axarr[k].legend(ncol=2,loc='upper right')
    

    fig.set_figheight(30)
    fig.set_figwidth(15)        
    plt.setp([a.get_xticklabels() for a in axarr], visible=True)   
    plt.show()         
    
    
def plot_medians(cIDs, cdict):    
    NPLOTS = 2
    m_fig, axarr = plt.subplots(NPLOTS, sharex=True, num=3)
    CMAP = plt.get_cmap('rainbow')
    COLORS = [CMAP(c) for c in np.linspace(0, 1, len(cIDs))]
    for i, cid in enumerate(cIDs):    
        c = cdict[cid]
        radii = c.stats['radii']
        clabel = c.name
        
        axarr[0].plot(radii,c.stats['median'],'-',color=COLORS[i],label=clabel) 
        axarr[0].set_title('Median (actual)')
        axarr[0].legend(ncol=2,loc='upper right')
        axarr[0].set_xlabel('Radius (# of crater radii)')
        
        axarr[1].plot(radii,c.stats['median_shift'],'+',color=COLORS[i]) # Downshifted metric
        axarr[1].plot(c.stats['fit_median_xarr'],c.stats['fit_median_shift'],color=COLORS[i],label=clabel) #Downshifted fit
        axarr[1].set_title('Exponential fit of Median, shifted down')
        axarr[1].legend(ncol=2,loc='upper right')
        axarr[1].set_xlabel('Radius (# of crater radii)')
    
    m_fig.set_figheight(20)
    m_fig.set_figwidth(15)
    axarr[0].set_xlim((1, radii[-1]))
#    axarr[0].set_ylim((-0.02,0.09))
#    axarr[1].set_ylim((-0.02,0.08))
    plt.setp([a.get_xticklabels() for a in axarr], visible=True)
    plt.show()    
    
def plot_pct(cIDs, cdict):    
    NPLOTS = 2
    m_fig, axarr = plt.subplots(NPLOTS, sharex=True, num=3)
    CMAP = plt.get_cmap('rainbow')
    COLORS = [CMAP(c) for c in np.linspace(0, 1, len(cIDs))]
    for i, cid in enumerate(cIDs):    
        c = cdict[cid]
        radii = c.stats['radii']
        clabel = c.name
        
        axarr[0].plot(radii,c.stats['pct95'],'-',color=COLORS[i],label=clabel) 
        axarr[0].set_title('95th Percentile (actual)')
        axarr[0].legend(ncol=2,loc='upper right')
        axarr[0].set_xlabel('Radius (# of crater radii)')
        
        axarr[1].plot(radii,c.stats['pct95_shift'],'+',color=COLORS[i]) # Downshifted metric
        axarr[1].plot(c.stats['fit_pct95_xarr'],c.stats['fit_pct95_shift'],color=COLORS[i],label=clabel) #Downshifted fit
        axarr[1].set_title('Exponential fit of 95th Percentile, shifted down')
        axarr[1].legend(ncol=2,loc='upper right')
        axarr[1].set_xlabel('Radius (# of crater radii)')
    
    m_fig.set_figheight(20)
    m_fig.set_figwidth(15)
    axarr[0].set_xlim((1, radii[-1]))
#    axarr[0].set_ylim((-0.02,0.09))
#    axarr[1].set_ylim((-0.02,0.08))
    plt.setp([a.get_xticklabels() for a in axarr], visible=True)
    plt.show()