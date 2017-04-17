#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:24:07 2016

@author: christian
"""
import ace_settings 
import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from timeit import default_timer as timer
import csv
import gdal
import pickle

import plotting_functions as pf
import statistical_functions as sf
from scipy.stats import gmean

class CraterDict(dict):
    """
    Dictionary of {crater ID: Crater}. 
    
    Can be populated from a properly formatted csv of craters as follows 
    (parentheses indicate optional columns): 
    |name|diam|lat|lon|(age)|(loc)|(omatnum)|(craterlist)|
    """
    def __init__(self, fname='', delimiter=','):
        if fname:
            cdict, cIDs= self._fromCSV(fname, delimiter)
            dict.__init__(self, cdict)
            self.cIDs = cIDs # list of crater names in original order in f
        else:
            dict.__init__(self, {})
            self.cIDs = []
        # Acerim sublists        
        self.aceIDs = []
        self.oobIDs = []
        self.noexpIDs = []
        self.nonaceIDs = []
        # Verified sublists
        self.vrayIDs = []
        self.vbrightIDs = []
        self.vdarkIDs = []
        self.vanomIDs = []


    def __add__(self, other):
        """Adds all entries with unique keys from other CraterDict to self."""
        if self.isace or self.isver or other.isace or other.isver:
            raise Exception('Cannot combine ace or verified dicts yet')
        import copy
        newCdict = copy.copy(self)
        for craterID in other:
            if craterID in self:
                print('Cannot add duplicate: {} == {}'.format(self[craterID].name, 
                                                         other[craterID].name))
            else:
                newCdict[craterID] = other[craterID]
                newCdict.cIDs.append(craterID)      
        return newCdict
        

    def _fromCSV(self, file_name, delimiter):
        """
        Return a dict of Crater populated with data from csv specified by file_name.
        
        Preconditions:
            All files must be present in the directory given in file_path.
            Data files must be in .csv format with the 1st row as column titles.
            Diameter stored in [km].
        Returns:
            cdict: dictionary of craters
            keys: list of string of column headings from the data file(s)
    
        """
        cdict = {}
        cIDs = []
        f = file_name
        with open(f, 'r', errors='ignore') as cf:
            if cf is None:
                raise ImportError('Error Opening crater file')
            headings = next(cf).split(',')
            csvreader = csv.reader(cf)
            for line in csvreader:
                cname = line.pop(0)
                cdata = [float(val) for val in line[:3]]
                diam,lat,lon = cdata[:3] # Depends on order of columns from file
                age = ''
                loc = ''
                try:
                    age = float(line[3])
                    loc = line[4]
                except:
                    Exception
                  
                cID = self._newID(lat,lon)
                if cID in self:
                    print('Cannot add duplicate: ' + cname)
                else: # Create Crater and add to dict
                    crater = Crater(cname, lat, lon, diam, loc, age)  
                    cdict[cID] = crater 
                    cIDs.append(cID)
        return cdict, cIDs
    
        
    def _newID(self, lat, lon):
        """
        Unique crater IDs used for the keys of CraterDict objects
        
        ID consists of crater lat and lon to 2 decimal places (sufficient for 
        craters >= 500m apart). Will exclude overlapping craters)
        """
        return 'C{0:.2f}N{1:.2f}E'.format(lat, lon)
        
    
    def isace(self):
        """Return True if self is an aceDict"""
        if len(self.cIDs) == len(self.aceIDs+self.oobIDs+self.noexpIDs+self.nonaceIDs):
            return True
        else:
            return False
            
    def isver(self):
        """Return True if self is a verified CraterDict"""
        if len(self.aceIDs) == len(self.vrayIDs+self.vbrightIDs+self.vdarkIDs+self.vanomIDs):
            return True
        else:
            return False
            
            
    def getID(self, name):
        ID = [cID for cID in self.cIDs if self[cID].name == name]
        if ID:    
            return ID[0]
        else:
            return ''
        
        
    def getNames(self, cIDs):
        """Return the names corresponding to input cIDs"""
        return [self[cID].name for cID in cIDs]


    def getSubdict(self, cIDs=[], drange=()):
        """
        Return a CraterDict containing only cIDS. Specify drange=(min_diam,max_diam) 
        to give a range of diameters to use to subset the dict. Drange takes
        precedence over supplied cIDs so only one or the other should be given
        to avoid confusion
        
        Go through each attribute and addthe relevant cIDs to corresponding 
        attribute of subdict. Raise KeyError if cid not in self.
        """
        if drange:
            cIDs = [cid for cid in self.cIDs if (self[cid].diam >= drange[0]) 
                        & (self[cid].diam < drange[1])]
        if isinstance(cIDs, str):
            cIDs = [cIDs]
        attrs = vars(self)
        subdict = CraterDict()
        for attr in attrs:
            subvalues = [val for val in attrs[attr] if val in cIDs]
            setattr(subdict, attr, subvalues)
        for cID in cIDs:
            subdict[cID] = self[cID]
        return subdict            

    
    def save(self, fname='temp'):
        """Save CraterDict to fname"""
        saveCdict(self, fname)
    
    
    def acerim(self, aceDS):
        """Oh bbbboy"""
        if self.isace():
            print('This is already an aceDict')
            return
        else: #Setup
            import ace_settings as s
            s.init()        
            StartTime = timer(); NewTime = timer()
            cIDs_todo = [cid for cid in self.cIDs if cid not in 
                        self.aceIDs+self.oobIDs+self.noexpIDs+self.nonaceIDs]
            n = len(cIDs_todo)
        print('\nBegin acerim {} craters\n'.format(n))
        for i, cID in enumerate(cIDs_todo):
            c = self[cID]
            inBounds = hf.getMetrics(c, aceDS, s.PLOT_ROI)
            if not inBounds:
                self.oobIDs.append(cID)
                continue
            fitToExp = hf.getFits(c,cID,s.METRICS)
            if not fitToExp:
                self.noexpIDs.append(cID)
                continue
            c.stats['ACEDOM'] = hf.getAcedomain(c, s.EXPDERIV_THLD)
            c.stats['ACERNG'] = hf.getAcerange(c, 0)
            if (c.stats['ACEDOM'] > s.ACEDOM_THLD) and (c.stats['ACERNG'] > s.ACERNG_THLD):
                self.aceIDs.append(cID)
            # Save dict and print update if SAVE_TIME interval has passed
            ElapsedTime  = timer() - NewTime
            if  ElapsedTime > s.ACESAVE_TIME:
                NewTime = timer()
                print('Finished crater {} out of {} ({}% completed)'.format(
                        i+1, n, 100*(i+1)/n))
                print('Last Crater: {}, diam={}'.format(c.name,c.diam))
                print('Time Elapsed: {}:{}:{}'.format(int(ElapsedTime//3600),
                      int((ElapsedTime%3600)//60), round((ElapsedTime%3600)%60,2)))
                self.save(s.CPATH+'ace_ipr')
        #Save Final aceDict
        TotalTime = timer() - StartTime
        print('\nFinished acerim in {}:{}:{}'.format(int(TotalTime//3600),
                      int((TotalTime%3600)//60), round((TotalTime%3600)%60,2)))
        print('TEST isace: ',self.isace())
        self.save(s.CPATH+s.ACESAVE_FILE)
     
     
    def compare(self, aceDSlist, cIDs=[], rmax = 0, SAVEFIG=False, figfolder='/figs'):
        """Show side-by-side comparison for ROI around cIDs in all the datasets 
        in aceDSlist"""
        import ace_settings as s
        s.init() 
        if not rmax:
            rmax = s.RMAX
        if not (isinstance(aceDSlist,list) or isinstance(aceDSlist,tuple)):
            aceDSlist = [aceDSlist]
        if not cIDs:
            cIDs = self.cIDs
        elif isinstance(cIDs,str):
            cIDs = [cIDs]
        for i,cID in enumerate(cIDs):
            c = self[cID]
            fig, axarr = plt.subplots(nrows=len(aceDSlist), sharex=False, sharey=True, dpi=s.DPI)
            for i,aceDS in enumerate(aceDSlist):
                try:
                    roi = aceDS.getROI(c, rmax) 
                except ImportError as e:
                    print('Could not import this region of '+aceDS.name)
                    roi = None
                else:
                     im = axarr[i].imshow(roi, cmap=aceDS.cmap, extent=c.extent, 
                                            vmin=aceDS.pltmin, vmax=aceDS.pltmax)
                     axarr[i].set_title(['OMAT','RA'][i],size=s.FS*1.5)
                     axarr[i].set_xlabel('Longitude (degrees)',size=s.FS)
                     axarr[i].set_ylabel('Latitude (degrees)',size=s.FS)
                     cax,kw = cbar.make_axes(axarr[i])
                     plt.colorbar(im, cax=cax, **kw)               
                plt.setp([a.get_xticklabels() for a in axarr], visible=True)
                fig.suptitle('"{}"\n Diameter = {:2}km. RA age = {}Mya'.format(c.name, c.diam, c.age),size=s.FS*1.75)
                plt.rcParams.update({'xtick.labelsize':'small', 'ytick.labelsize':'small'})
                fig.set_figwidth(s.FIGSIZE[0])                
                fig.set_figheight(s.FIGSIZE[1])   
                if SAVEFIG:
                    figname = cID+'.png'            
                    plt.savefig(s.CPATH+figfolder+figname)   
        plt.show()
    
    def cfd(self, mode='OMAT', region='fullmoon', fig=0, label='', mark='-', NPF=False):
        """Plot the crater frequency distribution. Specficy what region of the 
        Moon and the mode to get the lat extents"""
        latcorr = 1 # Factor to scale area by if ds only in a band around equator        
        if mode == 'OMAT':
            latcorr = np.cos((90-50)*np.pi/180) 
        elif mode == 'RA':
            latcorr = np.cos((90-80)*np.pi/180) 
        diams,sizefreq = sf.getSfd(self,self.cIDs,latcorr,region)
        # Plot cfd
        if not fig:
            plt.figure()
        else:
            plt.figure(fig)
#        if norm:
#            sizefreq = sizefreq/len(sizefreq)
        plt.plot(diams, sizefreq, mark, label=label+', N='+str(len(sizefreq)),fillstyle='none')         
        
        if NPF:
            a = ace_settings.a_cfd
            D = np.logspace(-2,np.log10(300))  
            j = np.arange(len(a))
            N = []
            for i in range(len(D)):
                N.append(np.sum(a*(np.log10(D[i])**j)))
            N = 10**(np.array(N))
            plt.plot(D,N,label='NPF, Neukum 2001')
        
        plt.legend()
        plt.title('Cumulative Size Distribution')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('diameter ($km$)')
        plt.ylabel('frequency ($km^{-2}yr^{-1}$)')  
            
        
    def rfd(self,mode='OMAT',region='fullmoon',fig=0,label='',mark='-',NPF=False):
        """Plot the relative frequency distribution as described in "Standard
        Techniques for Presentation and Analysis of Crater Size-Frequency Data"
        """
        import ace_settings as s
        RMOON = s.RMOON #km
        if mode == 'OMAT':
            latcorr = np.cos((90-50)*np.pi/180) 
        elif mode == 'RA':
            latcorr = np.cos((90-80)*np.pi/180) 
#        A = latcorr*4*np.pi*(RMOON**2)
        diams, sizefreq = sf.getSfd(self,self.cIDs,latcorr,region)
        dSfd = np.abs(np.gradient(sizefreq))
#        diams = np.array([self[cID].diam for cID in self.cIDs])
#        nbins = 100 #np.logspace(np.log10(diams[0],np.log10(diams[-1])))
#        #hist, bins = np.histogram(diams, bins=bins)  
#        bins = np.linspace(diams[0], diams[-1], nbins)
#        Dbar, R = np.zeros(nbins), np.zeros(nbins)
#        for i in range(nbins-1):
#            bin_diams = diams[(bins[i] <= diams) & (diams <= bins[i+1])]
#            Dbar[i] = gmean(bin_diams) #geometric mean
#            R[i] = (Dbar[i]**3)*len(bin_diams)/(A*(bins[i+1]-bins[i]))
        if not fig:
            plt.figure()
        else:
            plt.figure(fig)
#        plt.plot(Dbar, R, mark, label=label+', N='+str(len(diams)),fillstyle='none')        
        r = diams**3 *dSfd
        plt.plot(diams, r, mark, label=label+', N='+str(len(sizefreq)),fillstyle='none')
        
        if NPF:
            a = ace_settings.a_cfd
            n = len(a)
            D = np.logspace(-1,np.log10(150))  
            j = np.arange(n)
            logN, logdN = [], []
            for i in range(len(D)):
                logN.append(np.sum(a*(np.log10(D[i])**j)))
                logdN.append(np.sum(a[1:n]*(np.log10(D[i])**j[0:n-1])))
            N = 10**(np.array(logN))
            dN = np.abs(np.array(logdN))        
            dNdD = (N/D) * dN
            Dbar = np.sqrt(D[:-1]*D[1:])
            R = (Dbar**3) * dNdD
            plt.plot(Dbar,R,label='NPF, Neukum 2001')
#            
#        if NPF: # Add Neukum production function(R = D^3 * dN/dD)
#            from ace_settings import neuk_a
#            neuk_D = np.logspace(np.log10(diams[0]),np.log10(diams[-1]))
#            neuk_N = np.zeros(len(neuk_D)) # initialize
#            neuk_dN = np.zeros(len(neuk_D))
#            for i in range(len(neuk_D)):
#                N, dN = 0, 0
#                for j in range(12):
#                    N += neuk_a[j]*(np.log10(neuk_D[i]))**j
#                for j in range(1,12):
#                    dN += neuk_a[j]*(np.log10(neuk_D[i]))**j-1
#                neuk_N[i] = np.exp(N) 
#                neuk_dN[i] = np.exp(dN)
#            dNdD = (neuk_N/neuk_D)*neuk_dN
#            plt.plot(neuk_D, dNdD, label='Neukum 2001')

        plt.yscale('log')
        plt.xscale('log') 
        plt.legend(loc='best')
        plt.title('Relative Size Distribution')
        plt.xlabel('D (km)')
        plt.ylabel('R')
            #        
#            neuk_a = s.neuk_a
#            neuk_D = np.log10(np.logspace(np.log10(diams[0]),np.log10(diams[-1]),num=len(neuk_a)))
#            N = len(neuk_D)
#            neuk_x = neuk_D**np.arange(N) #x**i where x =log(D)
##            neuk_R = np.zeros(N) # initialize
#            for i in range(N):
#                D2 = -neuk_D**2
#                p10 = 10**(neuk_a*neuk_x)
#                p = np.arange(1,N)*neuk_a[1:N]*neuk_x[:-1]
#                
#            R = D2*np.sum(p10)*np.sum(p)
#            plt.plot(10**neuk_D,R,label='Neukum 1983')
#        
        
    def verify(self, aceDSlist):
        """Verify the aceIDs in this dict."""
        def vsave(fname):
            """Append cIDs to appropriate attributes then save using self.save()"""
            self.vrayIDs += [cID for i,cID in enumerate(cIDs_todo) if vraybool[i]]
            self.vbrightIDs += [cID for i,cID in enumerate(cIDs_todo) if vbrightbool[i]]
            self.vdarkIDs += [cID for i,cID in enumerate(cIDs_todo) if vdarkbool[i]]
            self.vanomIDs += [cID for i,cID in enumerate(cIDs_todo) if vanombool[i]]
            self.save(fname)
        if not self.aceIDs:
            print('There are no aceIDs to verify')
            return
        if self.isver():
            print('This is already a verified CraterDict')
            return
        else: #Setup
            import ace_settings as s
            s.init()   
            if not (isinstance(aceDSlist,list) or isinstance(aceDSlist,tuple)):
                aceDSlist = [aceDSlist]
            StartTime = timer(); NewTime = timer()
            cIDs_todo = [cid for cid in self.aceIDs if cid not in 
                        self.vrayIDs+self.vbrightIDs+self.vdarkIDs+self.vanomIDs]
            i = 0
            n = len(cIDs_todo) 
            vraybool = [False]*n
            vbrightbool = [False]*n
            vdarkbool = [False]*n
            vanombool = [False]*n
        print('\nBegin verify {} craters\n'.format(n))
        while i < n:
            cID = cIDs_todo[i]
            print('Crater {}/{}'.format(i,n))  
            # Display crater ROI
            try:
                if len(aceDSlist) == 1:
                    c = self[cID]
                    aceDSlist[0].getROI(c, s.RMAX, PLOT=True)  
                else:
                    self.compare(aceDSlist, cID)
            except ImportError as e:
                print(str(e))
                print('Out of bounds. Placing crater in anomIDs')
                self.vanomIDs.append(cID)
                continue
            # Get user input crater designation
            inpt = '' 
            while not inpt:
                inpt = input('Rays/Bright/Dark:   (1/2/3)  \
                            \nAnomalous:          (0)      \
                            \nGo (b)ack 1 crater: (b)      \
                            \nSave & (q)uit:      (q) \n')
                if inpt == '1':
                    vraybool[i] = True
                elif inpt == '2':
                    vbrightbool[i] = True
                elif inpt == '3':
                    vdarkbool[i] = True
                elif inpt == '0':
                    vanombool[i] = True
                elif inpt == 'b' and i > 0:
                    vraybool[i-1] = False
                    vbrightbool[i-1] = False
                    vdarkbool[i-1] = False
                    vanombool[i-1] = False
                    i -= 2
                elif inpt == 'q':
                    vsave(s.CPATH+'ver_ipr')
                    return
                else:
                    print('INVALID INPUT')
                    inpt = ''  
            # Save dict time interval is passed
            ElapsedTime  = timer() - NewTime
            if  ElapsedTime > s.VERSAVE_TIME:
                NewTime = timer()
                print('Saving verified dict...')
                vsave(s.CPATH+'ace_ipr')
                print('Time Elapsed: {}:{}:{}'.format(int(ElapsedTime//3600),
                      int((ElapsedTime%3600)//60), round((ElapsedTime%3600)%60,2)))
            i += 1
        #Save Final verifiedDict
        TotalTime = timer() - StartTime
        print('\nFinished acerim in {}:{}:{}'.format(int(TotalTime//3600),
                      int((TotalTime%3600)//60), round((TotalTime%3600)%60,2)))                   
        vsave(s.CPATH+s.VERSAVE_FILE) 
        print('TEST isver: ',self.isver()) 
                
            
def saveCdict(cdict, fname):
    """
    Save dict to fname
    """
    if fname[-4:] != '.pkl':
        fname += '.pkl'
    with open(fname,'wb') as f:
        try:
            pickle.dump(cdict, f, pickle.HIGHEST_PROTOCOL)
        except:
            print('Error saving Cdict. Try restarting kernal and reimporting')
        finally:
            print('Saved to '+fname)
    
    
def loadCdict(fname):
    if fname[-4:] != '.pkl':
        fname += '.pkl'
    try:
        with open(fname,'rb') as f:
            return pickle.load(f)
    except EOFError:
        return CraterDict()

	
	
class AceDict(CraterDict):
	
	def __init__(self, cdict, aceds):
		self= cdict
		self.moon_frac = hf.getMoonfrac(aceds.nlat, aceds.slat,aceds.wlon, aceds.elon) 
		
	
    
class Crater(object):
    """
    Stores raw crater data and later calculated metrics.
    """  
    def __init__(self, name, lat, lon, diam, loc='', age="nan"):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.diam = diam
        self.rad = 1000*diam/2
        self.loc = loc
        self.age = age
        self.stats = {}
        self.extent = ()
        

    def __str__(self):
        return self.name
        

    def __repr__(self):
        return 'Crater {} at ({:.2f}N,{:.2f}E)'.format(self.name, self.lat, self.lon)
        


class AceDataset(object):
    """
    A reference to a Dataset object (using gdal.Dataset) which also contains the
    extent and resolution info as well as plotting defaults.
    """
    def __init__(self, MODE):
        ace_settings.init_ds(MODE)   
        self.mode = MODE
        self.path = ace_settings.PATH
        self.dsname = ace_settings.DSNAME
        self.ds = gdal.Open(self.path+self.dsname)
        if self.ds is None:
            raise ImportError('File not found or import failed')
        self.name = ace_settings.DSNAME
        self.nlat = ace_settings.NLAT
        self.slat = ace_settings.SLAT
        self.wlon = ace_settings.WLON
        self.elon = ace_settings.ELON          
        self.ppd = ace_settings.PPD
        self.mpp = ace_settings.MPP
        self.pltmin = ace_settings.PLTMIN
        self.pltmax = ace_settings.PLTMAX
        self.cmap = ace_settings.CMAP
        self.lonarr = np.linspace(self.wlon, self.elon, self.ds.RasterXSize)
        self.latarr = np.linspace(self.nlat, self.slat, self.ds.RasterYSize)
        
        
    def __str__(self):
        return self.dsname
        
        
    def __repr__(self):
        return 'AceDataset {}'.format(self.dsname)
    
    
    def getROI(self, c, max_radius, PLOT=False):
        """
        Return square ROI centered on crater c which extends max_radius crater 
        radii from the crater center. 
        
        If the the lon extent of the dataset is crossed, use wrap_lon(). 
        If the lat extent is crossed, raise error.
    
        Arguments:
        ----------
        c: Crater
            Current crater containing lat, lon, radius.
        max_radius: float
            max radial extent of the ROI from center of crater. Amounts to half
            of the length/width of returned ROI
            
        Returns:
        --------
        roi: 2Darray
            The specified window of data from dataset.
        """
        def wrap_lon(self):
            """
            Extract an roi that crosses the dataset lon boundary by concatenating
            the part on the left side of boundary with the part on the right side.
            """
            if minlon < self.wlon: 
                low_lonsize = self.wlon - minlon
                low_xind = hf.getInd(minlon,self.lonarr-360)
                low_xsize = hf.deg2pix(low_lonsize, self.ppd) 
                low_roi = self.ds.ReadAsArray(low_xind, yind, low_xsize, ysize)
                
                high_lonsize = maxlon - self.wlon
                high_xind = hf.getInd(self.wlon,self.lonarr)
                high_xsize = hf.deg2pix(high_lonsize, self.ppd) 
                high_roi = self.ds.ReadAsArray(high_xind, yind, high_xsize, ysize)
                          
            elif maxlon > self.elon:
                low_lonsize = self.elon - minlon
                low_xind = hf.getInd(minlon,self.lonarr)
                low_xsize = hf.deg2pix(low_lonsize, self.ppd) 
                low_roi = self.ds.ReadAsArray(low_xind, yind, low_xsize, ysize)
                
                high_lonsize = maxlon - self.elon
                high_xind = hf.getInd(self.elon,self.lonarr+360)
                high_xsize = hf.deg2pix(high_lonsize, self.ppd) 
                high_roi = self.ds.ReadAsArray(high_xind, yind, high_xsize, ysize)               
            return np.concatenate((low_roi, high_roi), axis=1)  
                
        # If crater lon out of bounds, adjust to this ds [(0,360) <-> (-180,180)]
        if c.lon > self.elon: 
            c.lon -= 360
        if c.lon < self.wlon:
            c.lon += 360
            
        latsize = 2*hf.m2deg(max_radius*c.rad, self.mpp, self.ppd)
        lonsize = 2*hf.m2deg(max_radius*c.rad, self.mpp, self.ppd)
        minlat = c.lat-latsize/2
        maxlat = c.lat+latsize/2
        minlon = c.lon-lonsize/2
        maxlon = c.lon+lonsize/2
        c.extent = (minlon, maxlon, minlat, maxlat)
        # Throw error if window bounds are not in lat bounds.
        if minlat < self.slat or maxlat > self.nlat:
            raise ImportError('Latitude ({0},{1}) out of bounds ({2},{3}) '.format(
                                       minlat, maxlat, self.slat, self.nlat))

        yind = hf.getInd(maxlat,self.latarr) # get top index of ROI
        ysize = hf.deg2pix(latsize, self.ppd) 
        if minlon < self.wlon or maxlon > self.elon:
            roi = wrap_lon(self,)  
        else: 
            xind = hf.getInd(minlon,self.lonarr) # get left index of ROI
            xsize = hf.deg2pix(lonsize, self.ppd)
            roi = self.ds.ReadAsArray(xind, yind, xsize, ysize) # read ROI subarray
        
        if roi is None:
            raise ImportError('GDAL could not read dataset into array')
       
        if PLOT:
            self.plot_roi(roi, c.extent, c.name, c.diam)    
        return roi 
    
    
    def plot_roi(self, roi, extent=(), cname='',cdiam='?'):
        """
        Plot roi 2D array. 
        
        If extent, cname and cdiam are supplied, the axes will display the 
        lats and lons specified and title will inclue cname and cdiam.
        """
        plt.figure("ROI",figsize=(8,8))
        plt.imshow(roi, extent=extent, cmap=self.cmap, vmin=self.pltmin, vmax=self.pltmax)
        plt.title('Crater {}, Diam {}km'.format(cname, cdiam))
        plt.xlabel('Longitude (degrees)')
        plt.ylabel('Latitude (degrees)')
        plt.show()
    
