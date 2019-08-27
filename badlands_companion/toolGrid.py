##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to create regular grids used to build Badlands simulation.
"""

import math
import errno
import pandas
import netCDF4
import numpy as np
import pylab as pl
import pandas as pd
from pylab import *
import numpy.ma as ma
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os, sys, datetime, string
from scipy.interpolate import griddata
from mpl_toolkits.basemap import pyproj
from mpl_toolkits.basemap import Basemap, shiftgrid

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class toolGrid:
    """
    Class for creating Badlands regular grid.
    """

    def __init__(self, llcrnrlon=None, llcrnrlat=None, urcrnrlon=None, urcrnrlat=None):
        """
        Initialization function which takes the bounding box coordinates values
        (longitudes and latitudes). All values are in degrees.

        Parameters
        ----------
        variable : llcrnrlon
            Lower left longitude.

        variable: llcrnrlat
            Lower left latitude.

        variable: urcrnrlon
            Upper right longitude.

        variable: urcrnrlat
            Upper right latitude.

        """
        if llcrnrlon == None:
            raise RuntimeError('Lower left longitude value is required.')
        self.llcrnrlon = llcrnrlon

        if llcrnrlat == None:
            raise RuntimeError('Lower left latitude value is required.')
        self.llcrnrlat = llcrnrlat

        if urcrnrlon == None:
            raise RuntimeError('Upper right longitude value is required.')
        self.urcrnrlon = urcrnrlon

        if urcrnrlat == None:
            raise RuntimeError('Upper right latitude value is required.')
        self.urcrnrlat = urcrnrlat

        self.figsize = None
        self.llspace = None
        self.offset = None
        self.lon = None
        self.lat = None
        self.topo = None
        self.offset = None
        self.x = None
        self.y = None
        self.xi = None
        self.yi = None
        self.zi = None

        return

    def plotEPSG(self, epsg=3857, llspace=1., fsize=(8,8), title = None, saveFig = False, figName = 'epsg'):
        """
        Plot a map using ARCGIS image in the background showing the extend of
        the area to use for Badlands model.

        Parameters
        ----------
        variable : epsg
            EPSG code the default is 3857 which is a global ProjectedCRS grid using spherical
            development of ellipsoidal coordinates. The data is relative to WGS 84/World Mercator.

        variable: llspace
            Spacing between the parallels and meridians for Basemap plotting (in degrees).

        variable: fsize
            Size of the figure to plot.

        variable: title
            Title of the plot.

        variable: saveFig
            Saving the figure (boolean).

        variable: figName
            Name of the saved file.
        """

        plt.figure(figsize = fsize)
        self.llspace = llspace
        # Create the map with basemap
        epsgMap = Basemap(epsg = epsg, resolution = 'h', \
                          llcrnrlon = self.llcrnrlon,    \
                          llcrnrlat = self.llcrnrlat,    \
                          urcrnrlon = self.urcrnrlon,    \
                          urcrnrlat = self.urcrnrlat)

        # Load ARCGIS image
        epsgMap.arcgisimage()

        # Draw meridians and parallels
        epsgMap.drawmeridians( np.arange(int(self.llcrnrlon)-self.llspace, \
                              int(self.urcrnrlon)+1+self.llspace,self.llspace), \
                              labels=[0,0,0,1], color='w', fontsize=8 )
        epsgMap.drawparallels( np.arange(int(self.llcrnrlat)-self.llspace, \
                              int(self.urcrnrlat)+1+self.llspace,self.llspace), \
                              labels=[1,0,0,0], color='w', fontsize=8 )

        epsgMap.drawcoastlines( linewidth=2, color='0.5' )
        #epsgMap.drawcountries()

        titlepos = plt.title(str(title), fontsize = 10, weight='bold')
        titlepos.set_y( 1.01 )

        if saveFig:
            plotfile = str(figName)+'.pdf'
            plt.savefig(plotfile,dpi=150,orientation='portrait')
            print('PDF figure saved: ',plotfile)

        plt.show()

        return

    def _roundup(self, x, y):
        '''
        Rounding a value x to the closest value using increment y
        '''

        return int(math.ceil(x / float(y)) * y )

    def _laplace_X(self, F, M):
        '''
        Laplace filter along the X axis.
        '''

        jmax, imax = F.shape

        # Add strips of land.
        F2 = np.zeros((jmax, imax + 2), dtype=F.dtype)
        F2[:, 1:-1] = F
        M2 = np.zeros((jmax, imax + 2), dtype=M.dtype)
        M2[:, 1:-1] = M

        MS = M2[:, 2:] + M2[:, :-2]
        FS = F2[:, 2:] * M2[:, 2:] + F2[:, :-2] * M2[:, :-2]

        return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)

    def _laplace_Y(self, F, M):
        '''
        Laplace filter along the Y axis.
        '''

        jmax, imax = F.shape

        # Add strips of land.
        F2 = np.zeros((jmax + 2, imax), dtype=F.dtype)
        F2[1:-1, :] = F
        M2 = np.zeros((jmax + 2, imax), dtype=M.dtype)
        M2[1:-1, :] = M

        MS = M2[2:, :] + M2[:-2, :]
        FS = F2[2:, :] * M2[2:, :] + F2[:-2, :] * M2[:-2, :]

        return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)

    def _laplace_filter(self, F, M=None):
        '''
        Laplace filter used to perform smoothing of the topography.
        '''

        if not M:
            M = np.ones_like(F)

        return 0.5 * (self._laplace_X(self._laplace_Y(F, M), M) +
                      self._laplace_Y(self._laplace_X(F, M), M))

    def _LevelColormap(self, levels, cmap=None):
        '''
        Make a colormap based on an increasing sequence of levels.
        '''

        if cmap == None:
            cmap = pl.get_cmap()

        nlev = len(levels)
        S = pl.arange(nlev, dtype='float')/(nlev-1)
        A = cmap(S)

        levels = pl.array(levels, dtype='float')
        L = (levels-levels[0])/(levels[-1]-levels[0])

        R = [(L[i], A[i,0], A[i,0]) for i in range(nlev)]
        G = [(L[i], A[i,1], A[i,1]) for i in range(nlev)]
        B = [(L[i], A[i,2], A[i,2]) for i in range(nlev)]
        cdict = dict(red=tuple(R),green=tuple(G),blue=tuple(B))

        return matplotlib.colors.LinearSegmentedColormap(
            '%s_levels' % cmap.name, cdict, 256)

    def _get_indices(self, lons, lats):
        '''
        Find indices of a subset of the global data file.
        '''

        min_lat = self.llcrnrlat - self.offset
        max_lat = self.urcrnrlat + self.offset
        min_lon = self.llcrnrlon - self.offset
        max_lon = self.urcrnrlon + self.offset,

        distances1, distances2, indices = [], [], []
        index = 1
        for point in lats:
            s1 = max_lat - point
            s2 = min_lat - point
            distances1.append((np.dot(s1, s1), point, index))
            distances2.append((np.dot(s2, s2), point, index - 1))
            index = index + 1
        distances1.sort()
        distances2.sort()
        indices.append(distances1[0])
        indices.append(distances2[0])

        distances1, distances2 = [], []
        index = 1
        for point in lons:
            s1 = max_lon - point
            s2 = min_lon - point
            distances1.append((np.dot(s1, s1), point, index))
            distances2.append((np.dot(s2, s2), point, index - 1))
            index = index + 1
        distances1.sort()
        distances2.sort()
        indices.append(distances1[0])
        indices.append(distances2[0])

        res = np.zeros((4), dtype=np.float64)
        res[0] = indices[3][2]
        res[1] = indices[2][2]
        res[2] = indices[1][2]
        res[3] = indices[0][2]

        return res

    def getSubset(self, tfile = 'etopo1', offset = 0.1, smooth = False):
        '''
        Load the global dataset using THREDDS protocol and extract a subset of the global data.
        At the moment the only available dataset is etopo1.

        Parameters
        ----------
        variable : tfile
            Type of topographic / bathymetric file to load using THREDDS
            protocol (only etopo1 for now).

        variable: offset
            The offset to add to the grid to ensure the entire region of interest is still within
            the simulation area after reprojection in UTM coordinates.

        variable: smooth
            Use Laplace filter to smooth the topography (boolean).

        '''

        self.offset = offset

        if tfile == 'etopo1':
            #thredds = 'http://www.ngdc.noaa.gov/thredds/dodsC/relief/ETOPO1/thredds/ETOPO1_Bed_g_gmt4.nc'
            #thredds = 'http://dl.tpac.org.au/thredds/dodsC/bathymetry/ETOPO/etopo1/ETOPO1_Bed_g_gmt4.nc'
            thredds = 'http://thredds.socib.es/thredds/dodsC/ancillary_data/bathymetry/ETOPO1_Bed_g_gmt4.nc'
            #thredds = 'http://opendap.deltares.nl/thredds/dodsC/opendap/deltares/delftdashboard/bathymetry/etopo1/etopo1.nc'
        else:
            raise RuntimeError('ETOPO1 is the only dataset available for now.')

        etopo = Dataset(thredds, 'r')
        lons = etopo.variables["x"][:]
        lats = etopo.variables["y"][:]

        res = self._get_indices(lons, lats)

        self.lon, self.lat = np.meshgrid(lons[int(res[0]):int(res[1])], lats[int(res[2]):int(res[3])])
        self.topo = etopo.variables["z"][int(res[2]):int(res[3]),int(res[0]):int(res[1])]

        if smooth:
            self.topo = self._laplace_filter(self.topo, M=None)

        return

    def mapUTM(self, contour=10, fsize=(8,8), saveFig=False, nameFig='map'):
        """
        Convertion of lon/lat map to UTM coordinates and plotting of the converted regular grid.

        Parameters
        ----------
        variable : contour
            Elevation difference between contour lines on the plot (in metres).

        variable: fsize
            Size of the figure to plot.

        variable: title
            Title of the plot.

        variable: saveFig
            Saving the figure (boolean).

        variable: nameFig
            Name of the saved file.
        """

        fig = plt.figure(figsize = fsize)

        lonMin = self.lon.min() + self.offset
        latMin = self.lat.min() + self.offset
        lonMax = self.lon.max() - self.offset
        latMax = self.lat.max() - self.offset

        if lonMin < 0 and lonMax < 0:
            lon_0 = -(abs( lonMax ) + abs( lonMin )) / 2.0
        else:
            lon_0 = (abs( lonMax ) + abs( lonMin )) / 2.0

        topomin = int(self.topo.min())
        topomax = int(self.topo.max())

        if topomin < 0 and topomax > 0:
            topomin = -self._roundup(-topomin, contour)
            levels1 = np.arange(topomin, 0, contour)
            topomax = self._roundup(topomax, contour)
            levels2 = np.arange(0, topomax+contour, contour)

        elif topomin > 0 :
            topomin = self._roundup(topomin, contour)
            levels1 = np.arange(topomin, topomax+contour, contour)

        elif topomax < 0 :
            topomax = -self._roundup(-topomax, contour)
            levels1 = np.arange(topomin, topomax+contour, contour)

        map = Basemap(llcrnrlat = latMin, urcrnrlat = latMax,\
                llcrnrlon = lonMin, urcrnrlon = lonMax,\
                resolution = 'h', area_thresh = 10., projection = 'merc',\
                lat_1 = latMin, lon_0 = lon_0)

        self.x, self.y = map(self.lon, self.lat)
        map.drawmeridians( np.arange(lonMin,lonMax,self.llspace), labels = [0,0,0,1], fontsize = 8)
        map.drawparallels( np.arange(latMin,latMax,self.llspace), labels = [1,0,0,0], fontsize = 8)

        if topomin < 0 and topomax > 0:
            CS1 = map.contourf(self.x, self.y, self.topo, levels1,
                               cmap = self._LevelColormap(levels1, cmap = cm.Blues_r),
                               extend = 'neither',
                               alpha = 1.0,
                               origin='lower')
            CS2 = map.contourf(self.x, self.y, self.topo, levels2,
                               cmap = self._LevelColormap(levels2,cmap = cm.RdYlGn_r),
                               extend = 'neither',
                               alpha = 1.0,
                               origin = 'lower')
            CS1.axis = 'tight'
            CS2.axis = 'tight'
            CB1 = plt.colorbar(CS1, fraction = 0.03, shrink = 0.95, pad = -0.23)
            CB2 = plt.colorbar(CS2, fraction = 0.03, shrink = 0.95, pad = 0.05, orientation = 'horizontal')
            CB2.set_label('topography [m]',fontsize = 8)
            CB1.set_label('bathymetry [m]', fontsize = 8)
            CB1.ax.tick_params(labelsize = 8)
            CB2.ax.tick_params(labelsize = 8)
            titlepos = plt.title('ETOPO1 Map', fontsize=10, weight='bold')
            titlepos.set_y(1.02)

        elif topomin > 0 :
            CS1 = map.contourf(self.x, self.y, self.topo, levels1,
                               cmap = self._LevelColormap(levels1, cmap = cm.RdYlGn_r),
                               extend = 'neither',
                               alpha = 1.0,
                               origin='lower')
            CS1.axis = 'tight'
            CB1 = plt.colorbar(CS1, fraction = 0.03, shrink = 0.95) #, pad = 0.23)
            CB1.set_label('topography [m]',fontsize = 8)
            CB1.ax.tick_params(labelsize = 8)
            titlepos = plt.title('ETOPO1 Map', fontsize=10, weight='bold')
            titlepos.set_y(1.02)

        elif topomax < 0 :
            CS1 = map.contourf(self.x, self.y, self.topo, levels1,
                               cmap = self._LevelColormap(levels1, cmap = cm.Blues_r),
                               extend = 'neither',
                               alpha = 1.0,
                               origin='lower')
            CS1.axis = 'tight'
            CB1 = plt.colorbar(CS1, fraction = 0.03, shrink = 0.95) #, pad = -0.23)
            CB1.set_label('bathymetry [m]', fontsize = 8)
            CB1.ax.tick_params(labelsize = 8)
            titlepos = plt.title('UTM etopo1 Map', fontsize=10, weight='bold')
            titlepos.set_y(1.02)

        if saveFig:
            plotfile = str(nameFig)+'.pdf'
            plt.savefig(plotfile, dpi = 150, orientation = 'portrait')
            print('PDF figure saved: ',plotfile)

        plt.show()

        return

    def buildGrid(self, resolution=250., method='cubic', nameCSV='xyz'):
        """
        Convertion of lon/lat map to UTM coordinates and plotting of the converted regular grid.

        Parameters
        ----------
        variable : resolution
            Required resolution for the model grid (in metres).

        variable: method
            Method of interpolation. One of {'linear', 'nearest', 'cubic'}.

        variable: nameCSV
            Name of the saved CSV topographic file.
        """
        xscat = self.x.flatten()
        yscat = self.y.flatten()
        zscat = self.topo.flatten()

        xgrid = np.arange(int(self.x.min()),int(self.x.max()),resolution,dtype=np.float)
        ygrid = np.arange(int(self.y.min()),int(self.y.max()),resolution)
        self.xi, self.yi = np.meshgrid(xgrid, ygrid)

        self.zi = griddata((xscat,yscat), zscat, (self.xi, self.yi), method=str(method))

        df = pd.DataFrame({'X':self.xi.flatten(),'Y':self.yi.flatten(),'Z':self.zi.flatten()})
        df.to_csv(str(nameCSV)+'.csv',columns=['X', 'Y', 'Z'], sep=' ', index=False ,header=0)

        return


    def viewGrid(self, width = 800, height = 800, zmin = None, zmax = None, title='Export Grid'):
        """
        Use Plotly library to visualise the grid in 3D.

        Parameters
        ----------
        variable : resolution
            Required resolution for the model grid (in metres).

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: zmin
            Minimal elevation.

        variable: zmax
            Maximal elevation.

        variable: height
            Figure height.

        variable: title
            Title of the graph.
        """

        if zmin == None:
            zmin = self.zi.min()

        if zmax == None:
            zmax = self.zi.max()

        data = [ Surface( x=self.xi[0,:], y=self.yi[:,0], z=self.zi, colorscale='Earth')]

        layout = Layout(
            title = title,
            autosize=True,
            width=width,
            height=height,
            scene=dict(
                zaxis=dict(range=[zmin, zmax],autorange=False,nticks=10, \
                            gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                xaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2, \
                            zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                yaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2, \
                            zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                bgcolor="rgb(244, 244, 248)"
            )
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return
