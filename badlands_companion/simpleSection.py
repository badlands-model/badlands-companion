##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to build simple cross-section from Badlands outputs.
"""

import os
import math
import h5py
import errno
import numpy as np
import pandas as pd
import colorlover as cl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def viewSea(seafile):
    """
    Plot sea level curve.
    """

    df=pd.read_csv(seafile, sep=r'\s+',header=None)
    SLtime,sealevel = df[0],df[1]

    fig = plt.figure(figsize = (4,8))
    plt.rc("font", size=10)

    ax1 = fig.add_subplot(1, 1, 1)
    minZ = SLtime.min()
    maxZ = SLtime.max()
    minX = sealevel.min()
    maxX = sealevel.max()

    plt.plot(sealevel,SLtime,'-',color='#6666FF',linewidth=2)

    axes = plt.gca()
    plt.xlim( minX-10, maxX+10 )
    plt.ylim( minZ, maxZ )
    plt.grid(True)
    plt.xlabel('Sea level (m)',fontsize=11)
    plt.ylabel('Time (years)',fontsize=11)

    return SLtime,sealevel

def viewSections(width = 800, height = 400, cs = None, layNb = 2,
                    linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections data on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: layNb
        Number of layer to plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """

    colors = cl.scales['9']['div']['RdYlBu']
    hist = cl.interp( colors, layNb )
    colorrgb = cl.to_rgb( hist )

    nlay = layNb - 1

    lay = {}
    toplay = cs[nlay].top
    cs[nlay].ndepo = cs[nlay].depo
    cs[nlay].ntop = toplay

    botlay = cs[nlay].top - cs[nlay].depo

    for i in range(nlay-1,-1,-1):
        tmp1 = cs[i+1].ndepo - cs[i].depo
        tmp2 = cs[i+1].ndepo - tmp1.clip(min=0)
        cs[i].ndepo = tmp2
        cs[i].ntop = botlay + tmp2

    trace = {}
    data = []

    for i in range(1,nlay):
        trace[i-1] = Scatter(
            x=cs[i].dist,
            y=cs[i].ntop,
            mode='lines',
            name="layer "+str(i),
            line=dict(
                shape='spline',
                width = linesize-1,
                color = colorrgb[i-1] #,
                #dash = 'dash'
            )
        )
        data.append(trace[i-1])

    # Top line
    trace[nlay] = Scatter(
        x=cs[nlay].dist,
        y=cs[nlay].top,
        mode='lines',
        name="top",
        line=dict(
            shape='spline',
            width = linesize,
            color = 'rgb(102, 102, 102)'
        )
    )
    data.append(trace[nlay])

    # Bottom line
    trace[nlay+1] = Scatter(
        x=cs[nlay].dist,
        y=cs[nlay].top - cs[nlay].depo,
        mode='lines',
        name="base",
        line=dict(
            shape='spline',
            width = linesize,
            color = 'rgb(102, 102, 102)'
        )
    )
    data.append(trace[nlay+1])

    layout = dict(
            title=title,
            width=width,
            height=height
    )

    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

class simpleSection:
    """
    Class for creating simple cross-sections from Badlands outputs.
    """

    def __init__(self, folder=None, bbox=None, dx=None):
        """
        Initialization function which takes the folder path to Badlands outputs. It also takes the
        bounding box and discretization value at which one wants to interpolate the data.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: bbox
            Bounding box extent SW corner and NE corner.
        variable: dx
            Discretisation value in metres.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.x = None
        self.y = None
        self.z = None
        self.cumchange = None
        self.top = None
        self.depo = None
        self.dist = None
        self.dx = None
        self.nx = None
        self.ny = None
        self.ndepo = None
        self.ntop = None

        if dx == None:
            raise RuntimeError('Discretization space value is required.')
        self.dx = dx
        self.bbox = bbox

        return

    def loadHDF5(self, timestep=0):
        """
        Read the HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Time step to load.
        """

        df = h5py.File('%s/tin.time%s.hdf5'%(self.folder, timestep), 'r')
        coords = np.array((df['/coords']))
        cumdiff = np.array((df['/cumdiff']))
        x, y, z = np.hsplit(coords, 3)
        c = cumdiff

        if self.bbox == None:
            self.nx = int((x.max() - x.min())/self.dx+1)
            self.ny = int((y.max() - y.min())/self.dx+1)
            self.x = np.linspace(x.min(), x.max(), self.nx)
            self.y = np.linspace(y.min(), y.max(), self.ny)
            self.bbox = np.zeros(4,dtype=float)
            self.bbox[0] = x.min()
            self.bbox[1] = y.min()
            self.bbox[2] = x.max()
            self.bbox[3] = y.max()
        else:
            if self.bbox[0] < x.min():
                self.bbox[0] = x.min()
            if self.bbox[2] > x.max():
                self.bbox[2] = x.max()
            if self.bbox[1] < y.min():
                self.bbox[1] = y.min()
            if self.bbox[3] > y.max():
                self.bbox[3] = y.max()
            self.nx = int((self.bbox[2] - self.bbox[0])/self.dx+1)
            self.ny = int((self.bbox[3] - self.bbox[1])/self.dx+1)
            self.x = np.linspace(self.bbox[0], self.bbox[2], self.nx)
            self.y = np.linspace(self.bbox[1], self.bbox[3], self.ny)

        self.x, self.y = np.meshgrid(self.x, self.y)
        xyi = np.dstack([self.x.flatten(), self.y.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        z_vals = z[indices][:,:,0]
        zi = np.average(z_vals,weights=(1./distances), axis=1)
        c_vals = c[indices][:,:,0]
        ci = np.average(c_vals,weights=(1./distances), axis=1)

        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            ci[onIDs] = c[indices[onIDs,0]]

        self.z = np.reshape(zi,(self.ny,self.nx))
        self.cumchange = np.reshape(ci,(self.ny,self.nx))

        return

    def _cross_section(self, xo, yo, xm, ym, pts):
        """
        Compute cross section coordinates.
        """

        if xm == xo:
            ysec = np.linspace(yo, ym, pts)
            xsec = np.zeros(pts)
            xsec.fill(xo)
        elif ym == yo:
            xsec = np.linspace(xo, xm, pts)
            ysec = np.zeros(pts)
            ysec.fill(yo)
        else:
            a = (ym-yo)/(xm-xo)
            b = yo - a * xo
            xsec = np.linspace(xo, xm, pts)
            ysec = a * xsec + b

        return xsec,ysec

    def getEroDep(self, xo = None, yo = None, xm = None, ym = None,
                    pts = 100, gfilter = 5):
        """
        Extract a slice from the 3D data set and compute its deposition thicknesses.
        Parameters
        ----------
        variable: xo, yo
            Lower X,Y coordinates of the cross-section
        variable: xm, ym
            Upper X,Y coordinates of the cross-section
        variable: pts
            Number of points to discretise the cross-section
        variable: gfilter
            Gaussian smoothing filter
        """

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()


        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        self.dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)

        # Surface
        rect_B_spline = RectBivariateSpline(self.y[:,0], self.x[0,:], self.z)
        datatop = rect_B_spline.ev(ysec, xsec)
        self.top = filters.gaussian_filter1d(datatop,sigma=gfilter)

        # Cumchange
        rect_B_spline = RectBivariateSpline(self.y[:,0], self.x[0,:], self.cumchange)
        cumdat = rect_B_spline.ev(ysec, xsec)
        gcum = filters.gaussian_filter1d(cumdat,sigma=gfilter)
        self.depo = gcum.clip(min=0)

        return

    def view1Section(self, width = 800, height = 400,
                    linesize = 3, title = 'Cross section'):
        """
        Plot cross-section data on a graph.
        Parameters
        ----------
        variable: width
            Figure width.
        variable: height
            Figure height.
        variable: linesize
            Requested size for the line.
        variable: title
            Title of the graph.
        """
        trace0 = Scatter(
            x=self.dist,
            y=self.top - self.depo + 2000,
            mode='lines',
            name="'bottom'",
            line=dict(
                shape='spline',
                #color = color,
                width = linesize
            ),
            fill=None
        )

        trace1 = Scatter(
            x=self.dist,
            y=self.top+2000,
            mode='lines',
            name="'top'",
            line=dict(
                shape='spline',
                #color = color,
                width = linesize
            ),
            fill='tonexty'
        )

        data = [trace0, trace1]

        layout = dict(
            title=title,
            width=width,
            height=height
            )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return
