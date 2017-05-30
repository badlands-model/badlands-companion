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
from cmocean import cm
import colorlover as cl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

import plotly
from plotly import tools
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def readSea(seafile):
    """
    Plot sea level curve.
    Parameters
    ----------
    variable: seafile
    Absolute path of the sea-lelve data.
    """

    df=pd.read_csv(seafile, sep=r'\s+',header=None)
    SLtime,sealevel = df[0],df[1]

    return SLtime,sealevel

def viewData(x0 = None, y0 = None, width = 800, height = 400, linesize = 3, color = '#6666FF'):
    """
    Plot multiple data on a graph.
    Parameters
    ----------
    variable: x0, y0
    Data for plot 
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: linesize
        Requested size for the line.
    """
    trace = Scatter(
        x=x0,
        y=y0,
        mode='lines',
        line=dict(
            shape='line',
            color = color,
            width = linesize
        ),
        fill=None
    )
        
    layout = dict(
            width=width,
            height=height
            )
        
    fig = Figure(data=[trace], layout=layout)
    plotly.offline.iplot(fig)

    return

def buildShore(cs = None, cs_b = None, sealevel = None, sealevel_b = None):
    """
    Calculate the shoreline trajectory (shoreID and shoreElev), the change of accommodation (accom) 
    and sedimentation (sed) at shoreline, the end point of each depostional layer (depoend), 
    and sediment flux (sedflux) through time.
    Parameters
    ----------
    variable: cs, cs_b
    The cross-section at time t and previous timestep (t-dt)
    variable: sealevel, sealevel_b
    The value of sea-level at time t and previous timestep (t-dt)
    """
    shoreID = np.amax(np.where(cs.secDep[cs.nz-1]>=sealevel)[0])
    shoreElev = cs.secDep[cs.nz-1][shoreID]
    shoreID_b = np.amax(np.where(cs_b.secDep[cs_b.nz-1]>=sealevel_b)[0])
    accom = sealevel - cs.secDep[cs_b.nz-1][shoreID_b]
    sed = cs.secDep[cs.nz-1][shoreID_b] - cs.secDep[cs_b.nz-1][shoreID_b]
    depoend = np.amax(np.where(cs.secTh[cs.nz-1][shoreID_b:len(cs.secTh[0])]>0.001)[0]) + shoreID_b
    sedflux = 0.
    for i in range(cs_b.nz-1,cs.nz):
        sedflux = sedflux + sum(cs.secTh[i][shoreID:,])

    return shoreID, shoreElev, accom, sed, depoend, sedflux

def waterDepth(cs = None, envIDs = None):
    """
    Calculate the position of different depositional environments.
    Parameters
    ----------
    variable: envIDs
    range of water depth of each depostional environment.
    """
    waterdep = []
    IPs = np.zeros((cs.nz, len(envIDs)))
    for i in range(0,cs.nz):
        for j in range(len(envIDs)):
            waterdep.append(-cs.secElev[i])
            IPs[i][j] = np.amin(np.where(waterdep[i]>envIDs[j])[0])

    return IPs

def viewSection(width = 800, height = 400, cs = None, dnlay = None, 
                rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
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
    variable: dnlay
        Layer step to plot the cross-section.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)
    #colorrgb = []
    #cmap=cm.delta
    #for n in range(nlay):
        #colorrgb.append(cmap(n/float(nlay)))
    colors = cl.scales['9']['div']['BrBG']
    hist = cl.interp( colors, nlay )
    colorrgb = cl.to_rgb( hist )

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secDep[i],
            mode='lines',
            line=dict(
                shape='line',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colorrgb[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secDep[nlay-1],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colorrgb[nlay-1]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

def viewSection_Depth(cs = None, IPs = None, dnlay = None, color = None, 
                      rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot stratal stacking pattern colored by water depth.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    fig = plt.figure(figsize = (9,5))
    plt.rc("font", size=10)

    ax = fig.add_subplot(111)
    layID = []
    p = 0
    xi00 = cs.dist
    # color = ['limegreen','sandybrown','khaki','lightsage','c','dodgerblue']
    for i in range(0,cs.nz,dnlay):
        if i == cs.nz:
            i = i-1
        layID.append(i)
        if len(layID) > 1:
            for j in range(0,600):
                if (j<IPs[i][0]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[0])
                elif (j<IPs[i][1]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[0])
                elif (j<IPs[i][2]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[1])
                elif (j<IPs[i][3]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[2])
                elif (j<IPs[i][4]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[3])
                elif (j<IPs[i][5]):
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[4])
                else:
                    plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j], cs.secDep[layID[p-1]][j+1]], color=color[5])
                    plt.fill_between(xi00, strat.secDep[layID[p]], 0, color='white')
        p=p+1
    for i in range(0,cs.nz,dnlay):
        if i>0:
            plt.plot(xi00,cs.secDep[i],'-',color='k',linewidth=0.2)
    plt.plot(xi00,cs.secDep[cs.nz-1],'-',color='k',linewidth=0.7)
    plt.plot(xi00,cs.secDep[0],'-',color='k',linewidth=0.7)
    plt.xlim( rangeX ) 
    plt.ylim( rangeY )

    return

def viewSectionST(width = 800, height = 400, cs = None, dnlay = None, colors=None, 
                  rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections colored by system tracts on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: colors
        System tract color scale.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secDep[i],
            mode='lines',
            line=dict(
                shape='line',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colors[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secDep[nlay-1],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colors[nlay-1]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

def viewWheeler(width = 800, height = 400, cs = None, time = None, colors = None,
                    rangeE=None, rangeX = None, rangeY = None, contourdx = 50,
                    title = 'Wheeler diagram'):
    """
    Plot wheeler diagram colored by deposition environment on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: time
        Simulation time array dataset.
    variable: colors
        Depositional environments color scale.
    variable: rangeE
        Depositional environments extent.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: contourdx
        Distance between 2 contours.
    variable: title
        Title of the graph.
    """

    smoothelev = gaussian_filter(cs.secElev, sigma=2)
    emin = rangeE[0][0]
    emax = rangeE[0][1]
    for k in range(len(colors)):
        tmp1 = rangeE[k][0]
        tmp2 = rangeE[k][1]
        emin = min(emin,tmp1)
        emax = max(emax,tmp2)
    colorscale = []

    for k in range(len(colors)):
        tmp = (rangeE[k][0]-emin)/(emax-emin)
        list = [tmp,colors[k]]
        colorscale.append(list)
    colorscale.append([1., 'rgb(250,250,250)'])

    data = Data([
        Contour(
            x=cs.dist,
            y=time,
            z=smoothelev,
            line=dict(smoothing=0.8,color='black' ,width=0.5),
            colorscale=colorscale,
            autocontour=False,
            contours=dict(
                start=emin,
                end=emax,
                size=contourdx,
            ),
            colorbar=dict(
                thickness=25,
                thicknessmode='pixels',
                title='Elevation [m]',
                titleside='right',
                len=0.9,
                lenmode='fraction',
                outlinewidth=0
            )
        )
    ])

    layout = dict(title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='Time [year]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
    )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

class stratalSection:
    """
    Class for creating stratigraphic cross-sections from Badlands outputs.
    """

    def __init__(self, folder=None, ncpus=1):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus
        if ncpus > 1:
            raise RuntimeError('Multi-processors function not implemented yet!')

        self.x = None
        self.y = None
        self.xi = None
        self.yi = None
        self.dx = None
        self.dist = None
        self.dx = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dep = None
        self.th = None
        self.elev = None
        self.xsec = None
        self.ysec = None
        self.secTh = []
        self.secDep = []
        self.secElev = []
        self.shoreID = []
        self.shoreAccom = []
        self.shoreSed = []
        self.depoEnd = []

        return

    def loadStratigraphy(self, timestep=0):
        """
        Read the HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Time step to load.
        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/sed.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
            #print(list(df.keys()))
            coords = np.array((df['/coords']))
            layDepth = np.array((df['/layDepth']))
            layElev = np.array((df['/layElev']))
            layThick = np.array((df['/layThick']))
            if i == 0:
                x, y = np.hsplit(coords, 2)
                dep = layDepth
                elev = layElev
                th = layThick

        self.dx = x[1]-x[0]
        self.x = x
        self.y = y
        self.nx = int((x.max() - x.min())/self.dx+1)
        self.ny = int((y.max() - y.min())/self.dx+1)
        self.nz = dep.shape[1]
        self.xi = np.linspace(x.min(), x.max(), self.nx)
        self.yi = np.linspace(y.min(), y.max(), self.ny)
        self.dep = dep.reshape((self.ny,self.nx,self.nz))
        self.elev = elev.reshape((self.ny,self.nx,self.nz))
        self.th = th.reshape((self.ny,self.nx,self.nz))

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

        return xsec, ysec

    def buildSection(self, xo = None, yo = None, xm = None, ym = None,
                    pts = 100, gfilter = 5):
        """
        Extract a slice from the 3D data set and compute the stratigraphic layers.
        Parameters
        ----------
        variable: xo, yo
            Lower X,Y coordinates of the cross-section.
        variable: xm, ym
            Upper X,Y coordinates of the cross-section.
        variable: pts
            Number of points to discretise the cross-section.
        variable: gfilter
            Gaussian smoothing filter.
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
        self.xsec = xsec
        self.ysec = ysec
        for k in range(self.nz):
            # Thick
            rect_B_spline = RectBivariateSpline(self.yi, self.xi, self.th[:,:,k])
            data = rect_B_spline.ev(ysec, xsec)
            secTh = filters.gaussian_filter1d(data,sigma=gfilter)
            secTh[secTh < 0] = 0
            self.secTh.append(secTh)

            # Elev
            rect_B_spline1 = RectBivariateSpline(self.yi, self.xi, self.elev[:,:,k])
            data1 = rect_B_spline1.ev(ysec, xsec)
            secElev = filters.gaussian_filter1d(data1,sigma=gfilter)
            self.secElev.append(secElev)

            # Depth
            rect_B_spline2 = RectBivariateSpline(self.yi, self.xi, self.dep[:,:,k])
            data2 = rect_B_spline2.ev(ysec, xsec)
            secDep = filters.gaussian_filter1d(data2,sigma=gfilter)
            self.secDep.append(secDep)

        # Ensure the spline interpolation does not create underlying layers above upper ones
        topsec = self.secDep[self.nz-1]
        for k in range(self.nz-2,-1,-1):
            secDep = self.secDep[k]
            self.secDep[k] = np.minimum(secDep, topsec)
            topsec = self.secDep[k]

        return 