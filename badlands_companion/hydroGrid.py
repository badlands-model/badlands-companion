##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse hydrometrics from Badlands outputs.
"""

import os
import math
import h5py
import errno
import pandas
import numpy as np
from matplotlib import cm
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ETO
from scipy import interpolate
from scipy import spatial

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class tributaries:
    """
    Class for storing tributaries.
    """

    def __init__(self):
        """
        Initialization function for the tributary class.
        """
        self.streamX = None
        self.streamY = None
        self.streamZ = None
        self.streamFA = None
        self.streamChi = None
        self.streamPe = None
        self.streamLght = None
        self.dist = None
        self.Zdata = None
        self.FAdata = None
        self.Chidata = None
        self.Pedata = None
        self.maintribID = None

        return

class hydroGrid:
    """
    Class for analysing hydrometrics from Badlands outputs.
    """

    def __init__(self, folder=None, ptXY=None):
        """
        Initialization function which takes the folder path to Badlands outputs.
        It also takes a point coordinates contained in a catchment of interest.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.

        variable: ptXY
            X-Y coordinates of the point contained within the catchment.

        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.Z = None
        self.FA = None
        self.Chi = None
        self.ptXY = ptXY
        self.donX = None
        self.donY = None
        self.rcvX = None
        self.rcvY = None

        self.streamX = None
        self.streamY = None
        self.streamZ = None
        self.streamPe = None
        self.streamFA = None
        self.streamChi = None
        self.streamLght = None

        self.closeXY = None
        self.dist = None
        self.Zdata = None
        self.Pedata = None
        self.FAdata = None
        self.Chidata = None

        return

    def getCatchment(self, timestep=0):
        """
        Read the HDF5 file for a given time step and extract the catchment.

        Parameters
        ----------
        variable : timestep
            Time step to load.

        """

        df = h5py.File('%s/flow.time%s.hdf5'%(self.folder, timestep), 'r')
        vertices = np.array((df['/coords']))
        lbasin = np.array((df['/basin']))
        lfacc = np.array((df['/discharge']))
        lconnect = np.array((df['/connect']))
        lchi = np.array((df['/chi']))
        con1, con2 = np.hsplit(lconnect, 2)
        conIDs = np.append(con1, con2)
        IDs = np.unique(conIDs-1)

        Xl = np.zeros((len(lconnect[:,0]),2))
        Yl = np.zeros((len(lconnect[:,0]),2))
        Xl[:,0] = vertices[lconnect[:,0]-1,0]
        Xl[:,1] = vertices[lconnect[:,1]-1,0]
        Yl[:,0] = vertices[lconnect[:,0]-1,1]
        Yl[:,1] = vertices[lconnect[:,1]-1,1]
        Zl = vertices[lconnect[:,1]-1,2]
        FAl = lfacc[lconnect[:,0]-1]
        Chil = lchi[lconnect[:,0]-1]
        Basinl = lbasin[lconnect[:,0]-1]

        X1 = Xl[:,0]
        X2 = Xl[:,1]
        Y1 = Yl[:,0]
        Y2 = Yl[:,1]
        Z = Zl
        FA = FAl
        Chi = Chil
        Basin = Basinl

        self.XY = np.column_stack((X1, Y1))
        if self.ptXY[0] < X1.min() or self.ptXY[0] > X1.max():
            raise RuntimeError('X coordinate of given point is not in the simulation area.')

        if self.ptXY[1] < Y1.min() or self.ptXY[1] > Y1.max():
            raise RuntimeError('Y coordinate of given point is not in the simulation area.')

        distance,index = spatial.KDTree(self.XY).query(self.ptXY)
        self.basinID = Basin[index]
        basinIDs = np.where(Basin == Basin[index])[0]
        self.Basin = Basin

        self.donX = X1[basinIDs]
        self.donY = Y1[basinIDs]
        self.rcvX = X2[basinIDs]
        self.rcvY = Y2[basinIDs]
        self.Z = Z[basinIDs]
        self.FA = FA[basinIDs]
        self.Chi = Chi[basinIDs]

        return

    def extractMainstream(self):
        """
        Extract main stream for the considered catchment based on flow
        discharge.

        """

        IDs = np.where(self.FA == self.FA.max())[0]
        outlet = IDs[0]

        streamIDs = np.zeros(len(self.FA),dtype=int)
        streamIDs[0] = int(outlet)
        up = int(outlet)
        k = 1
        loop = True
        xnext,ynext = self.donX[up],self.donY[up]

        while (loop):
            n1 = np.where(np.logical_and(abs(self.donX - xnext) < 1.,
                                         abs(self.donY - ynext) < 1))[0]
            n2 = np.where(np.logical_and(abs(self.rcvX - xnext) < 1.,
                                         abs(self.rcvY - ynext) < 1))[0]
            n = np.hstack((n1, n2))
            if len(n) > 0 :
                nAll = np.unique(n)
                idx = np.where(np.logical_and(nAll != up,
                                              nAll != streamIDs[k-1]))[0]
                if len(idx) > 0:
                    nIDs =  nAll[idx]
                    next = nIDs[np.argmax(self.FA[nIDs])]
                    if k > 1 and (int(next) == streamIDs[k-1] or \
                        int(next) == streamIDs[k-2]):
                        break
                    streamIDs[k] = int(next)
                    up = int(next)
                    k += 1
                    xnext,ynext = self.donX[up],self.donY[up]
                else:
                    loop = False
            else:
                loop = False

        id = np.where(streamIDs > 0)[0]
        self.streamX = self.donX[streamIDs[id]]
        self.streamY = self.donY[streamIDs[id]]
        self.streamZ = self.Z[streamIDs[id]]
        self.streamFA = self.FA[streamIDs[id]]
        self.streamChi = self.Chi[streamIDs[id]]

        return

    def computeParams(self, kd=None, kc=None, m=None, n=None, num=500):
        """
        Computes the Peclet and cumulative main stream lenght.

        Parameters
        ----------
        variable : kd
            Hillslope diffusion coefficient.

        variable : kc
            Erodibility coefficient.

        variable : m
            Coefficient m of stream power law.

        variable : n
            Coefficient n of stream power law.

        variable : num
            Number of samples to generate.
        """

        lenght = 0.
        self.streamPe = np.zeros(len(self.streamX))
        self.streamLght = np.zeros(len(self.streamX))

        for p in range(1,len(self.streamX)):

            dx = self.streamX[p] - self.streamX[p-1]
            dy = self.streamY[p] - self.streamY[p-1]
            lenght = lenght + np.sqrt(dx**2 + dy**2)

            self.streamLght[p] = lenght
            self.streamPe[p] = kc * (self.streamLght[p]**(2*(m + 1) - n)) \
                / (kd * self.streamZ[p]**(1-n))

        self.streamFA = self.streamFA.reshape(len(self.streamFA))
        self.streamChi = self.streamChi.reshape(len(self.streamChi))

        # Interpolation functions
        self.dist = np.linspace(0., self.streamLght.max(), num=num,
                           endpoint=True)
        pecletFunc = interpolate.interp1d(self.streamLght,
                                           self.streamPe, kind='linear')
        faFunc = interpolate.interp1d(self.streamLght,
                                       self.streamFA, kind='linear')
        chiFunc = interpolate.interp1d(self.streamLght,
                                        self.streamChi, kind='linear')
        zFunc = interpolate.interp1d(self.streamLght,
                                      self.streamZ-self.streamZ[0], kind='linear')

        # Get data
        self.Zdata = zFunc(self.dist)
        self.FAdata = faFunc(self.dist)
        self.Chidata = chiFunc(self.dist)
        self.Pedata = pecletFunc(self.dist)

        return

    def computeTribParams(self, tribList=None, kd=None, kc=None, m=None, n=None, num=500):
        """
        Computes the Peclet and cumulative tributaries stream lenght.

        Parameters
        ----------
        variable : tribList
            All tributary dataset.

        variable : kd
            Hillslope diffusion coefficient.

        variable : kc
            Erodibility coefficient.

        variable : m
            Coefficient m of stream power law.

        variable : n
            Coefficient n of stream power law.

        variable : num
            Number of samples to generate.
        """
        maxlght = 0.
        for t in range(len(tribList)):

            trib = tribList[t]
            lenght = 0.
            trib.streamPe = np.zeros(len(trib.streamX))
            trib.streamLght = np.zeros(len(trib.streamX))

            if len(trib.streamX) > 1:
                for p in range(1,len(trib.streamX)):

                    dx = trib.streamX[p] - trib.streamX[p-1]
                    dy = trib.streamY[p] - trib.streamY[p-1]
                    lenght = lenght + np.sqrt(dx**2 + dy**2)

                    trib.streamLght[p] = lenght
                    trib.streamPe[p] = kc * (trib.streamLght[p]**(2*(m + 1) - n)) \
                        / (kd * trib.streamZ[p]**(1-n))

                trib.streamFA = trib.streamFA.reshape(len(trib.streamFA))
                trib.streamChi = trib.streamChi.reshape(len(trib.streamChi))

                # Interpolation functions
                trib.dist = np.linspace(0., trib.streamLght.max(), num=num,
                                   endpoint=True)
                if maxlght < trib.streamLght.max():
                    self.maintribID = t
                    maxlght = trib.streamLght.max()

                pecletFunc = interpolate.interp1d(trib.streamLght,
                                                   trib.streamPe, kind='linear')
                faFunc = interpolate.interp1d(trib.streamLght,
                                               trib.streamFA, kind='linear')
                chiFunc = interpolate.interp1d(trib.streamLght,
                                                trib.streamChi, kind='linear')
                zFunc = interpolate.interp1d(trib.streamLght,
                                              trib.streamZ-trib.streamZ[0], kind='linear')

                # Get data
                trib.Zdata = zFunc(trib.dist)
                trib.FAdata = faFunc(trib.dist)
                trib.Chidata = chiFunc(trib.dist)
                trib.Pedata = pecletFunc(trib.dist)

        return

    def _colorCS(self, color):
        """
        Compute color scale values from matplotlib colormap.
        """
        colorCS=[]
        for k in  range(256):
            r,g,b,e = color(k)
            colorCS.append([ k/255., '#%02x%02x%02x'%(int(r*255+0.5),
                                                      int(g*255+0.5),
                                                      int(b*255+0.5))])
        return colorCS

    def findTribTop(self, donor = None, rcv = None):
        """
        Find the top of each tributaries in the catchment.

        Parameters
        ----------

        variable: donor
            Donor points coordinates

        variable: rcv
            Receiver points coordinates

        Return
        --------

        variable: uniqueRcv
            List of node IDs for the top of each tributary.
        """

        # Combine donors and receivers
        mergeDR = np.vstack((donor,rcv))
        mDR = np.copy(mergeDR)
        ddr = mDR.astype(int)

        # Get the IDs of the top of each tributaries stream
        uniqueRcv = []
        for i in range(len(ddr)):
            xx = ddr[i,0]
            yy = ddr[i,1]
            id = np.where(np.logical_and(ddr[:,0]==xx, ddr[:,1]==yy))[0]
            if len(id) == 1:
                uniqueRcv.append(id[0])

        return uniqueRcv

    def extractAllStreams(self, startIDs = None):
        """
        Extract all tributaries in the catchment.

        Parameters
        ----------

        variable: startIDs
            List of node IDs for the top of each tributary.

        Return
        --------

        variable: streamlist
            List of all streams in the catchment
        """

        streamlist = []

        # Start from the top and recursively store tributaries connectivity

        for i in range(len(startIDs)):
            trib = tributaries()
            streamIDs = np.zeros(len(self.FA),dtype=int)
            down = int(startIDs[i])
            streamIDs[0] = down
            k = 1
            append = True
            if down>len(self.rcvX):
                loop = False
                append = False
            else:
                loop = True
                xnext,ynext = self.rcvX[down],self.rcvY[down]

            while (loop):
                n2 = np.where(np.logical_and(abs(self.donX - xnext) < 1.,
                                             abs(self.donY - ynext) < 1))[0]
                n1 = np.where(np.logical_and(abs(self.rcvX - xnext) < 1.,
                                             abs(self.rcvY - ynext) < 1))[0]
                n = np.hstack((n1, n2))
                if len(n) > 0 :
                    nAll = np.unique(n)
                    idx = np.where(np.logical_and(nAll != down,
                                   nAll != streamIDs[k-1]))[0]
                    if len(idx) > 0:
                        nIDs =  nAll[idx]
                        next = nIDs[np.argmax(self.FA[nIDs])]
                        if k > 1 and (int(next) == streamIDs[k-1] or \
                            int(next) == streamIDs[k-2]):
                            break
                        streamIDs[k] = int(next)
                        down = int(next)
                        if down>len(self.rcvX):
                            loop = False
                        else:
                            k += 1
                            xnext,ynext = self.rcvX[down],self.rcvY[down]
                    else:
                        loop = False
                else:
                    loop = False

            id = np.where(streamIDs > 0)[0]

            # Reverse order to store data from outlet to top
            if append :
                stIDs = streamIDs[id][::-1]
                trib.streamX = self.donX[stIDs]
                trib.streamY = self.donY[stIDs]
                trib.streamZ = self.Z[stIDs]
                trib.streamFA = self.FA[stIDs]
                trib.streamChi = self.Chi[stIDs]

                streamlist.append(trib)

        return streamlist

    def limitAllstreams(self,ptXY,streamlist):
        """
        Limit tributaries based on initial point position.

        Parameters
        ----------

        variable: ptXY
            Position of the point from where you want to extract the streams.

        variable: streamlist
            List of all streams in the catchment

        Return
        --------

        variable: limitStream
            List of limit streams in the catchment
        """

        distance,index = spatial.KDTree(self.XY).query(ptXY)
        if self.Basin[index] != self.basinID:
            print('The given point is not in the previously extracted basin.')
            print('You will need to modify your point coordinates.')
            return

        fXY = self.XY[index]
        limitStream = []

        for i in range(len(streamlist)):
            found = np.where(np.logical_and(streamlist[i].streamX == fXY[0],
                             streamlist[i].streamY == fXY[1]))[0]
            if len(found) > 0:
                selectIDs = np.where(streamlist[i].streamZ >= streamlist[i].streamZ[found])[0]
                ntrib = tributaries()
                ntrib.streamX = streamlist[i].streamX[selectIDs]
                ntrib.streamY = streamlist[i].streamY[selectIDs]
                ntrib.streamZ = streamlist[i].streamZ[selectIDs]-streamlist[i].streamZ[selectIDs].min()
                ntrib.streamFA = streamlist[i].streamFA[selectIDs]
                ntrib.streamChi = streamlist[i].streamChi[selectIDs]-streamlist[i].streamChi[selectIDs].min()
                limitStream.append(ntrib)

        return limitStream

    def viewNetwork(self, markerPlot = True, linePlot = False, lineWidth = 3, markerSize = 15, \
                   val = 'chi', width = 800, height = 400, colorMap = None, \
                   colorScale = None, title = '<br>Stream network graph'):
        """
        Visualise stream network.

        Parameters
        ----------

        variable: markerPlot
            Boolean to plot markers

        variable: linePlot
            Boolean to plot lines

        variable: lineWidth
            Line width

        variable: markerSize
            Size of markers

        variable: val
            Name of the dataset to plot: 'chi', 'FA', 'Z'

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: colorMap
            Matplotlib color map for the lines.

        variable: colorScale
            Color scale name for the marker.

        variable: title
            Title of the graph.
        """

        traces = []
        if val == 'chi':
            valData = self.Chi
        elif val == 'FA':
            IDo = np.where(self.FA == 0.)[0]
            self.FA[IDo] = 1.
            valData = self.FA
        elif val == 'Z':
            valData = self.Z
        else:
            raise RuntimeError('The requested data is unknown, options are: chi, Fa, Z.')

        valmax = valData.max()
        colorCS = self._colorCS(colorMap)

        if linePlot:
            for nn in range(0, len(self.FA)):
                traces.append(
                    Scatter(
                        x=[self.donX[nn], self.rcvX[nn], None],
                        y=[self.donY[nn], self.rcvY[nn], None],
                        line=dict(
                            width=lineWidth,
                            color = min(colorCS,key=lambda x: abs(float(x[0]) - valData[nn]/valmax))[1]
                        ),
                        hoverinfo='none',
                        mode='lines'
                    )
                )
        else:
            for nn in range(0, len(self.FA)):
                traces.append(
                    Scatter(
                        x=[self.donX[nn], self.rcvX[nn], None],
                        y=[self.donY[nn], self.rcvY[nn], None],
                        line=dict(
                            width=lineWidth,
                            color='#888'
                        ),
                        hoverinfo='none',
                        mode='lines'
                    )
                )

        if markerPlot:
            traces.append(
                Scatter(
                    x=self.donX,
                    y=self.donY,
                    text=[],
                    mode='markers',
                    hoverinfo='text',
                    name=val,
                    marker=dict(
                        showscale=True,
                        colorscale=colorScale,
                        color=valData,
                        size=markerSize,
                        colorbar=dict(
                            thickness=15,
                            title=val,
                            xanchor='left',
                            titleside='right'
                        ),
                        opacity = 0.8,
                        line=dict(width=0.2)
                    )
                )
             )

        fig = Figure(data=traces,
                  layout=Layout(
                      title=title,
                      titlefont=dict(size=16),
                      showlegend=False,
                      width=width,
                      height=height,
                      hovermode='closest',
                      xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=dict(showgrid=True, zeroline=True, showticklabels=True)
                  )
            )
        plotly.offline.iplot(fig)

        return

    def viewStream(self, linePlot = False, lineWidth = 3, markerSize = 15, \
                   val = 'chi', width = 800, height = 400, colorMap = None, \
                   colorScale = None, title = '<br>Main stream'):
        """
        Visualise main stream.

        Parameters
        ----------

        variable: linePlot
            Boolean to plot lines

        variable: lineWidth
            Line width

        variable: markerSize
            Size of markers

        variable: val
            Name of the dataset to plot: 'chi', 'FA', 'Z'

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: colorMap
            Matplotlib color map for the lines.

        variable: colorScale
            Color scale name for the marker.

        variable: title
            Title of the graph.
        """

        traces = []
        if val == 'chi':
            valData = self.Chi
            valStream = self.streamChi
        elif val == 'FA':
            IDo = np.where(self.FA == 0.)[0]
            self.FA[IDo] = 1.
            valData = self.FA
            IDo = np.where(self.streamFA == 0.)[0]
            self.streamFA[IDo] = 1.
            valStream = self.streamFA
        elif val == 'Z':
            valData = self.Z
            valStream = self.streamZ
        else:
            raise RuntimeError('The requested data is unknown, options are: chi, Fa, Z.')

        valmax = valData.max()
        colorCS = self._colorCS(colorMap)

        if linePlot:
            for nn in range(0, len(self.FA)):
                traces.append(
                    Scatter(
                        x=[self.donX[nn], self.rcvX[nn], None],
                        y=[self.donY[nn], self.rcvY[nn], None],
                        line=dict(
                            width=lineWidth,
                            color = min(colorCS,key=lambda x: abs(float(x[0]) - valData[nn]/valmax))[1]
                        ),
                        hoverinfo='none'
                    )
                )
        else:
            for nn in range(0, len(self.FA)):
                traces.append(
                    Scatter(
                        x=[self.donX[nn], self.rcvX[nn], None],
                        y=[self.donY[nn], self.rcvY[nn], None],
                        line=dict(
                            width=lineWidth,
                            color='#888'
                        ),
                        hoverinfo='none'
                    )
                )

        traces.append(
            Scatter(
                x=self.streamX,
                y=self.streamY,
                text=[],
                mode='markers',
                hoverinfo='text',
                name=val,
                marker=dict(
                    showscale=True,
                    colorscale=colorScale,
                    color=valStream,
                    size=markerSize,
                    colorbar=dict(
                        thickness=15,
                        title=val,
                        xanchor='left',
                        titleside='right'
                    ),
                    opacity = 1.,
                    line=dict(width=0.2)
                )
            )
        )

        fig = Figure(data=traces,
                  layout=Layout(
                      title=title,
                      titlefont=dict(size=16),
                      showlegend=False,
                      width=width,
                      height=height,
                      hovermode='closest',
                      xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=dict(showgrid=True, zeroline=True, showticklabels=True)
                  )
            )
        plotly.offline.iplot(fig)

        return

    def viewPlot(self, lineWidth = 3, markerSize = 5, xval = 'dist', yval = 'chi', \
                   width = 800, height = 500, colorLine = None, colorMarker = None, \
                   opacity = None, title = None):
        """
        Plot flow parameters.

        Parameters
        ----------

        variable: lineWidth
            Line width

        variable: markerSize
            Size of markers

        variable: xval
            Name of the dataset to plot along the X direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: yval
            Name of the dataset to plot along the Y direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: colorLine
            Color for the line.

        variable: colorMarker
            Color for the markers.

        variable: opacity
            Opacity for the marker.

        variable: title
            Title of the graph.
        """

        if xval == 'dist':
            xdata = self.dist
        elif xval == 'FA':
            IDo = np.where(self.FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            xdata = self.FAdata
        elif xval == 'chi':
            xdata = self.Chidata
        elif xval == 'Z':
            xdata = self.Zdata
        elif xval == 'Pe':
            xdata = self.Pedata
        else:
            raise RuntimeError('The requested X value is unknown, options are: dist, chi, Fa, Z, Pe')

        if yval == 'dist':
            ydata = self.dist
        elif yval == 'FA':
            IDo = np.where(self.FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            ydata = self.FAdata
        elif yval == 'chi':
            ydata = self.Chidata
        elif yval == 'Z':
            ydata = self.Zdata
        elif yval == 'Pe':
            ydata = self.Pedata
        else:
            raise RuntimeError('The requested Y value is unknown, options are: dist, chi, Fa, Z, Pe')

        data = [
            Scatter(
                x=xdata,
                y=ydata,
                mode='lines+markers',
                name="'spline'",
                opacity = 1.,
                line=dict(
                    shape='spline',
                    color = colorLine,
                    width = lineWidth
                ),
                marker = dict(
                    symbol='circle',
                    size = markerSize,
                    color = colorMarker,
                    opacity = opacity,
                    line = dict(
                        width = 0.1,
                        color = '#888'
                    )
                )
            )
        ]

        layout = dict(
            title=title,
            width=width,
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return

    def viewTribPlot(self, tribWidth = 2, tribColor = 'rgba(207, 215, 222, 0.5)', \
                     mainWidth = 5, mainColor = 'rgba(233, 109, 196, 1)', \
                     xval = 'dist', yval = 'chi', tribList = None, distLimit = 0, \
                     width = 800, height = 500, title = None):
        """
        Plot main stream and tributary flow parameters.

        Parameters
        ----------

        variable: tribWidth
            Line width for tributaries

        variable: tribColor
            Line color for tributaries

        variable: mainWidth
            Line width for main stream

        variable: mainColor
            Line color for main stream

        variable: xval
            Name of the dataset to plot along the X direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: yval
            Name of the dataset to plot along the Y direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: tribList
            List of all streams in the catchment

        variable: lenLimit
            Limit the number of tributaries to plot based on each tributary length (m)

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: title
            Title of the graph.
        """

        if xval == 'dist':
            xdata = self.dist
        elif xval == 'FA':
            IDo = np.where(self.FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            xdata = self.FAdata
        elif xval == 'chi':
            xdata = self.Chidata
        elif xval == 'Z':
            xdata = self.Zdata
        elif xval == 'Pe':
            xdata = self.Pedata
        else:
            raise RuntimeError('The requested X value is unknown, options are: dist, chi, Fa, Z, Pe')

        if yval == 'dist':
            ydata = self.dist
        elif yval == 'FA':
            IDo = np.where(self.FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            ydata = self.FAdata
        elif yval == 'chi':
            ydata = self.Chidata
        elif yval == 'Z':
            ydata = self.Zdata
        elif yval == 'Pe':
            ydata = self.Pedata
        else:
            raise RuntimeError('The requested Y value is unknown, options are: dist, chi, Fa, Z, Pe')

        traces = []

        for i in range(len(tribList)):
            if len(tribList[i].streamX) > 1 and tribList[i].dist[-1] > distLimit:

                if xval == 'dist':
                    xd = tribList[i].dist
                elif xval == 'FA':
                    IDo = np.where(tribList[i].FAdata == 0.)[0]
                    tribList[i].FAdata[IDo] = 1.
                    xd = tribList[i].FAdata
                elif xval == 'chi':
                    xd = tribList[i].Chidata
                elif xval == 'Z':
                    xd = tribList[i].Zdata
                else:
                    xd = tribList[i].Pedata

                if yval == 'dist':
                    y1 = tribList[i].dist
                elif yval == 'FA':
                    IDo = np.where(tribList[i].FAdata == 0.)[0]
                    self.FAdata[IDo] = 1.
                    y1 = tribList[i].FAdata
                elif yval == 'chi':
                    y1 = tribList[i].Chidata
                elif yval == 'Z':
                    y1 = tribList[i].Zdata
                else:
                    y1 = tribList[i].Pedata

                traces.append(
                    Scatter(
                        x=xd,
                        y=y1,
                        mode='lines',
                        name="'spline'",
                        opacity = 1.,
                        line=dict(
                            shape='spline',
                            color = tribColor,
                            width = tribWidth
                        ),
                    )
                )

        traces.append(
            Scatter(
                x=xdata,
                y=ydata,
                mode='lines',
                name="'spline'",
                opacity = 1.,
                line=dict(
                    shape='spline',
                    color = mainColor,
                    width = mainWidth
                )
            )
        )

        layout = dict(
            showlegend=False,
            title=title,
            width=width,
            height=height,
            hovermode='closest',
            xaxis=XAxis(showgrid=True, zeroline=True, showticklabels=True),
            yaxis=YAxis(showgrid=True, zeroline=True, showticklabels=True)
        )

        fig = Figure(data=traces, layout=layout)
        plotly.offline.iplot(fig)

        return

    def viewTribLimitPlot(self, tribWidth = 2, tribColor = 'rgba(207, 215, 222, 0.5)', \
                     mainWidth = 5, mainColor = 'rgba(233, 109, 196, 1)', \
                     xval = 'dist', yval = 'chi', onlyMain = 0, tribList = None, distLimit = 0, \
                     width = 800, height = 500, title = None):
        """
        Plot main stream and tributary flow parameters.

        Parameters
        ----------

        variable: tribWidth
            Line width for tributaries

        variable: tribColor
            Line color for tributaries

        variable: mainWidth
            Line width for main stream

        variable: mainColor
            Line color for main stream

        variable: xval
            Name of the dataset to plot along the X direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: yval
            Name of the dataset to plot along the Y direction: 'dist', 'chi', 'FA', 'Z', 'Pe'

        variable: onlyMain
            Flag to plot mainstream only if set to 1 otherwise 0 will plot the tributaries as well.

        variable: tribList
            List of all streams in the catchment

        variable: lenLimit
            Limit the number of tributaries to plot based on each tributary length (m)

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: title
            Title of the graph.
        """

        if xval == 'dist':
            xdata = tribList[self.maintribID].dist #self.dist
        elif xval == 'FA':
            IDo = np.where(tribList[self.maintribID].FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            xdata = tribList[self.maintribID].FAdata
        elif xval == 'chi':
            xdata = tribList[self.maintribID].Chidata
        elif xval == 'Z':
            xdata = tribList[self.maintribID].Zdata
        elif xval == 'Pe':
            xdata = tribList[self.maintribID].Pedata
        else:
            raise RuntimeError('The requested X value is unknown, options are: dist, chi, Fa, Z, Pe')

        if yval == 'dist':
            ydata = tribList[self.maintribID].dist
        elif yval == 'FA':
            IDo = np.where(tribList[self.maintribID].FAdata == 0.)[0]
            self.FAdata[IDo] = 1.
            ydata = tribList[self.maintribID].FAdata
        elif yval == 'chi':
            ydata = tribList[self.maintribID].Chidata
        elif yval == 'Z':
            ydata = tribList[self.maintribID].Zdata
        elif yval == 'Pe':
            ydata = tribList[self.maintribID].Pedata
        else:
            raise RuntimeError('The requested Y value is unknown, options are: dist, chi, Fa, Z, Pe')

        traces = []
        if onlyMain == 0:
            for i in range(len(tribList)):
                if len(tribList[i].streamX) > 1 and tribList[i].dist[-1] > distLimit:

                    if xval == 'dist':
                        xd = tribList[i].dist
                    elif xval == 'FA':
                        IDo = np.where(tribList[i].FAdata == 0.)[0]
                        tribList[i].FAdata[IDo] = 1.
                        xd = tribList[i].FAdata
                    elif xval == 'chi':
                        xd = tribList[i].Chidata
                    elif xval == 'Z':
                        xd = tribList[i].Zdata
                    else:
                        xd = tribList[i].Pedata

                    if yval == 'dist':
                        y1 = tribList[i].dist
                    elif yval == 'FA':
                        IDo = np.where(tribList[i].FAdata == 0.)[0]
                        self.FAdata[IDo] = 1.
                        y1 = tribList[i].FAdata
                    elif yval == 'chi':
                        y1 = tribList[i].Chidata
                    elif yval == 'Z':
                        y1 = tribList[i].Zdata
                    else:
                        y1 = tribList[i].Pedata

                    traces.append(
                        Scatter(
                            x=xd,
                            y=y1,
                            mode='lines',
                            name="'spline'",
                            opacity = 1.,
                            line=dict(
                                shape='spline',
                                color = tribColor,
                                width = tribWidth
                            ),
                        )
                    )

        traces.append(
            Scatter(
                x=xdata,
                y=ydata,
                mode='lines',
                name="'spline'",
                opacity = 1.,
                line=dict(
                    shape='spline',
                    color = mainColor,
                    width = mainWidth
                )
            )
        )

        layout = dict(
            showlegend=False,
            title=title,
            width=width,
            height=height,
            hovermode='closest',
            xaxis=XAxis(showgrid=True, zeroline=True, showticklabels=True),
            yaxis=YAxis(showgrid=True, zeroline=True, showticklabels=True)
        )

        fig = Figure(data=traces, layout=layout)
        plotly.offline.iplot(fig)

        return

    def timeProfiles(self, pData = None, pDist = None, width = 800, height = 400, linesize = 2,
                    title = 'Profile evolution with time'):
        """
        Plot profile mean, max and min.

        Parameters
        ----------

        variable: pData
            Dataset to plot along Y axis.

        variable: pDist
            Dataset to plot along X axis.

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: color
            Color scale.

        variable: linesize, markersize
            Requested size for the line and markers.

        variable: title
            Title of the graph.

        Return:

        variable: minZ, meanZ, maxZ
            Y values for the profile (minZ, meanZ, maxZ)
        """

        trace = {}
        data = []

        for i in range(0,len(pData)):
            trace[i] = Scatter(
                x=pDist[i],
                y=pData[i],
                mode='lines',
                line=dict(
                    shape='spline',
                    width = linesize,
                    #color = color
                ),
            )
            data.append(trace[i])

        layout = dict(
            title=title,
            width=width,
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
