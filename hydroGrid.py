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

class hydroGrid: 
    """
    Class for analysing hydrometrics from Badlands outputs.
    """
    
    def __init__(self, folder=None, ncpus=1, ptXY=None):
        """
        Initialization function which takes the folder path to Badlands outputs 
        and the number of CPUs used to run the simulation. It also takes a point 
        coordinates contained in a catchment of interest.
        
        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.

        variable: ncpus
            Number of CPUs used to run the simulation.

        variable: ptXY
            X-Y coordinates of the point contained within the catchment.

        """
        
        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')
            
        self.ncpus = ncpus
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

        for i in range(0, self.ncpus):
            
            df = h5py.File('%s/flow.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
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
            
            if i == 0:
                X1 = Xl[:,0]
                X2 = Xl[:,1]
                Y1 = Yl[:,0]
                Y2 = Yl[:,1]
                Z = Zl
                FA = FAl
                Chi = Chil
                Basin = Basinl
            else:
                X1 = np.append(X1, Xl[:,0])
                X2 = np.append(X2, Xl[:,1])
                Y1 = np.append(Y1, Yl[:,0])
                Y2 = np.append(Y2, Yl[:,1])
                Z = np.append(Z, Zl)
                FA = np.append(FA, FAl)
                Chi = np.append(Chi, Chil)
                Basin = np.append(Basin, Basinl)
        
        XY = np.column_stack((X1, Y1))
        if self.ptXY[0] < X1.min() or self.ptXY[0] > X1.max():
            raise RuntimeError('X coordinate of given point is not in the simulation area.')
        
        if self.ptXY[1] < Y1.min() or self.ptXY[1] > Y1.max():
            raise RuntimeError('Y coordinate of given point is not in the simulation area.')
            
        distance,index = spatial.KDTree(XY).query(self.ptXY)
        basinIDs = np.where(Basin == Basin[index])[0]
        
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

        while (loop ):
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
                    if k > 1 and int(next) == streamIDs[k-1] or \
                        int(next) == streamIDs[k-2]:
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
    
    
    def viewNetwork(self, markerPlot = True, linePlot = False, lineWidth = 3, markerSize = 15, \
                   val = 'chi', width = 800, height = 400, colorMap = None, \
                   colorScale = None, reverse = False, title = '<br>Stream network graph'):
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
            
        variable: reverse
            Boolean for reverse color scale.
            
        variable: title
            Title of the graph.
        """
            
        traces = []
        if val == 'chi':
            valData = self.Chi
        elif val == 'FA':
            IDo = np.where(self.FA == 0.)[0]
            self.FA[IDo] = 1.
            valData = np.log(self.FA)
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
                        line=Line(
                            width=lineWidth,
                            reversescale=reverse,
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
                        line=Line(
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
                    marker=Marker(
                        showscale=True,
                        colorscale=colorScale,
                        reversescale=reverse,
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
        
        fig = Figure(data=Data(traces),
                  layout=Layout(
                      title=title,
                      titlefont=dict(size=16),
                      showlegend=False, 
                      width=width,
                      height=height,
                      hovermode='closest',
                      xaxis=XAxis(showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=YAxis(showgrid=True, zeroline=True, showticklabels=True)
                  )
            )
        plotly.offline.iplot(fig)
        
        return
    
    def viewStream(self, linePlot = False, lineWidth = 3, markerSize = 15, \
                   val = 'chi', width = 800, height = 400, colorMap = None, \
                   colorScale = None, reverse = False, title = '<br>Main stream'):
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
            
        variable: reverse
            Boolean for reverse color scale.
            
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
            valData = np.log(self.FA)
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
                        line=Line(
                            width=lineWidth,
                            reversescale=reverse,
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
                        line=Line(
                            width=lineWidth,
                            color='#888'
                        ),
                        hoverinfo='none',
                        mode='lines'
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
                marker=Marker(
                    showscale=True,
                    colorscale=colorScale,
                    reversescale=reverse,
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
        
        fig = Figure(data=Data(traces),
                  layout=Layout(
                      title=title,
                      titlefont=dict(size=16),
                      showlegend=False, 
                      width=width,
                      height=height,
                      hovermode='closest',
                      xaxis=XAxis(showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=YAxis(showgrid=True, zeroline=True, showticklabels=True)
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
            xdata = np.log(self.FAdata)
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
            ydata = np.log(self.FAdata)
        elif yval == 'chi':
            ydata = self.Chidata
        elif yval == 'Z':
            ydata = self.Zdata
        elif yval == 'Pe':
            ydata = self.Pedata
        else:
            raise RuntimeError('The requested Y value is unknown, options are: dist, chi, Fa, Z, Pe')
    
        data = Data([
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
        ])

        layout = dict(
            title=title,
            width=width, 
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
        
        return