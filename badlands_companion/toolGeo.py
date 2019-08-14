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
import numpy as np
import pylab as pl
import pandas as pd
from pylab import *
import numpy.ma as ma
import matplotlib.pyplot as plt
import os, sys, datetime, string

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class toolGeo:
    """
    Class for creating Badlands simple regular grid.
    """

    def __init__(self, extentX=None, extentY=None, dx=None):
        """
        Initialization function which takes the extent of the X,Y coordinates and the discretization value.

        Parameters
        ----------
        variable : extentX
            Lower and upper values of the X axis in metres.

        variable: extentY
            Lower and upper values of the Y axis in metres.

        variable: dx
            Discretisation value in metres.

        """
        if extentX == None:
            raise RuntimeError('Extent X-axis values are required.')
        self.extentX = extentX

        if extentY == None:
            raise RuntimeError('Extent Y-axis values are required.')
        self.extentY = extentY

        if dx == None:
            raise RuntimeError('Discretization space value is required.')
        self.dx = dx

        self.x = np.arange(self.extentX[0],self.extentX[1],self.dx,dtype=np.float)
        self.y = np.arange(self.extentY[0],self.extentY[1],self.dx,dtype=np.float)

        self.Xgrid, self.Ygrid = np.meshgrid(self.x, self.y)
        self.Z = None

        self.nx = None
        self.ny = None

        return

    def _slope(self, axis, slp, base):
        return slp * axis + base

    def buildSlope(self, base=0., slope=0.1, axis='X'):
        """
        Build a simple slope surface topography.

        Parameters
        ----------
        variable : base
            Lower and upper values of the X axis in metres.

        variable: slope
            Lower and upper values of the Y axis in metres.

        variable: axis
            Slope along X or Y axis.
        """
        if axis == 'X':
            zs = np.array([self._slope(x,slope,base) for x,y in zip(np.ravel(self.Xgrid), np.ravel(self.Ygrid))])

            return zs.reshape(self.Xgrid.shape)

        if axis == 'Y':
            zs = np.array([self._slope(y,slope,base) for y,x in zip(np.ravel(self.Ygrid), np.ravel(self.Xgrid))])

            return zs.reshape(self.Ygrid.shape)

    def _distance(self, x, y, a, b, xo, yo):
        return (x-xo)**2/a**2 + (y-yo)**2/b**2

    def _dome(self, c, zo, dist):
        return zo + c * np.sqrt(1. - dist )

    def buildDome(self, a=None, b=None, c=None, base=None, xcenter=None, ycenter=None):
        """
        Build a simple dome surface topography.

        Parameters
        ----------
        variable : a,b,c
            Ellipsoid parameters.

        variable: base
            Base of the model in metres.

        variable: xcenter,ycenter
            X,Y coordinates for the centre of the dome.
        """

        self.nx = len(self.x)
        self.ny = len(self.y)

        zgrid = np.zeros((self.ny,self.nx))

        for j in range(0,self.ny):
            for i in range(0,self.nx):
                dist = self._distance(self.x[i],self.y[j],a,b,xcenter,ycenter)
                if dist < 1.:
                    zgrid[j,i] = self._dome(c, base, dist)

        return zgrid

    def buildWave(self, A=None, P=None, base=None, xcenter=None):
        """
        Build a simple sine wave surface topography.

        Parameters
        ----------
        variable : A,P
            Sine wave parameters.

        variable: base
            Base of the model in metres.

        variable: xcenter
            X coordinates for the centre of the sine wave.
        """

        self.nx = len(self.x)
        self.ny = len(self.y)

        zgrid = np.zeros((self.ny,self.nx))
        zgrid.fill(base)

        for j in range(0,self.ny):
            for i in range(0,self.nx):
                if abs(self.x[i] - xcenter) <= P*0.5:
                    zgrid[j,i] = 0.5 * A * np.cos( 2.* np.pi * (self.x[i] - xcenter) / P) + base + 0.5 * A

        return zgrid

    def buildGrid(self, elevation=None, nameCSV='xyz'):
        """
        Define the CSV grid for Badlands simulation.

        Parameters
        ----------
        variable: elevation
            Elevation of the surface.

        variable: nameCSV
            Name of the saved CSV topographic file.
        """
        df = pd.DataFrame({'X':self.Xgrid.flatten(),'Y':self.Ygrid.flatten(),'Z':elevation.flatten()})
        df.to_csv(str(nameCSV)+'.csv',columns=['X', 'Y', 'Z'], sep=' ', index=False ,header=0)

        return

    def viewGrid(self, width = 800, height = 800, zmin = None, zmax = None, zData = None, title='Export Grid'):
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

        variable: zData
            Elevation data to plot.

        variable: title
            Title of the graph.
        """

        if zmin == None:
            zmin = self.zi.min()

        if zmax == None:
            zmax = self.zi.max()

        data = [ Surface( x=self.Xgrid, y=self.Ygrid, z=zData, colorscale='ylgnbu' ) ]

        layout = Layout(
            title='Export Grid',
            autosize=True,
            width=width,
            height=height,
            scene=dict(
                zaxis=dict(range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                xaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                yaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                bgcolor="rgb(244, 244, 248)"
            )
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return
