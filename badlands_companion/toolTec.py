##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to create displacement fields on regular grids for
Badlands simulation.
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

class toolTec:
    """
    Class for creating Badlands cumulative vertical displacement maps.
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
        self.disp = None

        self.nx = None
        self.ny = None

        return

    def _slope(self, axis, slp, base):
        return slp * axis + base

    def slopeTec(self, base=0., slope=0.1, axis='X'):
        """
        Build a simple slope-like displacement map.

        Parameters
        ----------
        variable : base
            Base of the displacements in metres.

        variable: slope
            Lower and upper values of the Y axis in metres.

        variable: axis
            Slope displacements along X or Y axis.
        """
        if axis == 'X':
            disps = np.array([self._slope(x,slope,base) for x,y in zip(np.ravel(self.Xgrid), np.ravel(self.Ygrid))])

            return disps.reshape(self.Xgrid.shape)

        if axis == 'Y':
            disps = np.array([self._slope(y,slope,base) for y,x in zip(np.ravel(self.Ygrid), np.ravel(self.Xgrid))])

            return disps.reshape(self.Ygrid.shape)

    def stepTec(self, A=None, base=0., edge1=None, edge2=None, axis='X'):
        """
        Build a simple step-like displacements map.

        Parameters
        ----------
        variable : A
            Amplitude of the step function (in metres).

        variable : base
            Base of the displacements in metres.

        variable: edge1, edge2
            Extent of the step function i.e. min/max (in metres).

        variable: axis
            Slope displacements along X or Y axis.
        """

        self.nx = len(self.x)
        self.ny = len(self.y)

        dispgrid = np.zeros((self.ny,self.nx))
        dispgrid.fill(base)

        for j in range(0,self.ny):
            for i in range(0,self.nx):
                if axis == 'X':
                    if self.x[i] >= edge1 and self.x[i] <= edge2 :
                        dispgrid[j,i] += A

                if axis == 'Y':
                    if self.y[j] >= edge1 and self.y[j] <= edge2 :
                        dispgrid[j,i] += A

        return dispgrid

    def waveTec(self, A=None, P=None, base=None, center=None, axis='X'):
        """
        Build a simple sine wave displacement map.

        Parameters
        ----------
        variable : A,P
            Sine wave parameters.

        variable: base
            Base of the displacement in metres.

        variable: center
            X coordinates for the centre of the sine wave.

        variable: axis
            Slope displacements along X or Y axis.
        """

        self.nx = len(self.x)
        self.ny = len(self.y)

        dispgrid = np.zeros((self.ny,self.nx))
        dispgrid.fill(base)

        for j in range(0,self.ny):
            for i in range(0,self.nx):
                if axis == 'X':
                    if abs(self.x[i] - center) <= P*0.5:
                        dispgrid[j,i] = 0.5 * A * np.cos( 2.* np.pi * (self.x[i] - center) / P) + base + 0.5 * A

                if axis == 'Y':
                    if abs(self.y[j] - center) <= P*0.5:
                        dispgrid[j,i] = 0.5 * A * np.cos( 2.* np.pi * (self.y[j] - center) / P) + base + 0.5 * A

        return dispgrid

    def dispGrid(self, disp=None, nameCSV='disp'):
        """
        Define the CSV displacement map for Badlands simulation.

        Parameters
        ----------
        variable: disp
            Displacement of the grid.

        variable: nameCSV
            Name of the saved CSV topographic file.
        """
        df = pd.DataFrame({'Z':disp.flatten()})
        df.to_csv(str(nameCSV)+'.csv',columns=['Z'], sep=' ', index=False ,header=0)

        return

    def dispView(self, width = 800, height = 800, dispmin = None, dispmax = None, dispData = None, title='Export Grid'):
        """
        Use Plotly library to visualise the displacement grid in 3D.

        Parameters
        ----------
        variable : resolution
            Required resolution for the model grid (in metres).

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: dispmin
            Minimal displacement.

        variable: dispmax
            Maximal displacement.

        variable: dispData
            Displacement data to plot.

        variable: title
            Title of the graph.
        """

        if dispmin == None:
            dispmin = self.dispi.min()

        if dispmax == None:
            dispmax = self.dispi.max()

        data = [ Surface( x=self.x, y=self.y, z=dispData, colorscale='Portland' ) ]

        layout = Layout(
            title='Export Grid',
            autosize=True,
            width=width,
            height=height,
            scene=dict(
                zaxis=dict(range=[dispmin, dispmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                xaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                yaxis=dict(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
                bgcolor="rgb(244, 244, 248)"
            )
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return
