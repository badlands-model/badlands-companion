##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to create dynamic topography files for Badlands inputs.
"""

import os
import math
import h5py
import errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO

import plotly
from plotly import tools
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class toolDynTopo:
    """
    Class for creating Badlands dynamic topography displacement maps.
    """

    def __init__(self, extentX=None, extentY=None, dx=None, filename='data/disp'):
        """
        Initialization function which takes the extent of the X,Y coordinates and the discretization value.
        Parameters
        ----------
        variable : extentX
            Lower and upper values of the X axis in metres.
        variable: extentY
            Lower and upper values of the Y axis in metres.
        variable: dx
            Discretisation values of the X-Y axes in metres.
        """
        if extentX == None:
            raise RuntimeError('Extent X-axis values are required.')
        self.extentX = extentX

        if extentY == None:
            raise RuntimeError('Extent Y-axis values are required.')
        self.extentY = extentY

        if dx == None:
            raise RuntimeError('Discretization space value along X axis is required.')
        self.dx = dx
        self.dy = self.dx

        self.x = np.arange(self.extentX[0],self.extentX[1]+self.dx,self.dx,dtype=np.float)
        self.y = np.arange(self.extentY[0],self.extentY[1]+self.dx,self.dy,dtype=np.float)

        self.nx = None
        self.ny = None
        self.stepNb = None
        self.filename = filename
        
        return
    
    def waveDT(self, A=None, L=None, V=None, endTime=None, dispTimeStep=None, axis='X'):
        """
        Build a simple sine wave displacement map.
        Parameters
        ----------
        variable : A, L, V
            The amplitude, wavelength, velocity of the Sine wave.
        variable: endTime
            The end time of the simulation.
        variable: dispTimeStep
            The time step of the each output of the dynamic topogrpahy data.
        variable: axis
            Slope displacements along X or Y axis.
        """

        self.nx = len(self.x)
        self.ny = len(self.y)
        
        self.stepNb = int(endTime/dispTimeStep)
        z = np.zeros((self.nx,self.ny))
        
        if axis == 'X':
            for k in range(0,self.stepNb):
                f = self.filename+str(k)+'.csv'
                
                disp = np.zeros((self.nx,self.ny))
                zn = np.zeros((self.nx,self.ny))
                
                t = (k+1)*dispTimeStep  # the wave starts at the first timestep
                posit = int(t*V/self.dx)  # the position of wave at time t
                
                if (posit<=int(L/self.dx)):  # when the wave does not fully reach the surface 
                    tmp = A*np.sin(np.pi*(V*t-self.x[:posit])/L)
                    zn[:posit,:] = np.array([tmp,]*self.ny).transpose()
                    disp[:posit,:] = zn[:posit,:] - z[:posit,:]
                elif (posit<=int(self.nx)):  # when the wave reaches the surface but does not leave
                    posit_pass = posit - int(L/self.dx)
                    tmp = A*np.sin(np.pi*(V*t-self.x[posit_pass:posit])/L)
                    zn[posit_pass:posit,:] = np.array([tmp,]*self.ny).transpose()
                    disp[posit_pass:posit,:] = zn[posit_pass:posit,:] - z[posit_pass:posit,:]
                else:                        # when the wave starts to leave the surface
                    posit_pass = posit - int(L/self.dx)
                    tmp = A*np.sin(np.pi*(V*t-self.x[posit_pass:])/L)
                    zn[posit_pass:,:] = np.array([tmp,]*self.ny).transpose()
                    disp[posit_pass:,:] = zn[posit_pass:,:] - z[posit_pass:,:]
                
                df = pd.DataFrame({'disp':disp.flatten('F')})
                df.to_csv(str(f),columns=['disp'], sep=' ', index=False ,header=0)           
                z = np.copy(zn)
               
        if axis == 'Y':   
            for k in range(0,self.stepNb):
                f = self.filename+str(k)+'.csv'
                
                disp = np.zeros((self.nx,self.ny))
                zn = np.zeros((self.nx,self.ny))
                
                t = (k+1)*dispTimeStep  # the wave starts at the first timestep
                posit = int(t*V/self.dx)  # the position of wave at time t
                
                if (posit<=int(L/self.dx)):  # when the wave does not fully reach the surface 
                    tmp = A*np.sin(np.pi*(V*t-self.y[:posit])/L)
                    zn[:,:posit] = np.array([tmp,]*self.nx)
                    disp[:,:posit] = zn[:,:posit] - z[:,:posit]
                elif (posit<=int(self.ny)):  # when the wave reaches the surface but does not leave
                    posit_pass = posit - int(L/self.dx)
                    tmp = A*np.sin(np.pi*(V*t-self.y[posit_pass:posit])/L)
                    zn[:,posit_pass:posit] = np.array([tmp,]*self.nx) 
                    disp[:,posit_pass:posit] = zn[:,posit_pass:posit] - z[:,posit_pass:posit]
                else:                        # when the wave starts to leave the surface
                    posit_pass = posit - int(L/self.dx)
                    tmp = A*np.sin(np.pi*(V*t-self.y[posit_pass:])/L)
                    zn[:,posit_pass:] = np.array([tmp,]*self.nx) 
                    disp[:,posit_pass:] = zn[:,posit_pass:] - z[:,posit_pass:]
                
                df = pd.DataFrame({'disp':disp.flatten('F')})
                df.to_csv(str(f),columns=['disp'], sep=' ', index=False ,header=0)
                z = np.copy(zn)

        return