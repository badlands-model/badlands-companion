##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to export sea level curves into Badlands simulation.
"""

import errno
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class toolSea:
    """
    Class for creating Badlands sea level curve.
    """

    def __init__(self, curve1=None, curve2=None):
        """
        Initialization function which takes the Haq87 and Haq87 normalized curves. If you build
        your own sea-level curve you need to define these parameters as None.

        Parameters
        ----------
        variable : curve1
            HAQ 87 (EXXON) Long-term Sea Level Curve, digitized by Sabin Zahirovic, 13 June 2014
            Digitization error is at best -/+ 1 m and up to +/- 5 m
                + Column 1 = Sea Level (m)
                + Column 2 = Absolute Age (Ma)

        variable : curve2
            Normalized version of curve 1 from Haq 87.

        """
        self.df1 = None
        self.df2 = None


        self.time1 = None
        self.time2 = None
        self.sea1 = None
        self.sea2 = None
        self.func1 = None
        self.func2 = None

        self.minsea = None
        self.maxsea = None
        self.mintime = None
        self.maxtime = None

        self.periodSea = None
        self.periodEnd = None
        self.periodStart = None
        self.zoomTime = None
        self.zoomSea1 = None
        self.zoomSea2 = None
        self.minZsea = None
        self.maxZsea = None
        self.minZtime = None
        self.maxZtime = None

        if curve1 != None and curve2 != None:
            self.build = False
            self.df1 = pd.read_csv(curve1, sep=r'\s+', header=None, names=['h','t'])
            self.df2 = pd.read_csv(curve2, sep=r'\s+', header=None, names=['h','t'])
        else:
            self.build = True
            self.sl1 = None
            self.sl2 = None
            self.sl = None
            self.time = None

        return

    def buildCurve(self, timeExt = None, timeStep = None, seaExt = None,
                   ampExt = None, periodExt = None):
        """
        Curve created which interpolate linearly the averaged values of sea-level
        trends over the specified time period.

        Parameters
        ----------
        variable: timeExt
            Extent of the simulation time: start/end time (in years)

        variable: timeStep
            Discretisation step for time range (in years).

        variable: seaExt
            Sea level value for starting and ending times (in metres)

        variable: ampExt
            Amplitude of the sea level wave for starting and ending times (in metres)

        variable: periodExt
            Period of the sea level wave for starting and ending times (in years)
        """

        dt = float(timeStep)
        so = float(seaExt[0])
        sm = float(seaExt[1])
        to = float(timeExt[0])
        tm = float(timeExt[1])+dt
        Ao = float(ampExt[0])
        Am = float(ampExt[1])
        Po = float(periodExt[0])
        Pm = float(periodExt[1])

        self.time = np.arange(to,tm,dt,dtype=np.float)

        # Sea-level
        a0 = (sm - so)/(tm - to)
        b0 = so - a0 * to
        self.sl = a0 * self.time + b0
        # Amplitude
        a1 = (Am - Ao)/(tm - to)
        b1 = Ao - a1 * to
        A = a1 * self.time + b1
        # Period
        a2 = (Pm - Po)/(tm - to)
        b2 = Po - a2 * to
        P = a2 * self.time + b2
        # Extent
        self.sl1 = a0 * self.time + b0 - 1.
        self.sl2 = a0 * self.time + b0 + 1.

        for t in range(len(self.time)):
            self.sl[t] += A[t] * np.cos(2.* np.pi * (self.time[t] - to) / P[t])
            self.sl1[t] += -A[t]
            self.sl2[t] += A[t]

        return


    def readCurve(self, timeMax = 250, timeMin = 0, dt = 0.1):
        """
        Read digitized curves.

        Parameters
        ----------
        variable: timeMax
            Simulation time start in Ma.

        variable: timeMin
            Simulation time end in Ma.

        variable: dt
            Discretisation step for time range (in Ma).
        """

        self.df1.columns[1]
        list1 = list(self.df1)
        self.time1 = self.df1[list1[0:len(list1)-1]].values[:,0]
        self.sea1 = self.df1[list1[len(list1)-1]].values
        self.func1 = interpolate.interp1d(self.time1, self.sea1)

        self.df2.columns[1]
        list2 = list(self.df2)
        self.time2 = self.df2[list2[0:len(list2)-1]].values[:,0]
        self.sea2 = self.df2[list2[len(list2)-1]].values
        self.func2 = interpolate.interp1d(self.time2, self.sea2)

        self.minsea = min(self.sea1.min(),self.sea2.min())
        self.maxsea = max(self.sea1.max(),self.sea2.max())
        self.mintime = min(self.time1.min(),self.time2.min())
        self.maxtime = max(self.time1.max(),self.time2.max())

        # The base of the model is fixed to -600 m
        self.periodSea = [-500,500]

        self.periodEnd = [timeMax,timeMax]
        self.periodStart = [timeMin,timeMin]

        self.zoomTime = np.arange(timeMin, timeMax+dt, dt)
        self.zoomSea1 = self.func1(self.zoomTime)
        self.zoomSea2 = self.func2(self.zoomTime)

        self.minZsea = min(self.zoomSea1.min(),self.zoomSea2.min())
        self.maxZsea = max(self.zoomSea1.max(),self.zoomSea2.max())

        self.minZtime = self.zoomTime.min()
        self.maxZtime = self.zoomTime.max()

        return

    def plotCurves(self, fsize=(8,8), saveFig = False, figName = 'sealevel'):
        """
        Plot the 2 Haq curves and zoom to the region of interest.

        Parameters
        ----------
        variable: fsize
            Size of the figure to plot.

        variable: saveFig
            Saving the figure (boolean).

        variable: figName
            Name of the saved file or of the plot if you have created your own sea-level curve.
        """

        if self.build == False:

            f = plt.figure(figsize=fsize)
            plt.subplot(1, 2, 1)

            plt.plot(self.periodSea, self.periodStart,'--', color='dimgrey', linewidth=1.5)
            plt.plot(self.periodSea, self.periodEnd,'--', color='dimgrey', linewidth=1.5)
            plt.fill_between(self.periodSea, self.periodStart, self.periodEnd,
                             color='silver', alpha='0.1')


            plt.plot(self.sea1, self.time1, color='slateblue', linewidth=2, label='Haq 87')
            plt.plot(self.sea2, self.time2, color='darkcyan', linewidth=2, label='Haq 87 [norm]')

            titlepos = plt.title('Eustatic curve', fontsize=11, weight='bold')
            titlepos.set_y(1.02)

            plt.xlabel('Sea Level [m]',fontsize=10)
            plt.ylabel('Time [Ma]',fontsize=10)

            plt.grid(True)

            plt.xlim( self.minsea-10, self.maxsea+10)
            plt.ylim( self.mintime-5, self.maxtime+5 )

            plt.tick_params(axis='both', which='major', labelsize=8)

            ax = plt.subplot(1, 2, 2)

            plt.plot(self.periodSea, self.periodStart,'--', color='dimgrey', linewidth=1.5)
            plt.plot(self.periodSea, self.periodEnd,'--', color='dimgrey', linewidth=1.5)
            plt.fill_between(self.periodSea, self.periodStart, self.periodEnd,
                             color='silver', alpha='0.1')

            plt.plot(self.zoomSea1, self.zoomTime, color='slateblue',
                     linewidth=2, label='Haq 87')
            plt.plot(self.zoomSea2, self.zoomTime, color='darkcyan',
                     linewidth=2, label='Haq 87 [norm]')

            titlepos = plt.title('Eustatic curve zoom', fontsize=11, weight='bold')
            titlepos.set_y(1.02)

            plt.xlabel('Sea Level [m]',fontsize=10)
            plt.ylabel('Time [Ma]',fontsize=10)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            plt.grid(True)

            plt.xlim( self.minZsea-10, self.maxZsea+10)
            plt.ylim( self.minZtime-5, self.maxZtime+5 )
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=10, frameon=False)

            plt.tick_params(axis='both', which='major', labelsize=8)

            if saveFig:
                plotfile = str(figName)+'.pdf'
                plt.savefig(plotfile, dpi = 150, orientation = 'portrait')
                print('PDF figure saved: ',plotfile)

            plt.show()

        else:

            title = figName
            linesize = 3
            markersize = 0.001
            width = fsize[0]*100
            height = fsize[1]*100
            data = [
                Scatter(
                    x=self.sl1,
                    y=self.time,
                    fill= None,
                    mode='lines',
                    line=dict(color='rgba(250, 250, 250, 0.)'),
                    showlegend=False,
                    ),
                Scatter(
                    x=self.sl2,
                    y=self.time,
                    fill='tonextx',
                    fillcolor='rgba(68, 68, 68, 0.1)',
                    line=dict(color='rgba(250, 250, 250, 0.)'),
                    showlegend=False,
                    ),
                Scatter(
                    x=self.sl,
                    y=self.time,
                    mode='lines+markers',
                    #name="'spline'",
                    line=dict(
                        shape='spline',
                        color='rgb(31, 119, 180)',
                        width = linesize
                        ),
                    showlegend=False,
                    marker = dict(
                        symbol='circle',
                        size = markersize,
                        color = 'white',
                        line = dict(
                            width = 1,
                            color = 'black'
                            ),
                        )
                    ),
                ]
            layout = dict(
                title=title,
                width=width,
                height=height
                )

            fig = Figure(data=data, layout=layout)
            plotly.offline.iplot(fig)

        return

    def exportCurve(self, curve='HaqNorm', factor=1.e6, nameCSV='sea'):
        """
        Write CSV sea level file following Badlands requirements:
            + 2 columns file containing time in years (1st column) and sea level in metres (2nd column),
            + time is ordered in increasing order starting at the oldest time,
            + past times are negative,
            + the separator is a space.

        Parameters
        ----------
        varaible: curve
            Choosing between normalized and not normalized Haq curve.

        variable : factor
            Factor to convert from given time unit to years (ex: Ma -> a).

        variable: nameCSV
            Name of the saved CSV sea-level file.
        """

        if self.build == False:
            flipTime = -np.flipud(self.zoomTime) * factor

            if curve == 'HaqNorm':
                flipSea = np.flipud(self.zoomSea2)
            else:
                flipSea = np.flipud(self.zoomSea1)

            df = pd.DataFrame({'X':np.around(flipTime, decimals=0),'Y':np.around(flipSea, decimals=3)})
            df.to_csv(str(nameCSV)+'.csv',columns=['X', 'Y'], sep=' ', index=False ,header=0)

        else:
            df = pd.DataFrame({'X':np.around(self.time, decimals=0),'Y':np.around(self.sl, decimals=3)})
            df.to_csv(str(nameCSV)+'.csv',columns=['X', 'Y'], sep=' ', index=False ,header=0)

        return
