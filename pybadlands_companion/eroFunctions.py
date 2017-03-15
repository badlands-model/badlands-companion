##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to build dependency of erodibility coefficient to 
precipitation and sediment supply.
"""

import os
import math
import numpy as np
import pandas as pd

class eroFunctions():
    """
    Class for creating simple dependencies functions for erodibility coefficients.
    """
    
    def __init__(self, min=0., max=None, sample=None):
        
        self.x = np.linspace(min,max,num=sample,endpoint=True)
        
        return

    def sinfct(self):
        """
        Sinusoidal function centered  along the X-axis.

        Returns
        -------
        y : 1d array
            Centered sinusoidal function for x

        """
        return np.sin(2.*np.pi*self.x*0.5/self.x.max())


    def gaussfct(self, mean, sigma):
        """
        Gaussian function.

        Parameters
        ----------
        mean : float
            Gaussian parameter for center (mean) value.
        sigma : float
            Gaussian parameter for standard deviation.

        Returns
        -------
        y : 1d array
            Gaussian function for x

        """
        return np.exp(-((self.x - mean) ** 2.) / float(sigma) ** 2.)



    def gauss2fct(self, mean1, sigma1, mean2, sigma2):
        """
        Gaussian function of two combined Gaussians.

        Parameters
        ----------
        mean1 : float
            Gaussian parameter for center (mean) value of left-side Gaussian.
            Note mean1 <= mean2 reqiured.
        sigma1 : float
            Standard deviation of left Gaussian.
        mean2 : float
            Gaussian parameter for center (mean) value of right-side Gaussian.
            Note mean2 >= mean1 required.
        sigma2 : float
            Standard deviation of right Gaussian.

        Returns
        -------
        y : 1d array
            Function with left side up to `mean1` defined by the first
            Gaussian, and the right side above `mean2` defined by the second.
            In the range mean1 <= x <= mean2 the function has value = 1.

        """
        assert mean1 <= mean2, 'mean1 <= mean2 is required.  See docstring.'
        y = np.ones(len(self.x))
        idx1 = self.x <= mean1
        idx2 = self.x > mean2
        y[idx1] = np.exp(-((self.x[idx1] - mean1) ** 2.) / float(sigma1) ** 2.) 
        y[idx2] = np.exp(-((self.x[idx2] - mean2) ** 2.) / float(sigma2) ** 2.) 
        
        return y

    def gbellfct(self, a, b, c):
        """
        Generalized Bell function generator.

        Parameters
        ----------
        a : float
            Bell function parameter controlling width.
        b : float
            Bell function parameter controlling slope.
        c : float
            Bell function parameter controlling center.

        Returns
        -------
        y : 1d array
            Generalized Bell function.

        Notes
        -----
        Definition of Generalized Bell function is:

            y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])

        """
        return 1. / (1. + np.abs((self.x - c) / a) ** (2 * b))
    
    def trapfct(self, abcd):
        """
        Trapezoidal function generator.

        Parameters
        ----------
        abcd : 1d array, length 4
            Four-element vector.  Ensure a <= b <= c <= d.

        Returns
        -------
        y : 1d array
            Trapezoidal function.

        """
        assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
        a, b, c, d = np.r_[abcd]
        assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                              a <= b <= c <= d.'
        y = np.ones(len(self.x))

        idx = np.nonzero(self.x <= b)[0]
        y[idx] = trimf(self.x[idx], np.r_[a, b, b])

        idx = np.nonzero(self.x >= c)[0]
        y[idx] = trimf(self.x[idx], np.r_[c, c, d])

        idx = np.nonzero(self.x < a)[0]
        y[idx] = np.zeros(len(idx))

        idx = np.nonzero(self.x > d)[0]
        y[idx] = np.zeros(len(idx))

        return y
    
    def trifct(self, abc):
        """
        Triangular function generator.

        Parameters
        ----------
        abc : 1d array, length 3
            Three-element vector controlling shape of triangular function.
            Requires a <= b <= c.

        Returns
        -------
        y : 1d array
            Triangular function.

        """
        assert len(abc) == 3, 'abc parameter must have exactly three elements.'
        a, b, c = np.r_[abc]     # Zero-indexing in Python
        assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

        y = np.zeros(len(self.x))

        # Left side
        if a != b:
            idx = np.nonzero(np.logical_and(a < self.x, self.x < b))[0]
            y[idx] = (self.x[idx] - a) / float(b - a)

        # Right side
        if b != c:
            idx = np.nonzero(np.logical_and(b < self.x, self.x < c))[0]
            y[idx] = (c - self.x[idx]) / float(c - b)

        idx = np.nonzero(self.x == b)
        y[idx] = 1
        return y
    
    def sigfct(self, b, c):
        """
        The basic sigmoid function generator.

        Parameters
        ----------
        b : float
            Offset or bias.  This is the center value of the sigmoid, where it
            equals 1/2.
        c : float
            Controls 'width' of the sigmoidal region about `b` (magnitude); also
            which side of the function is open (sign). A positive value of `a`
            means the left side approaches 0.0 while the right side approaches 1.;
            a negative value of `c` means the opposite.

        Returns
        -------
        y : 1d array
            Generated sigmoid values, defined as y = 1 / (1. + exp[- c * (x - b)])

        """
        return 1. / (1. + np.exp(- c * (self.x - b)))
 
    def linfct(self, a, b):
        """
        The basic linear function generator.

        Parameters
        ----------
        a : float
            maximum value for the function
        b : float
            Offset from 0.

        Returns
        -------
        y : 1d array
            Generated linear function y = s x + b

        """
        s = (a-b)/self.x.max()
        return s * (self.x) + b
    
    def exportFunction(self, val=None, nameCSV='sedsupply'):
        """
        Write CSV file following Badlands requirements:
            + 2 columns file containing the X values (1st column) and the Y values (2nd column),
            + the separator is a space.
        
        Parameters
        ----------
        
        variable : val
            Function used in either the sediment supply or slope dependency.
            
        variable: nameCSV
            Name of the saved CSV file.
        """
        df = pd.DataFrame({'X':self.x,'Y':val})
        df.to_csv(str(nameCSV)+'.csv',columns=['X', 'Y'], sep=' ', index=False ,header=0)
        
        return