##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to build the time series evolution of the stratigraphy
grid based on wedges topology.
"""

import os
import math
import h5py
import errno
import numpy as np

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class stratiMesh:
    """
    Class for creating irregular stratigraphic mesh from Badlands outputs.
    """

    def __init__(self, folder=None, xdmfName = 'stratal_series', ncpus=1, layperstep=1, dispTime=None):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable : xdmfName
            Name of Badlands stratigraphic grid outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        variable: layperstep
            Number of layers created between each display
            interval (obtained from the XmL input file).
        variable: dispTime
            Time interval in years used to display badlands outputs.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus

        self.x = None
        self.y = None
        self.elev = None
        self.dep = None
        self.th = None
        self.timestep = 0
        self.layperstep = 0
        self.laynb = 0
        self.startStep = None
        self.endStep = None

        # Assign file names
        self.h5TIN = 'h5/tin.time'
        self.h5Strat = 'h5/strat.time'
        self.xmffile = 'xmf/stratal.time'
        self.xdmfName = xdmfName+'.xdmf'
        self.dispTime = dispTime
        self.tnow = None

        return

    def _loadTIN(self, step, rank):
        """
        Load TIN grid to extract cells connectivity and vertices position.

        Parameters
        ----------
        variable : step
            Specific step at which the TIN variables will be read.
        variable: rank
            TIN file for the considered CPU.
        """

        h5file = self.folder+'/'+self.h5TIN+str(step)+'.p'+str(rank)+'.hdf5'
        df = h5py.File(h5file, 'r')
        coords = np.array((df['/coords']))
        cells = np.array((df['/cells']),dtype=int)

        return coords, cells

    def _write_hdf5(self, xt, yt, zt, cellt, step, rank):
        """
        Write the HDF5 file containing the stratigraphic mesh variables.

        Parameters
        ----------
        variable : xt
            X-axis coordinates of the vertices.
        variable : yt
            Y-axis coordinates of the vertices.
        variable : zt
            Z-axis coordinates of the vertices.
        variable : cellt
            Wedge cells connectivity.
        variable : step
            Specific step at which the TIN variables will be read.
        variable: rank
            TIN file for the considered CPU.
        """

        h5file = self.folder+'/'+self.h5Strat+str(step)+'.p'+str(rank)+'.hdf5'
        with h5py.File(h5file, "w") as f:

            # Write node coordinates and elevation
            f.create_dataset('coords',shape=(len(xt),3), dtype='float32', compression='gzip')
            f["coords"][:,0] = xt[:,0]
            f["coords"][:,1] = yt[:,0]
            f["coords"][:,2] = zt[:,0]

            f.create_dataset('cells',shape=(len(cellt[:,0]),6), dtype='int32', compression='gzip')
            f["cells"][:,:] = cellt

        return

    def _write_xmf(self, step, elems, nodes):
        """
        Write the XMF file which load and read the hdf5 parameters files at any given step.

        Parameters
        ----------
        variable : step
            Specific step at which the TIN variables will be read.
        variable: elems
            Number of wedges elements per processor mesh.
        variable: nodes
            Number of irregular points per processor mesh
        """

        xmf_file = self.folder+'/'+self.xmffile+str(step)+'.xmf'
        print xmf_file
        f= open(str(xmf_file),'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')
        f.write('    <Grid GridType="Collection" CollectionType="Spatial">\n')
        f.write('      <Time Type="Single" Value="%s"/>\n'%self.tnow)

        for p in range(self.ncpus):
            pfile = self.h5Strat+str(step)+'.p'+str(p)+'.hdf5'
            f.write('      <Grid Name="Block.%s">\n' %(str(p)))
            f.write('         <Topology Type="Wedge" NumberOfElements="%d" BaseOffset="1">\n'%elems[p])
            f.write('          <DataItem Format="HDF" DataType="Int" ')
            f.write('Dimensions="%d 6">%s:/cells</DataItem>\n'%(elems[p],pfile))
            f.write('         </Topology>\n')

            f.write('         <Geometry Type="XYZ">\n')
            f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
            f.write('Dimensions="%d 3">%s:/coords</DataItem>\n'%(nodes[p],pfile))
            f.write('         </Geometry>\n')
            f.write('      </Grid>\n')

        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()

        return

    def _write_xdmf(self):
        """
        Write the XDMF file which load and read the XMF parameters files for the requested steps.
        """

        f= open(self.folder+'/'+self.xdmfName,'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')
        f.write('    <Grid GridType="Collection" CollectionType="Temporal">\n')

        for p in range(self.startStep,self.endStep+1):
            xfile = self.xmffile+str(p)+'.xmf'
            f.write('      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)"/>\n' %xfile)

        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()

        return

    def outputSteps(self, startTime=0, endTime=5000):
        """
        Define the steps that need to be visualise.

        Parameters
        ----------
        variable : startTime
            First Badlands output time to visualise.
        variable: endStep
            Last Badlands output time to visualise.
        """

        self.startTime = startTime
        self.endTime = endTime
        self.tnow = startTime

        self.startStep = int(startTime/self.dispTime)
        self.endStep = int(endTime/self.dispTime)

        if not os.path.isdir(self.folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        assert self.startStep<=self.endStep, 'ERROR: End step lower than Start step.'

        for s in range(self.startStep,self.endStep+1):
            ptsnb = []
            cellnb = []
            for i in range(0, self.ncpus):
                coords, cells = self._loadTIN(s,i)
                x, y, z = np.hsplit(coords, 3)
                xt = np.concatenate((x, x), axis=0)
                yt = np.concatenate((y, y), axis=0)
                zt = np.concatenate((z, z+100), axis=0)
                cellt = np.concatenate((cells, cells+len(x)), axis=1)
                self._write_hdf5(xt, yt, zt, cellt, s, i)
                cellnb.append(len(cellt))
                ptsnb.append(len(xt))

            self._write_xmf(s, cellnb, ptsnb)
            self.tnow += self.dispTime

        self._write_xdmf()

        return
