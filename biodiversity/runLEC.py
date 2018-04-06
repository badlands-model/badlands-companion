import time
import numpy as np
import pandas as pd
from skimage import graph
from LECmetrics import LEC

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

biodiv = LEC.LEC(filename='dataset/dem.csv',periodic=False,symmetric=False,
                 sigmap=0.1,sigmav=None,connected=False,
                 delimiter=',',header=0)

time0 = time.clock()
biodiv.computeLEC(timeit=True, fout=500)
if biodiv.rank == 0:
    print 'Compute LEC function took ',time.clock()-time0

time0 = time.clock()
biodiv.writeLEC('dataset/LECout.csv')
if biodiv.rank == 0:
    print 'Output function took ',time.clock()-time0
