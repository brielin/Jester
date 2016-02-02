import os
import fnmatch
import numpy as np
import scipy as sp
import pandas as pd
import sys
from glob import glob
from scipy import stats
from math import exp
from time import time
from IPython import embed

def collect(fbase,alpha=0.05):
    fsplit = fbase.rsplit('/',1)
    if len(fsplit) > 1:
        directory=fsplit[0]
        fb=fsplit[1]
    else:
        directory='.'
        fb=fbase
    firstZ = True
    firstJ = True
    ZpList = []
    JpList = []
    t=time()
    for i,file in enumerate(glob(directory+'/'+fb+'*')):
        if i%1000 == 0: print i, time()-t
        if fnmatch.fnmatch(file, directory+'/'+fb+'*.Z.pkl'):
            ZpList.append(pd.read_pickle(file)[0:1000])
        elif fnmatch.fnmatch(file, directory+'/'+fb+'*.J.pkl'):
            JpList.append(pd.read_pickle(file).iloc[:,:,0:1000])

    ZpVals = pd.concat(ZpList,axis=0)
    JpVals = pd.concat(JpList,axis=2)
    numSamples = len( ZpVals )
    index = int( np.round( (alpha)*(numSamples-1) ))
    sortZpVals = ZpVals.sort(columns=0)
        # Hack because there is no way to sort in pd.Panel
    sortJpVals = pd.Panel( np.sort( JpVals ), items = JpVals.items,
                           major_axis=JpVals.major_axis,
                           minor_axis = range(numSamples) )
    alphaC_Z = sortZpVals.iloc[index]
    alphaC_J = sortJpVals.minor_xs(index)
    print alphaC_Z, alphaC_J
    return alphaC_Z[0], alphaC_J
