import os
import fnmatch
import numpy as np
import scipy as sp
import pandas as pd
import sys
from scipy import stats
from math import exp
from time import time

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
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, fb+'*.Z.pkl'):
            if firstZ:
                ZpVals = pd.read_pickle(file)
                firstZ=False
            else:
                ZpVals =  pd.concat([ ZpVals, pd.read_pickle(file)])
        elif fnmatch.fnmatch(file, fb+'*.J.pkl'):
            if firstJ:
                JpVals = pd.read_pickle(file)
                firstJ=False
            else:
                JpVals =  pd.concat([ JpVals, pd.read_pickle(file)], axis=2)
    numSamples = len( ZpVals )
    index = int( np.round( (alpha)*(numSamples-1) ))
    sortZpVals = ZpVals.sort(columns=0)
        # Hack because there is no way to sort in pd.Panel
    sortJpVals = pd.Panel( np.sort( JpVals ), items = JpVals.items,
                           major_axis=JpVals.major_axis,
                           minor_axis = range(numSamples) )
    alphaC_Z = sortZpVals.iloc[index]
    alphaC_J = sortJpVals.minor_xs(index)
    return alphaC_Z[0], alphaC_J
