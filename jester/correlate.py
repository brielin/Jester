import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from collections import deque
from scipy import stats
from math import exp
from time import time
from . import util as ju
from IPython import embed

def correlate(IN, wt=100, rMin=0.0, rMax=1.0, minMAF=0.0, verbose=False, L=0):
    IN.getSNPIterator()
    if verbose:
        print IN.N,"individuals. Beginning to process",IN.numSNPs,"SNPs."
    if L==0: L=IN.numSNPs
    nanarr = np.zeros((L,wt))
    nanarr[:] = np.NaN
    res = pd.DataFrame(nanarr,columns=(np.arange(nanarr.shape[1])+1))
    t=time()
    win = deque()
    index = []
    for i in range(L):
        if (i%1000 == 0) and verbose: print "At SNP", i, "time spent:", time()-t
        snp,chrm,id,pos = IN.next()
        af = np.mean(snp)
        if (af < minMAF) or (af > 1- minMAF):
            continue
        rVals = np.array([ju.corr(x,snp,IN.P) for x in win])
        win.appendleft(snp)
        if( len(win) > wt ):
            win.pop()
        res.iloc[i,0:len(rVals)] = rVals
        index.append(id)
    res.index = index
    return res
