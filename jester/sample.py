import numpy as np
import scipy as sp
import pandas as pd
from collections import deque
from scipy import stats
from time import time
from jester import util as ju

def EZJVwrapper(Z,r,ZVals,R,rMax):
    if r**2 < rMax:
        return ju.EZJV( Z, ZVals, r, R )
    else:
        return np.zeros((len(R), len(Z)))

def sample(IN, wt=100, wr=100, wStep=0, rMin=0.0, rMax=1.0,rRange=False,
           numSamples=1000, minMAF=0.05, verbose=False, L=0):
    if rRange:
        R = np.array([0.0, 0.0001, 0.0004, 0.001, 0.002, 0.004, 0.008  ])
    else:
        R = np.array([rMin])
    if( wStep > 0 ):
        W = np.arange( wStep, wt + wStep, wStep ) ## wStep to wt
    else:
        W = np.array( [wt] )

    ZMaxStats = np.zeros(numSamples)
    JMaxStats = np.zeros( (len(W), len(R), numSamples) )

    IN.getSNPIterator()

    if verbose:
        print IN.N,"individuals. Beginning to process",IN.numSNPs,"SNPs."
    if L==0: L=IN.numSNPs
    t = time()
    index = 0
    win = deque()
    Sig22I = None
    ZMat = None
    for i in range(L):
        if( i%10 == 0 ): print "At SNP", i, "time spent:", time()-t
        snp,chrm,id,pos = IN.next()
        af = np.mean(snp)
        if (af < minMAF) or (af > 1- minMAF):
            continue

        rVals = [ju.corr(x,snp,IN.P) for x in win]
        try:
            ZVals, SigI_next = ju.sampleNorm(Sig22I, rVals, ZMat,
                                             numSamples, wr)
        except np.linalg.LinAlgError:
            continue
        if len(win) > 0:
            JVals = np.array([EZJVwrapper(Z,r,ZVals,R,rMax)
                              for Z,r in zip(ZMat[:wt,], rVals[:wt])])
            JMaxSlide = np.array([np.max(JVals[:w,:,:], axis = 0) for w in W])
            JMaxStats = np.maximum( JMaxStats, JMaxSlide )
        win.appendleft(snp)
        if index == 0:
            ZMat = np.array([ZVals])
        else:
            ZMat = np.append([ZVals],ZMat,axis=0)
            #ZMat = np.vstack( (ZVals, ZMat) )
        if( len(win) > wr ):
            win.pop()
            ZMat = ZMat[:-1,:]
        Sig22I = SigI_next
        ZMaxStats = np.maximum( ZMaxStats, abs(ZVals) )
        index += 1

    print "Completed", index, "tests. Saving output."

    ZpVals = 2*(1 - stats.norm.cdf(ZMaxStats))
    JpVals = pd.Panel([[np.minimum(1 - stats.chi2.cdf(y, 2), ZpVals) for y in x]
                       for x in JMaxStats], items = W, major_axis = R)
    ZpVals = pd.DataFrame(ZpVals)
    return (ZpVals, JpVals)

#### Used in collect, just here for testing ####
# index = int( np.round( (alpha)*(numSamples-1) ))
# sortZpVals = np.sort( ZpVals )
# sortJpVals = pd.Panel( np.sort( JpVals ), items = JpVals.items,
# major_axis=JpVals.major_axis, minor_axis = range(numSamples) )
# alphaC_Z = sortZpVals[index]
# #alphaC_J = sortJpVals[:,index]
# alphaC_J = sortJpVals.minor_xs(index)
# effNumTests_Z = alpha/alphaC_Z
# effNumTests_J = alpha/alphaC_J
# pd.set_option('display.float_format', lambda x: '%.3g' % x)
# print alphaC_J
# print alphaC_Z
####
