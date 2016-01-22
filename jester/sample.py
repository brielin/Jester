import numpy as np
import scipy as sp
import pandas as pd
from collections import deque
from scipy import stats
from time import time
from . import util as ju
from . import sliding_window as sw
from IPython import embed

def JVwrapper(Z,r,ZVals,R,rMax,LMO):
    if r**2 < rMax:
        if LMO:
            return ju.JVLM( Z, ZVals, r, R )
        else:
            return ju.EZJV( Z, ZVals, r, R )
    else:
        return np.zeros((len(R), len(Z)))

# def TW_wrapper(Z, ZVals, r, rtw):
#     if r**2 < rMax:
#         if LMO:
#             return ju.JVLM2( Z, ZVals, r).reshape((1,len(Z)))
#         else:
#             return ju.EZJV2( Z, ZVals, r).reshape((1,len(Z)))
#     else:
#         return np.zeros((1,len(Z)))

def sample(IN, wt=100, wr=100, wStep=0, rMin=0.0, rMax=1.0,rRange=False,
           numSamples=1000, minMAF=0.05, twoWindows=False, verbose=False,
           LMO=False, from_bp=None, to_bp=None, L=0):
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
    tmin = stats.t.ppf(1-5e-08,IN.N-2)
    rtw = tmin/np.sqrt(IN.N-2+tmin**2)
    t = time()
    index = 0
    win = deque()
    Sig22I = None
    ZMat = None
    WIN = sw.sliding_window(win_len=wr,arr_len=10*wr,vec_len=numSamples)
    for i in range(L):
        if (i%100 == 0) and verbose: print "At SNP", i, "time spent:", time()-t
        snp,chrm,id,pos = IN.next()
        if (from_bp is not None) and (pos < from_bp):
            continue
        if (to_bp is not None) and (pos > to_bp):
            continue
        af = np.mean(snp)
        if (af < minMAF) or (af > 1- minMAF):
            #print i, af, id
            continue
        rVals = np.array([ju.corr(x,snp,IN.P) for x in win])
        rVkeep = (rVals>rtw)|(rVals<-rtw)
        try:
            ZVals, SigI_next = ju.sampleNorm(Sig22I, rVals, ZMat,
                                             numSamples, wr)
        except np.linalg.LinAlgError:
            #print i,"skipping",id, np.max(rVals)
            continue
        # ##
        # if len(rVals)==0:
        #     winCor=np.array([[1]])
        # elif len(rVals)==1:
        #     winCor = np.array([[1.0,float(rVals)],[float(rVals),1]])
        # elif len(rVals)>1:
        #     winCor = np.vstack((np.hstack((1.0,rVals)),np.hstack((np.array([rVals]).T,winCor))))
        #     if winCor.shape[0] > wr:
        #         winCor=winCor[0:wr,0:wr]
        ##
        if len(win) > 0:
            JVals = np.zeros((len(win),len(R),numSamples))
            for j,(Z,r) in enumerate(zip(ZMat[:wt,], rVals[:wt])):
                JVals[j] = JVwrapper(Z,r,ZVals,R,rMax,LMO)#dim=(len(R),nsamples)
            if twoWindows:
                for j,(Z,r) in enumerate(zip(ZMat[wt:,], rVals[wt:])):
                    JVals[j+wt] = JVwrapper(Z,r,ZVals,R,rMax,LMO)
                JMaxSlide = []
                rw = rVkeep.copy() #Can do this since w is increasing
                JMaxSlide = np.zeros(JMaxStats.shape)
                for i,w in enumerate(W):
                    rw[:w] = True
                    JMaxSlide[i]=(np.max(JVals[np.array(rw),:,:],axis=0))
            else:
                JMaxSlide = np.array([np.max(JVals[:w,:,:], axis = 0)
                                      for w in W])
            JMaxStats = np.maximum( JMaxStats, JMaxSlide )
        win.appendleft(snp)
        ZMat = WIN.next(ZVals)
        if( len(win) > wr ): win.pop()
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
