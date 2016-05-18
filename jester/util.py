from __future__ import division
import numpy as np
import statsmodels.api as sm
import bottleneck as bn
from IPython import embed


def _reorder(X,iids):
    D={}
    for i,x in enumerate(X.values):
        D[tuple(x[0:2])]=i
    order=[]
    for i in iids:
        order.append(D[tuple(i)])
    return X.values[order,2:].astype(float)

def _fast_var(X,m):
    return ((X - m)**2).mean(0)

def _impute_missing(X):
    m = bn.nanmean(X,0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(m,inds[1])
    return X

def _norm_data_in_place(X):
    m = np.mean(X,0)
    np.subtract(X,m,out=X)
    v = X.var(0)
    np.divide(X,np.sqrt(v),out=X)

def _norm_data(X):
    m = np.mean(X,0)
    Y = np.subtract(X,m)
    v = Y.var(0)
    Y = np.divide(Y,np.sqrt(v),out=Y)
    return Y

def get_allele_frequency(bed,args):
    s = args.SNPs_to_read
    af = np.zeros((bed.sid_count))
    var = np.zeros((bed.sid_count))
    if (args.from_bp is not None) and (args.to_bp is not None):
        k0 = np.where((bed.pos[:,2]>=args.from_bp))[0][0]
        k1 = np.where((bed.pos[:,2]<=args.to_bp))[0][-1]
        X = bed[:,k0:k1].read().val
        af[k0:k1] = bn.nanmean(X,0)/2.0
        var[k0:k1] = _fast_var(X,2*af[k0:k1])
    else:
        for i in xrange(int(np.ceil(bed.sid_count/s))):
            X = bed[:,i*s:(i+1)*s].read().val
            af[i*s:(i+1)*s] = bn.nanmean(X,0)/2.0
            var[i*s:(i+1)*s] = _fast_var(X,2*af[i*s:(i+1)*s])
    af[var==0]=0
    return af

def get_windows(pos,M,window_size,window_type):
    if window_type == 'BP':
        coords = pos[:,2]
        ws = 1000*window_size
    elif window_type == 'SNP':
        coords = np.array(range(M))
        ws = window_size
    wl = []
    wr = []
    j=0
    for i in xrange(M):
        while j<M and abs(coords[j]-coords[i])<ws:
            j+=1
        wl.append(i)
        wr.append(j)
    return np.array([wl,wr]).T
