import sys
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

def JT( b1, b2, p1, p2, r):
    chi1 = stats.chi2.ppf( 1-p1, 1 )
    chi2 = stats.chi2.ppf( 1-p2, 1 )
    Z1 = np.sign(b1)*np.sqrt(chi1)
    Z2 = np.sign(b2)*np.sqrt(chi2)
    chiJ = (Z1**2 + Z2**2 - 2*r*Z1*Z2)/(1-r**2)
    pJ = 1-stats.chi2.cdf(chiJ,2)
    return chiJ,pJ

def dflip(a1,a2,maf1,b1,b2,baf):
    # Either SNPs are same, complemented, flipped, or flipped and complimented
    # otherwise alleles don't match and we skip this SNP
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    try:
        if not(((a1==b1) and (a2==b2)) or ((a1==comp[b1]) and (a2==comp[b2])) or\
                   ((a1==b2) and (a2 == b1)) or ((a1==comp[b2]) and (a2==comp[b1]))):
            sys.stderr.write("bad SNP\n")
            sys.stderr.write(a1+'\t'+a2+'\n'+b1+'\t'+b2+'\n')
            return 0
        elif comp[a1] == a2: # If alt allele is complement need to handle specially
            d1 = abs(maf1 - 0.5)
            d2 = abs(baf - 0.5)
            if d1 < 0.1 or d2 < 0.1:
                return 0
            elif np.sign(maf1 - 0.5) != np.sign(baf - 0.5): # minor flipped
                return -1
            else: return 1
        elif (a1 == b2 and a2 == b1) or (a1 == comp[b2] and a2 == comp[b1]):
            return -1
        else:
            return 1
    except KeyError:
        sys.stderr.write("bad SNP\n")
        sys.stderr.write(a1+'\t'+a2+'\n'+b1+'\t'+b2+'\n')
        return 0

def test_summary(IN, rFile = None, frqFile = None,  wt=100, rMin=0.0, rMax=1.0,
                 verbose=False, minp=1e-5, L=0):
    chimin = stats.chi2.ppf(1-minp,2)
    IN.getSNPIterator()
    cor_df = pd.read_csv(rFile, index_col=0, sep='\t')
    if frqFile is None:
        sys.stdout.write("No frq file provided. Assuming SNPs with an AF above"
                         " 0.5 need to be flipped")
    else:
        frq_df = pd.read_table(frqFile,sep='\s*',index_col=1)
        if np.sum(cor_df.index != frq_df.index):
            sys.stderr.write("Correlation file SNPs do not match frqfile SNPs.")
            sys.exit(1)
    if verbose:
        print IN.N,"individuals. Beginning to process",IN.numSNPs,"SNPs."
        print "Read correlations for", cor_df.shape[0],"SNPs with window size",\
        cor_df.shape[1]
    if IN.numSNPs != cor_df.shape[0]:
        sys.stderr.write("SNPs in input summary statistics file and provided "
                         "correlation file do not match. Exiting\n")
        sys.exit(1)
    joint_res = []
    jointResCols=['chr','rsid1','rsid2','dist','pos1','pos2','af1','af2','corr',
              'beta1_M','pval1_M','beta2_M','pval2_M','beta1_J','beta1_J_se',
              'or1_J','t1_J','pval1_J','beta2_J','beta2_J_se','or2_J','t2_J',
              'pval2_J','Chi2_J','pval_J','beta1_X','beta1_X_se',
              'or1_X','t1_X','pval1_X','beta2_X','beta2_X_se','or2_X','t2_X',
              'pval2_X','beta3_X','beta3_X_se','or3_X','t3_X','pval3_X',
              'Chi2_X','pval_X']
    if L==0: L = IN.numSNPs
    i = 0 # Tracks which SNP # we're at
    index = 0 # Tracks how many tests we've done
    t=time()
    store = deque()
    for i in range(L):
        if (verbose & (i%100 == 0)):
            print "At SNP", i, "time spent:", time()-t
        chrm1,id1,pos1,maf1,a1,a2,nCase1,nCont1,beta1,beta1SE,pV1 = IN.next()
        cor = cor_df.iloc[i]
        cor_index = cor_df.index[i]
        if cor_index != id1:
            sys.stderr.write("Mismatch in cor SNP ID and summary stats ID at"
                             " position "+str(i)+'\n')
        b1,b2,baf = frq_df.loc[cor_index,[1,2,3]]
        flip = dflip(a1,a2,maf1,b1,b2,baf)
        beta1 = flip*beta1
        if flip == 0:
            pV1 = -1
        for dist,(r,(chrm2,id2,pos2,maf2,nCase2,nCont2,beta2,beta2SE,pV2)) in \
                enumerate(zip(cor,store)):
            n = np.min((nCase1+nCont1,nCase2+nCont2))
            cf1 = nCase1/(nCase1+nCont1)
            cf2 = nCase2/(nCase2+nCont2)
            varY = (cf1*(1-cf1)+cf2*(1-cf2))/2.0
            den = 1.0/(1-r**2)
            hv1 = 2*n*maf1*(1.0-maf1)
            hv2 = 2*n*maf2*(1.0-maf2)
            hcv = np.sqrt(hv1*hv2)*r
            XXTIhat = np.array([[hv2,hcv],[hcv,hv1]])/(hv1*hv2-hcv**2)

            B1J = den*(beta1 - r*beta2)
            B2J = den*(beta2 - r*beta1)

            resid = varY*(n/(n-2)) - (B1J*hv1*beta1 + B2J*hv2*beta2)/(n-2)
            varBJ = resid*XXTIhat
            B1JSE = np.sqrt(varBJ[0,0])
            B2JSE = np.sqrt(varBJ[1,1])
            OR1J = np.exp(B1J)
            OR2J = np.exp(B2J)
            T1J = B1J/B1JSE
            T2J = B2J/B2JSE
            pV1J = 1-stats.t.cdf(T1J, n-2)
            pV2J = 1-stats.t.cdf(T2J, n-2)
            chiJ, pJ = JT(beta1, beta2, pV1, pV2, r)

            res = [chrm1, id1, id2, dist, pos1, pos2, maf1, maf2, r, beta1,
                   pV1, beta2, pV2, B1J, B1JSE, OR1J, T1J, pV1J, B2J, B2JSE,
                   OR2J, T2J, pV2J, chiJ, pJ,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,
                   -9, -9 ,-9, -9,-9 ,-9]
            index +=1
            joint_res.append(res)
        store.appendleft((chrm1,id1,pos1,maf1,nCase1,nCont1,beta1,beta1SE,pV1))
        if( len(store) > wt ): store.pop()
    joint_resDF = pd.DataFrame(joint_res,columns=jointResCols)
    joint_resDF[['dist','pos1','pos2']] = \
        joint_resDF[['dist','pos1','pos2']].astype(int)
    return joint_resDF
