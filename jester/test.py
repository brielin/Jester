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

def test(IN, wt=100, rMin=0.0, rMax=1.0, verbose=False, crossTest=False,
         minp=1e-05, L=0):
    chimin = stats.chi2.ppf(1-minp,2)
    IN.getSNPIterator()
    if verbose:
        print IN.N,"individuals. Beginning to process",IN.numSNPs,"SNPs."

    if verbose: print "Fitting null model."
    Y = IN.phenos
    print len(Y), "phenotypes found.", sum(np.isnan(Y))[0], "missing."
    null = sm.OLS( Y, IN.cov, missing='drop' ).fit(disp=0)
    # null = sm.Logit( Y, IN.cov, missing='drop' ).fit(disp=0)
    print "Done."

    ncov = IN.cov.shape[1]
    margResCols = ['chr','rsid','pos','beta','beta_se','or',
                  't_stat','Chi2','p_val']
    jointResCols=['chr','rsid1','rsid2','dist','pos1','pos2','af1','af2','corr',
              'beta1_M','pval1_M','beta2_M','pval2_M','beta1_J','beta1_J_se',
              'or1_J','t1_J','pval1_J','beta2_J','beta2_J_se','or2_J','t2_J',
              'pval2_J','Chi2_J','pval_J','beta1_X','beta1_X_se',
              'or1_X','t1_X','pval1_X','beta2_X','beta2_X_se','or2_X','t2_X',
              'pval2_X','beta3_X','beta3_X_se','or3_X','t3_X','pval3_X',
              'Chi2_X','pval_X']
    chiResCols = ['chr','rsid1','rsid2','dist','corr','chi1','chi2',
                'chi2v1','chi1v2','chiJ']
    marg_res = []
    joint_res = []
    chi_res = []
    if L==0: L = IN.numSNPs
    i = 0 # Tracks which SNP # we're at
    index = 0 # Tracks how many tests we've done
    t = time()
    win = deque()
    store = deque()
    for i in range(L):
        if (verbose & (i%100 == 0)):
            print "At SNP", i, "time spent:", time()-t
        snp,chrm,id,pos = IN.next()
        af = np.mean(snp)
        # Do Marginal Test
        X = np.hstack((IN.cov,snp.reshape((len(snp),1))))
        marg = sm.OLS(Y, X, missing='drop').fit(disp=0)
        #marg = sm.Logit(Y, X, missing='drop').fit(disp=0)
        marg_b = marg.params[ncov]
        marg_or = np.exp(marg.params[ncov])
        marg_se = marg.bse[ncov]
        marg_p = marg.pvalues[ncov]
        marg_t = marg.tvalues[ncov]
        marg_Chi2 = 2*(marg.llf - null.llf)
        marg_res.append([chrm,id,pos,marg_b, marg_se, marg_or, marg_t,
                        marg_Chi2, marg_p])
        for dist,(w,(id2,pos2,af2,marg2)) in enumerate(zip(win,store)):
            t0=time()
            #r = ju.corr(w,snp,IN.P,missing='drop',pheno=Y)
            r = np.corrcoef(w,snp)[0,1]
            #print time()-t0
            Z1 = np.sign(marg_b)*stats.norm.ppf(1-marg_p/2)
            ## NEED THE OTHER Z
            #Z2h = r*Z1 + np.sqrt( -chimin*r**2 + chimin + r**2*Z1**2 -Z1**2)
            #Z2l = r*Z1 - np.sqrt( -chimin*r**2 + chimin + r**2*Z1**2 -Z1**2)
            resNA = [chrm,id,id2,dist+1,pos,pos2,af,af2,r,marg_b,marg_p,
                     marg2.params[ncov],marg2.pvalues[ncov], -9, -9, -9, -9, -9,
                     -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
                     -9, -9, -9, -9, -9, -9, -9, -9, -9]
            Z2 = np.sign(marg2.params[ncov])*\
                stats.norm.ppf(1-marg2.pvalues[ncov]/2.0)
            chi1 = Z1**2
            chi2 = Z2**2
            if r < 1.0:
                chiJ = (Z1**2 + Z2**2 - 2*r*Z1*Z2)/(1-r**2)
            else:
                chiJ = Z1**2
            chi1v2 = chiJ - chi2
            chi2v1 = chiJ - chi1
            chi_res.append([chrm,id,id2,dist+1,r,chi1,chi2,chi2v1,chi1v2,chiJ])
            if chiJ > chimin:
                print Z1, Z2, r, chiJ
            if( (r**2 > rMax) | (r**2 < rMin)):
                res = resNA
            elif chiJ < chimin:
                res = resNA
            else:
                print i, dist
                X = np.hstack(( IN.cov, snp.reshape((len(snp),1)),
                                w.reshape((len(w),1)),
                                (snp*w).reshape((len(snp),1)) ))
                try:
                    joint = sm.OLS(Y,  X[:,:-1], missing='drop').fit(disp=0)
                    #joint = sm.Logit(Y,  X[:,:-1], missing='drop').fit(disp=0)
                    joint_b1 = joint.params[ncov]
                    joint_or1 = np.exp( joint.params[ncov] )
                    joint_se1 = joint.bse[ncov]
                    joint_p1 = joint.pvalues[ncov]
                    joint_t1 = joint.tvalues[ncov]
                    joint_b2 = joint.params[ncov+1]
                    joint_or2 = np.exp( joint.params[ncov+1] )
                    joint_se2 = joint.bse[ncov+1]
                    joint_p2 = joint.pvalues[ncov+1]
                    joint_t2 = joint.tvalues[ncov+1]
                    joint_n = joint.nobs
                    joint_Chi2 = 2*(joint.llf - null.llf)
                    joint_pv = 1 - stats.chi2.cdf( joint_Chi2, 2 )
                    if( crossTest ):
                        cross = sm.OLS( Y,  X, missing = 'drop' ).fit(disp=0)
                        #cross = sm.Logit( Y,  X, missing = 'drop' ).fit(disp=0)
                        cross_b1 = cross.params[ncov]
                        cross_or1 = np.exp( cross.params[ncov] )
                        cross_se1 = cross.bse[ncov]
                        cross_p1 = cross.pvalues[ncov]
                        cross_t1 = cross.tvalues[ncov]
                        cross_b2 = cross.params[ncov+1]
                        cross_or2 = np.exp( cross.params[ncov+1] )
                        cross_se2 = cross.bse[ncov+1]
                        cross_p2 = cross.pvalues[ncov+1]
                        cross_t2 = cross.tvalues[ncov+1]
                        cross_bx = cross.params[ncov+2]
                        cross_orx = np.exp( cross.params[ncov+2] )
                        cross_sex = cross.bse[ncov+2]
                        cross_px = cross.pvalues[ncov+2]
                        cross_tx = cross.tvalues[ncov+2]
                        cross_Chi2 = 2*(cross.llf - null.llf)
                        cross_pv = 1 - stats.chi2.cdf( cross_Chi2, 3 )
                    else:
                        (cross_b1,cross_or1,cross_se1,cross_p1,cross_t1,
                         cross_b2,cross_or2,cross_se2,cross_p2,cross_t2,
                         cross_bx,cross_orx,cross_sex,cross_px,cross_tx,
                         cross_Chi2,cross_pv) = [-1 for ii in range(17)]
                    res = [chrm,id,id2,dist+1,pos,pos2,af,af2,r,marg_b,marg_p,
                           marg2.params[ncov],marg2.pvalues[ncov],joint_b1,
                           joint_se1,joint_or1,joint_t1,joint_p1, joint_b2,
                           joint_se2,joint_or2,joint_t2,joint_p2,joint_Chi2,
                           joint_pv,cross_b1,cross_se1,cross_or1,cross_t1,
                           cross_p1,cross_b2,cross_se2,cross_or2,cross_t2,
                           cross_p2,cross_bx,cross_sex,cross_orx,cross_tx,
                           cross_px,cross_Chi2,cross_pv]
                    index += 1
                    joint_res.append(res)
                except np.linalg.linalg.LinAlgError as err:
                    print "Error encountered in a joint test of SNP", i, "and",\
                        i-(dist+1),". Continuing"
                    res = resNA
        win.appendleft(snp)
        store.appendleft((id,pos,af,marg))
        if( len(win) > wt ):
            win.pop()
            store.pop()
    if verbose:
        print (index+1),"tests done in: ",time() - t,"seconds. Saving output."
    if len(joint_res) > 0:
        joint_resDF = pd.DataFrame(joint_res,columns=jointResCols)
    else:
        joint_resDF = pd.DataFrame(columns=jointResCols)
    marg_resDF = pd.DataFrame(marg_res,columns=margResCols)
    chi_resDF = pd.DataFrame(chi_res,columns=chiResCols)
    chi_resDF['dist'] = chi_resDF['dist'].astype(int)
    # Need to set the type so that it isn't cut off if a float format is set
    joint_resDF[['dist','pos1','pos2']] = \
        joint_resDF[['dist','pos1','pos2']].astype(int)
    return joint_resDF, marg_resDF, chi_resDF
