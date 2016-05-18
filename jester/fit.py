import numpy as np
import scipy as sp
import pandas as pd
import sys
import statsmodels.api as sm
from pysnptools.snpreader import Bed
from pysnptools.standardizer import Unit
from scipy import stats
from scipy import optimize
from math import exp
from time import time
from . import util as ju
from IPython import embed

class fit(object):
    def __init__(self,args):
        self.bed = Bed(args.bfile) #
        self.N = self.bed.iid_count
        if args.covfile is not None:
            cov = pd.read_table(args.covfile,header=None)
            self.cov = sm.add_constant(ju._reorder(cov,self.bed.iid))
            self.ncov = self.cov.shape[1] # + constant
        else:
            self.cov = np.ones((self.N,1))
            self.ncov = 1 # Constant
        if args.phenofile is not None:
            Y = pd.read_table(args.phenofile,header=None,na_values='-9')
        else:
            try:
                Y = pd.read_table(args.bfile+'.pheno',header=None,na_values='-9')
            except IOError:
                print("Phenotype file not found.")
                exit(1)
        self.Y = ju._reorder(Y,self.bed.iid)
        af = ju.get_allele_frequency(self.bed,args) #
        snps = (af>args.maf)&(af<1-args.maf) #
        if (args.from_bp is not None) and (args.to_bp is not None):
            k = (bed.pos[:,2]>args.from_bp)&(bed.pos[:,2]<args.to_bp)
            snp1 = snps&k
        snps_to_use = self.bed.sid[snps]
        if args.extract is not None:
            keep = np.array([l.strip() for l in open(args.extract,'r')])
            snps_to_use = np.intersect1d(snps_to_use,keep)
        self.bed_index = np.sort(self.bed.sid_to_index(snps_to_use)) #
        pos = self.bed.pos[self.bed_index] #
        bim=pd.read_table(self.bed.filename+'.bim',header=None,
                          names=['chm','id','pos_mb','pos_bp','a1','a2'])
        self.af = af[self.bed_index] #
        self.M = len(self.bed_index) #
        self.windows = ju.get_windows(pos,self.M,args.window_size,args.window_type)
        self.pos = pos[:,2]
        self.chr = pos[:,0]
        self.id = self.bed.sid[self.bed_index]
        self.A1 = bim['a1'].loc[self.bed_index]
        self.A2 = bim['a2'].loc[self.bed_index]
        self.logistic = False
        self.chimin = stats.chi2.ppf(1-args.minp,2)

        # Fit null
        if (not args.linear) and (self.Y.min() >= 0 and self.Y.max() <= 1):
            self.null = sm.Logit(self.Y, self.cov, missing='drop').fit(disp=0)
            self.logistic = True
        else:
            self.null = sm.OLS(self.Y, self.cov, missing='drop').fit(disp=0)
        if self.ncov > 1:
            self.cov = sm.add_constant(self.null.fittedvalues)
        self.marg_res, self.joint_res = self.compute(args)

    def compute(self,args):
        t=time()
        marg_res = []
        joint_res = []
        Z = []
        windex = 0
        li,ri = self.windows[windex]
        nstr = np.max((args.SNPs_to_read,ri-li))
        offset = li
        G = self.bed[:,self.bed_index[li:(li+nstr)]].read().val
        G = ju._impute_missing(G) # replace missing with mean
        self.compute_marg(marg_res,Z,G,li,args)
        A = ju._norm_data(G)
        while ri < offset+nstr:
            st = li-offset
            fi = ri-offset
            # All correlations of SNP j with SNPs in its window
            R = np.dot(np.atleast_2d(A[:,st]/self.N),A[:,(st+1):fi]).flatten()
            Zl = Z[li]
            Zr = np.array(Z[(li+1):ri])
            # Use marginal Z-scores and R to compute expected joint chi2s
            ChiP = (1/(1-R**2))*(Zl**2+Zr**2-2*R*Zl*Zr)
            ChiP[R**2 < args.r2min] = -1
            self.compute_joint(joint_res,G,ChiP,offset,li,ri,args)
            windex += 1
            li,ri = self.windows[windex]
        for i in xrange(offset+nstr,self.M,nstr):
            sys.stdout.flush()
            sys.stdout.write("SNP: %d, %f\r" % (i, time()-t))
            Gn = self.bed[:,self.bed_index[i:(i+nstr)]].read().val
            Gn = ju._impute_missing(Gn)
            An = ju._norm_data(Gn)
            self.compute_marg(marg_res,Z,Gn,i,args)
            G = np.hstack((G,Gn))
            A = np.hstack((A,An))
            if G.shape[1] > args.SNPs_to_store:
                G = G[:,nstr:]
                A = A[:,nstr:]
                offset += nstr
            while ri < i+nstr:
                st = li-offset
                fi = ri-offset
                # All correlations of SNP j with SNPs in its window
                R = np.dot(np.atleast_2d(A[:,st]/self.N),A[:,(st+1):fi]).flatten()
                Zl = Z[li]
                Zr = np.array(Z[(li+1):ri])
                ChiP = (1/(1-R**2))*(Zl**2+Zr**2-2*R*Zl*Zr)
                ChiP[R**2 < args.r2min] = -1
                self.compute_joint(joint_res,G,ChiP,offset,li,ri,args)
                try:
                    windex += 1
                    li,ri = self.windows[windex]
                except IndexError:
                    break
        marg_res = pd.DataFrame(marg_res)
        joint_res = pd.DataFrame(joint_res)
        return marg_res, joint_res

    def compute_joint(self,joint_res,G,ChiP,offset,li,ri,args):
        st = li-offset
        fi = ri-offset
        snp1 = G[:,st]
        for i,snp2 in enumerate(G[:,(st+1):fi].T):
            if ChiP[i] > self.chimin:
                X = np.hstack((self.cov,snp1.reshape((len(snp1),1)),
                               snp2.reshape((len(snp2),1))))
                if self.logistic:
                    joint = sm.Logit(self.Y,X).fit(disp=0)
                else:
                    joint = sm.OLS(self.Y,X).fit(disp=0)
                joint_b1 = joint.params[-2]
                joint_b2 = joint.params[-1]
                joint_or1 = np.exp(joint_b1)
                joint_or2 = np.exp(joint_b2)
                joint_se1 = joint.bse[-2]
                joint_se2 = joint.bse[-1]
                joint_p1 = joint.pvalues[-2]
                joint_p2 = joint.pvalues[-1]
                joint_t1 = joint.tvalues[-2]
                joint_t2 = joint.tvalues[-1]
                joint_Chi2 = 2*(joint.llf - self.null.llf)
                pv = stats.chi2.sf(joint_Chi2,2)
                joint_res.append([self.chr[li],self.id[li],self.id[offset+i],
                                  self.pos[li],self.pos[offset+i],joint_b1,joint_se1,
                                  joint_or1,joint_t1,joint_p1,joint_b2,joint_se2,
                                  joint_or2,joint_t2,joint_p2,joint_Chi2,pv])
            else:
                continue

    def compute_marg(self,marg_res,Z,G,offset,args):
        for i,snp in enumerate(G.T):
            X = np.hstack((self.cov,snp.reshape((len(snp),1))))
            if self.logistic:
                marg = sm.Logit(self.Y,X).fit(disp=0)
            else:
                marg = sm.OLS(self.Y,X).fit(disp=0)
            marg_b = marg.params[-1]
            marg_or = np.exp(marg_b)
            marg_se = marg.bse[-1]
            marg_p = marg.pvalues[-1]
            marg_t = marg.tvalues[-1]
            marg_Chi2 = 2*(marg.llf - self.null.llf)
            pv = stats.chi2.sf(marg_Chi2,1)
            marg_res.append([self.chr[offset+i],self.id[offset+i],
                             self.pos[offset+i],marg_b, marg_se, marg_or, marg_t,
                             marg_p,marg_Chi2,pv])
            Z.append(marg_b/marg_se)
