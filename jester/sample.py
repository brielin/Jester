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


class sample(object):
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
        self.sample_windows = ju.get_windows(pos,self.M,args.sample_window_size,
                                             args.sample_window_type)
        self.pos = pos[:,2]
        self.chr = pos[:,0]
        self.id = self.bed.sid[self.bed_index]
        self.A1 = bim['a1'].loc[self.bed_index]
        self.A2 = bim['a2'].loc[self.bed_index]
        self.numSamples = args.numSamples
        self.JMaxStats, self.ZMaxStats = self.sample(args)
        self.JMinP = stats.chi2.sf(self.JMaxStats,2)
        self.ZMinP = stats.chi2.sf(self.ZMaxStats**2,1)
        self.minP = np.minimum(self.JMinP,self.ZMinP)

    def sample(self,args):
        t=time()
        nz = 0
        ZMaxStats = np.zeros((self.numSamples,1))
        JMaxStats = np.zeros((self.numSamples,1))
        windex = 0
        sli,sri = self.sample_windows[windex]
        tli,tri = self.windows[windex]
        nstr = np.max((args.SNPs_to_read,sri-sli))
        offset = sli
        G = self.bed[:,self.bed_index[sli:(sli+nstr)]].read().val
        G = ju._impute_missing(G)
        A = ju._norm_data(G)
        # Sample Z-scores and do joint tests of first window
        R = np.dot(A[:,sli:sri].T/self.N,A[:,sli:sri])
        Z=np.random.multivariate_normal(np.zeros((R.shape[0])),R,args.numSamples)
        nz += R.shape[0]
        zli,zri = sli,sri # position of Z relative to full genotype
        gli,gri = zli,zri # position of Z relative to genotype in memory
        Rp = R[(tli+1):tri,0]
        to_test = Rp**2 > args.r2min
        Rp = Rp[to_test]
        Zl = np.atleast_2d(Z[:,0]).T
        Zr = np.array(Z[:,1:(tri-tli)])[:,to_test]
        ChiP = (1/(1-Rp**2))*(Zl**2+Zr**2-2*Rp*Zl*Zr)
        ZMaxStats = np.atleast_2d(np.hstack((ZMaxStats,abs(Z))).max(1)).T
        # ZMaxStats = np.maximum(ZMaxStats,abs(Z.max(1)))
        # JMaxStats = np.maximum(JMaxStats,ChiP.max(1))
        JMaxStats = np.atleast_2d(np.hstack((JMaxStats,ChiP)).max(1)).T
        # Slide through genotype in memory
        while True:
            windex += 1
            sli,sri = self.sample_windows[windex]
            tli,tri = self.windows[windex]
            if sri >= offset+nstr: break
            tst,tfi,sst,sfi = np.array([tli,tri,sli,sri])-offset
            #print sli, sri, zli, zri, gli, gri, Z.shape[1]
            if zli < sli: # drop zli..sli and update indices
                Z = Z[:,(sli-zli):]
                zli,gli = sli,sst
            if zri < sri: # marginal sample everything from zri..sri
                S = A[:,gli:gri] # G that overlaps Z
                Sn = A[:,gri:sri] # G about to have Z scores sampled
                r12 = S.T.dot(Sn)/self.N
                r11 = Sn.T.dot(Sn)/self.N
                Zn = self.sample_func(Z,S,Sn,r11,r12,args)
                Z = np.hstack((Z,Zn))
                nz += (sri-gri)
                zri,gri = sri,sfi
            ZMaxStats = np.atleast_2d(np.hstack((ZMaxStats,abs(Zn))).max(1)).T
            # ZMaxStats = np.maximum(ZMaxStats,abs(Zn.flatten()))
            # All correlations of SNP tli with SNPs in its window
            # Surely these are already computed and some cleverness
            #  can be used to re-use them but its a fast calculation anyways
            if sri-sli > 1:
                R = np.dot(np.atleast_2d(A[:,tst]/self.N),
                           A[:,(tst+1):tfi]).flatten()
                to_test = R**2 > args.r2min
                R = R[to_test]
                Zl = np.atleast_2d(Z[:,0]).T
                Zr = np.array(Z[:,1:(sri-sli)])[:,to_test]
                ChiP = (1/(1-R**2))*(Zl**2+Zr**2-2*R*Zl*Zr)
                JMaxStats = np.atleast_2d(np.hstack((JMaxStats,ChiP)).max(1)).T
                #JMaxStats = np.maximum(JMaxStats,ChiP.max(1))
        for i in xrange(offset+nstr,self.M,nstr):
            sys.stdout.flush()
            sys.stdout.write("SNP: %d, %f\r" % (i, time()-t))
            Gn = self.bed[:,self.bed_index[i:(i+nstr)]].read().val
            Gn = ju._impute_missing(Gn)
            An = ju._norm_data(Gn)
            G = np.hstack((G,Gn))
            A = np.hstack((A,An))
            if G.shape[1] > args.SNPs_to_store:
                G = G[:,nstr:]
                A = A[:,nstr:]
                offset += nstr
                gli -= nstr
                gri -= nstr
            while sri < i+nstr:
                tst,tfi,sst,sfi = np.array([tli,tri,sli,sri])-offset
                if zli < sli: # drop zli..sli and update indices
                    Z = Z[:,(sli-zli):]
                    zli,gli = sli,sst
                if zri < sri: # marginal sample everything from zri..sri
                    S = A[:,gli:gri] # G that overlaps Z
                    Sn = A[:,gri:sfi] # G about to have Z scores sampled
                    r12 = S.T.dot(Sn)/self.N
                    r11 = Sn.T.dot(Sn)/self.N
                    Zn = self.sample_func(Z,S,Sn,r11,r12,args)
                    Z = np.hstack((Z,Zn))
                    nz += (sfi-gri)
                    zri,gri = sri,sfi
                ZMaxStats = np.atleast_2d(np.hstack((ZMaxStats,abs(Zn))).max(1)).T
                # ZMaxStats = np.maximum(ZMaxStats,abs(Zn.flatten()))
                if sri-sli > 1:
                    R = np.dot(np.atleast_2d(A[:,tst]/self.N),
                               A[:,(tst+1):tfi]).flatten()
                    to_test = R**2 > args.r2min
                    R = R[to_test]
                    Zl = np.atleast_2d(Z[:,0]).T
                    Zr = np.array(Z[:,1:(sri-sli)])[:,to_test]
                    ChiP = (1/(1-R**2))*(Zl**2+Zr**2-2*R*Zl*Zr)
                    #JMaxStats = np.maximum(JMaxStats,ChiP.max(1))
                    JMaxStats = np.atleast_2d(np.hstack((JMaxStats,ChiP)).max(1)).T
                try:
                    windex += 1
                    sli,sri = self.sample_windows[windex]
                    tli,tri = self.windows[windex]
                except IndexError:
                    break
        # print "HERE:", nz
        return JMaxStats.flatten(), ZMaxStats.flatten()

    def sample_func(self,Z,S,Sn,r11,r12,args):
        S22IS12 = sp.linalg.lstsq(S,Sn,cond=1e-8)[0]
        muC = Z.dot(S22IS12)
        SigC = r11-r12.T.dot(S22IS12)
        Zn = np.random.multivariate_normal(np.zeros((SigC.shape[0])),
                                              SigC,size=args.numSamples)
        Zn += muC
        return Zn
