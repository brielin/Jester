import numpy as np
import scipy as sp
import statsmodels.api as sm
from IPython import embed

### FUNCTIONS for MTC computation ###
def sampleReg(snp, SMat, Sig12, ZMat, numSamples):
    if SMat is None :
        ZVals = np.random.normal(size=numSamples)
    else:
        S22IS12 = sp.linalg.lstsq(SMat.T,snp,cond=1e-8)[0]
        muC = ZMat.T.dot(S22IS12)
        SigC = 1-Sig12.dot(S22IS12)
        if SigC <= 0:
            ZVals=muC
        else:
            ZVals = np.random.normal(loc=muC, scale=np.sqrt(SigC),
                                     size=numSamples)
    return ZVals

def sampleNorm( Sig22I, rVals, ZMat, numSamples, wr):
    zerTol = 1e-8
    if Sig22I is None :
        ZVals = np.random.normal( size = numSamples )
        SigI_next = 1 # Not used, needs to be set for return val
    elif len(rVals) == 1:
        Sig12 = rVals[0]
        muC = Sig12*ZMat
        Sig2C = 1 - Sig12**2
        ZVals = np.random.normal(loc = muC, scale = np.sqrt(Sig2C),
                                 size = numSamples)
        # Invert the matrix for  3 SNP sampling by hand
        SigI_next = (1.0/Sig2C)*np.array([[1,-Sig12],[-Sig12,1]])
    else:
        Sig11 = 1
        Sig12 = np.array([rVals])
        ## Use inverse of Sig22 with Sig12 to get conditional mean and var
        SR = Sig22I.dot( Sig12.T )
        den = 1 - Sig12.dot( SR )
        Int = Sig22I.dot( ZMat )
        muC = Sig12.dot( Int )
        Sig2C = den
        # ## Rank one updates of Sig22 inverse to include next SNP
        SC = 1.0/den
        B = -SC*SR
        SigI_next = np.vstack((np.hstack((SC, B.T)),
                               np.hstack((B, Sig22I - SR.dot(B.T) )) ))
        if( SigI_next.shape[0] > wr ):
            E = SigI_next[0:wr, 0:wr]
            f = SigI_next[0:wr, wr:(wr+1)]
            h = SigI_next[wr,wr]
            SigI_next = E - (f.dot(f.T))/h
        if( Sig2C < zerTol ):
            #ZVals = muC.reshape((numSamples))
            raise np.linalg.LinAlgError('Singular Matrix')
        else:
            ZVals = np.random.normal(loc = muC, scale = np.sqrt( Sig2C ),
                                 size = numSamples)
    return ZVals, SigI_next

def sampleNormNoInv(Sig22I, rVals, ZMat, numSamples, wr):
    zerTol = 1e-8
    if Sig22I is None :
        ZVals = np.random.normal( size = numSamples )
        SigI_next = 1 # Not used, needs to be set for return val
    elif len(rVals) == 1:
        Sig12 = rVals[0]
        muC = Sig12*ZMat
        Sig2C = 1 - Sig12**2
        ZVals = np.random.normal(loc = muC, scale = np.sqrt(Sig2C),
                                 size = numSamples)
    else:
        Sig11 = 1
        Sig12 = np.array([rVals])
        ## Use inverse of Sig22 with Sig12 to get conditional mean and var
        SR = Sig22I.dot( Sig12.T )
        den = 1 - Sig12.dot( SR )
        Int = Sig22I.dot( ZMat )
        muC = Sig12.dot( Int )
        Sig2C = den
        if( Sig2C < zerTol ):
            print Sig2C
            ZVals = muC.reshape((numSamples))
            #raise np.linalg.LinAlgError('Singular Matrix')
        else:
            ZVals = np.random.normal(loc = muC, scale = np.sqrt( Sig2C ),
                                 size = numSamples)
    return ZVals

def covUpdate(AI,r):
    Sig22I = AI
    Sig11 = 1
    Sig12 = np.array([r])
    SR = Sig22I.dot( Sig12.T )
    den = 1 - Sig12.dot( SR )
    print den
    den = np.array([[0.1]])
    SC = 1.0/den
    B = -SC*SR
    SigI_next = np.vstack((np.hstack((SC, B.T)),
                               np.hstack((B, Sig22I - SR.dot(B.T) )) ))
    return SigI_next

def eigPI(A):
    zer = 1e-08
    d,P = np.linalg.eigh(A)
    dI = 1/d[d>zer]
    nz = len(d)-len(dI)
    Pz = P[:,nz:]
    return Pz.dot((dI*Pz).T)

def maxEntInv(A):
    zer = 1e-08
    d,P = np.linalg.eigh(A)
    d[d<zer] = np.min(d[d > zer])
    #d = np.maximum(d,np.ones(d.shape))
    dI = 1/d
    return P.dot((dI*P).T)

def getCorMatrix(C, rsid, w=100):
    if 2*w > C.shape[1]:
        sys.stderr.write("Not enough correlation information for window.")
        sys.exit()
    loc = C.index.get_loc(rsid)
    begin = np.max((0,loc-w))
    end = np.min((C.shape[0]-1,loc+w))
    lenWin = end-begin
    rpos = loc-begin
    cov = C.iloc[begin:end,0:lenWin].as_matrix().copy()
    A = np.zeros((len(cov),len(cov)))
    A[np.tril_indices(len(cov),-1)] = \
        np.concatenate([cov[i,0:i][::-1] for i in range(len(cov))])
    A +=  np.tril(A).T
    np.fill_diagonal(A,np.ones(len(A)))
    print np.min(np.linalg.eigvalsh(A))
    sig_i = np.delete(A[rpos],rpos)
    Sig = remove_ij(A,rpos,rpos)
    return sig_i,Sig

def getZScores(S, rsid, w=100):
    loc = S.index.get_loc(rsid)
    begin = np.max((0,loc-w))
    end = np.min((C.shape[0]-1,loc+w))
    P = S.iloc[begin:end,7].as_matrix()
    B = S.iloc[begin:end,5].as_matrix()
    Chi = np.sqrt(stats.chi2.ppf(1-P,1))
    Chi[np.isinf(Chi)] = 8.4 ## 0
    Z = np.sign(B)*Chi
    return np.delete(Z,w)

def ImputeZScore( Sig, sig_i, Z, lam=0.1):
    W = sig_i.dot(np.linalg.inv((1-lam)*Sig + lam*np.identity(len(Sig))))
    den = np.sqrt( W.dot(0.9*Sig+0.1*np.identity(len(Sig))).dot(W.T) )
    zi = W.dot(Z)
    zc = zi/den
    return zi, zc, 1-stats.chi2.cdf(zc**2,1)

def rep(rsid,lam=0.1):
    sig_i, Sig = getCorMatrix(C,rsid,w=100)
    Z = getZScores(S,rsid,w=100)
    print ImputeZScore( Sig, sig_i, Z, lam)

def remove_ij(x, i, j):
    # Row i and column j divide the array into 4 quadrants
    y = x[:-1,:-1]
    y[:i,j:] = x[:i,j+1:]
    y[i:,:j] = x[i+1:,:j]
    y[i:,j:] = x[i+1:,j+1:]
    return y

def pinv_update(Sig22I, rVals):
    Sig11 = 1
    m=length(rVals)
    Sig12 = np.array([rVals])
        ## Use inverse of Sig22 with Sig12 to get conditional mean and var
    SR = Sig22I.dot( Sig12.T )
    den = 1 - Sig12.dot( SR )
    if den < 1e-8:
        SC = 1.0/den
        B = -SC*SR
        SigI_next = np.vstack((np.hstack((SC, B.T)),
                               np.hstack((B, Sig22I - SR.dot(B.T) )) ))
    else:
        A = np.pinv(Sig22I)
        k=SR
        B=np.vstack((np.identity(m),k.T))
        d=float(1/(1+k.T.dot(k)))
        E=1-d*k.dot(k.T)
        #E=np.linalg.pinv(1+k.dot(k.T))
        R=E.dot(Sig22I.dot(E))
        #R=E.dot(A.dot(E))
        # R=(d**2)*Sig22I
        MI=B.dot(R.dot(B.T))

# Here rows (0,1,2..len(R)) correspond to testing SNPs with r**2
# only above the corresponding value in R (0,0.05,0.1..0.95)
def EZJV(Z1, Z2, r, R):
    mat = np.zeros( (len(R), len(Z1)) )
    try:
        score = (1/(1-r**2))*(Z1**2 + Z2**2 - 2*r*Z1*Z2)
    except ZeroDivisionError:
        score = Z1**2
    mat[ R <= r**2, : ] = score
    return mat

def EZJV2(Z1, Z2, r):
    try:
        score = (1/(1-r**2))*(Z1**2 + Z2**2 - 2*r*Z1*Z2)
    except ZeroDivisionError:
        score = Z1**2
    return score

def JVLM(Z1, Z2, r, R):
    mat = np.zeros((len(R), len(Z1)))
    keep=(r<0.3 and r>-0.3)|((np.sign(Z1)==np.sign(Z2))&(r<=0)|
                             (np.sign(Z1)!=np.sign(Z2))&(r>=0))
    try:
        score = (1/(1-r**2))*(Z1**2 + Z2**2 - 2*r*Z1*Z2)
    except ZeroDivisionError:
        score = Z1**2
    A = np.zeros(len(Z1))
    A[keep]=score[keep]
    mat[ R <= r**2, : ] = score
    return mat

def JVLM2(Z1, Z2, r):
    keep=((np.sign(Z1)==np.sign(Z2))&(r<=0)|(np.sign(Z1)!=np.sign(Z2))&(r>=0))
    try:
        score = (1/(1-r**2))*(Z1**2 + Z2**2 - 2*r*Z1*Z2)
    except ZeroDivisionError:
        score = Z1**2
    A = np.zeros(len(Z1))
    A[keep]=score[keep]
    return A

def EZChi2( Y, snp, Cov, missing ):
    return EZZ( Y, snp, Cov, missing)**2

def EZZ( Y, snp, Cov = None, missing = 'drop' ):
    N = len(Y)
    r = corr( Y, snp, Cov, missing)
    return np.sqrt((N-1))*r

def JV( Yall, snp, s, r, R, resN, X0 = None, missing = 'drop' ):
    mat = np.zeros( (len(R), len(resN)) )
    X = np.vstack((snp,s)).T
    if X0 is not None:
        X = np.hstack( (X0,X) )
    score = -2*(resN - np.array([sm.OLS(Y, X, missing = missing).fit().llf
                                 for Y in Yall]))
    mat[ R <= r**2, : ] = score
    return mat

def XV( Yall, snp, s, r, R, resN, X0 = None, missing = 'drop' ):
    mat = np.zeros( (len(R), len(resN)) )
    X = np.vstack((snp,s)).T
    if X0 is not None:
        X = np.hstack( (X0,X) )
    score = -2*(resN - np.array([sm.OLS(Y, X, missing = missing).fit().llf
                                 for Y in Yall]))
    mat[ R <= r**2, : ] = score
    return mat

# TODO: Implement more than just drop and raise, add corresponding option
# If X0 is supplied, computes the partial correlation between X1 and X2 given X0
def corr( X1, X2, X0=None, missing=None, pheno=None ):
    if missing == 'drop':
        if pheno is not None:
            mask = True - (np.isnan(X1) | np.isnan(X2) |
                           np.isnan(pheno.reshape(pheno.shape[0])))
        else:
            mask = True - (np.isnan(X1) | np.isnan(X2))
        X1 = X1[mask]
        X2 = X2[mask]
    elif missing == 'raise':
        if sum(np.isnan(X1)):
            sys.stderr.write("Missing value detected. Exiting")
            sys.exit(1)
    if X0 is not None:
        # X = np.hstack((X1.reshape((len(X1), 1)),X2.reshape((len(X2),1)),X0))
        # P = np.linalg.pinv(np.corrcoef(X, rowvar = False ))
        # r = -P[0,1]/np.sqrt( P[0,0]*P[1,1] )
        r1 = sm.OLS(X1,X0).fit().resid
        r2 = sm.OLS(X2,X0).fit().resid
        r = np.corrcoef(r1,r2)[0,1]
    else:
        r = np.corrcoef(X1,X2)[0,1]
    return r

def HaplotypeEM( G1, G2, fin, niter = 5 ):
    G1 = np.round(G1)
    G2 = np.round(G2)
    N = len(G1)
    n = np.zeros( 9 ) # = [ n00, n01, n02, n10, n11, n12, n20, n21, n22 ]
    # Get genotype pair counts
    for g1,g2 in zip(G1,G2):
        n[ int( str(int(g1))+str(int(g2)), 3) ] += 1
    # Store haplotype counts with exact solutions
    h = np.array( [n[0], n[1]/2.0, n[3]/2.0, 0, n[1]/2.0, n[2], 0, n[5]/2.0,
                   n[3]/2.0, 0, n[6], n[7]/2.0, 0, n[5]/2.0, n[7]/2.0, n[8] ] )
    p = np.random.random(4)
    fin = p/sum(p)
    f = fin.copy()
    fnew = f.copy()
    for it in range(niter):
        # E-step
        for hap in [['00','11'],['01','10'],['10','01'],['11','00'] ]:
            h[ int( hap[0] + hap[1], 2 ) ] = n[4]*f[int(hap[0],2)]*f[int(hap[1],2)]/(2.0*( f[0]*f[3] + f[1]*f[2] ))
        # M-Step
        for h1 in ['00','01','10','11']:
            fnew[ int(h1,2) ] = sum( [ h[ int(h1+h2,2) ] for h2 in ['00','01','10','11'] ] )/N
        eps = sum( abs(fnew - f) )
        f = fnew.copy()
        print it, eps
    ## Find most probable haplotypes given frequencies.
    H1, H2 = np.array( zip( *[ GetHaplotype( g, f ) for g in zip(G1, G2) ] ) )
    return H1, H2

def GetHaplotype( g, f ):
    if g == (0,0): return ('00','00')
    elif g == (0,1):
        return [('00','01'), ('01','00') ][np.random.choice( 2, p=[0.5, 0.5])]
    elif g == (0,2):
        return ('01','01')
    elif g == (1,0):
        return [('10','00'), ('00','10') ][np.random.choice( 2, p=[0.5, 0.5])]
    elif g == (1,1):
        return [('01','10'), ('10','01'), ('11','00'), ('00','11')][
            np.random.choice(4, p = normVec(
                    np.array([f[1]*f[2], f[1]*f[2], f[0]*f[3], f[0]*f[3]])))]
    elif g == (1,2):
        return [('11','01'), ('01','11')][np.random.choice(2, p=[0.5, 0.5])]
    elif g == (2,0):
        return ('10','10')
    elif g == (2,1):
        return [('10','11'), ('11','10') ][np.random.choice(2, p=[0.5, 0.5])]
    elif g == (2,2):
        return ('11','11')

def normVec(a): return a/sum(a)

def BuildHapTable( Y, H1, H2 ):
    res = pd.DataFrame( np.zeros((2,4)), columns=['00','01','10','11'])
    for y,h1,h2 in zip(Y,H1,H2):
        res.loc[y,h1] += 1
        res.loc[y,h2] += 1
    return res
